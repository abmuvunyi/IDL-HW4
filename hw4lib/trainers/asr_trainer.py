from .base_trainer import BaseTrainer
from typing import Dict, Any, Optional, List, Tuple, Union
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from ..decoding.sequence_generator import SequenceGenerator
from ..utils import create_scheduler, create_optimizer
from ..model import DecoderOnlyTransformer
import torchaudio.functional as aF
import json
import torchmetrics.text as tmt
from torch.utils.data import Subset
import pandas as pd


class ASRTrainer(BaseTrainer):
    """
    ASR (Automatic Speech Recognition) Trainer class that handles:
    1. Training loop with gradient accumulation, mixed precision training, optional CTC
    2. Validation loop for model evaluation
    3. Recognition loops for different decoding strategies
    4. Language model shallow fusion

    Implementation tasks are labeled in the docstring.
    """

    def __init__(self, model, tokenizer, config, run_name, config_file, device=None):
        super().__init__(model, tokenizer, config, run_name, config_file, device)

        # 1) Initialize cross-entropy loss
        pad_id = self.tokenizer.pad_id
        label_smoothing = self.config['loss'].get('label_smoothing', 0.0)
        self.ce_criterion = nn.CrossEntropyLoss(
            ignore_index=pad_id,
            label_smoothing=label_smoothing
        )

        # 2) Optional CTC loss
        self.ctc_weight = self.config['loss'].get('ctc_weight', 0.0)
        self.ctc_criterion = None
        if self.ctc_weight > 0:
            # We'll use pad_id for blank as per instructions
            self.ctc_criterion = nn.CTCLoss(
                blank=pad_id,
                zero_infinity=True
            )

    def _train_epoch(self, dataloader):
        """
        Train for one epoch.

        Returns:
            metrics, last_batch_attn
        """
        self.model.train()
        batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc="[Training ASR]")

        running_ce_loss = 0.0
        running_ctc_loss = 0.0
        running_joint_loss = 0.0
        total_tokens = 0

        last_batch_attn = {}

        # Clear gradients initially
        self.optimizer.zero_grad()

        for i, batch in enumerate(dataloader):
            feats, targets_shifted, targets_golden, feat_lengths, transcript_lengths = batch

            # Move to device
            feats = feats.to(self.device)
            feat_lengths = feat_lengths.to(self.device)
            # SHIFTED/GOLDEN can be None for test set, but we are training => should not be None
            targets_shifted = targets_shifted.to(self.device) if targets_shifted is not None else None
            targets_golden = targets_golden.to(self.device) if targets_golden is not None else None
            transcript_lengths = transcript_lengths.to(self.device) if transcript_lengths is not None else None

            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=(self.scaler is not None)):
                # Forward pass => The model typically uses:
                #   model.forward(feats, padded_targets=targets_shifted, source_lengths=feat_lengths, target_lengths=transcript_lengths)
                #   returns seq_out, attn, ctc_inputs
                seq_out, attn, ctc_inputs = self.model(
                    padded_sources=feats,
                    padded_targets=targets_shifted,
                    source_lengths=feat_lengths,
                    target_lengths=transcript_lengths
                )
                last_batch_attn = attn

                # Compute CE loss
                B, T, C = seq_out.shape
                seq_out_2d = seq_out.view(B * T, C)
                gold_1d = targets_golden.view(B * T)
                ce_loss = self.ce_criterion(seq_out_2d, gold_1d)

                # Optionally compute CTC
                ctc_loss = torch.tensor(0.0, device=feats.device)
                if self.ctc_weight > 0 and ctc_inputs is not None:
                    # ctc_inputs['log_probs'] => shape (T', B, vocab_size)
                    # ctc_inputs['lengths'] => shape (B,)
                    # We must ensure transcripts have no SOS/EOS. Possibly remove them.
                    # If your transcripts have shape (B, Ttext) with an eos at end, remove it:
                    # (Alternatively your model might do so already.)
                    # For demonstration, we'll assume golden excludes final EOS:
                    ctc_log_probs = ctc_inputs['log_probs']  # (T', B, C)
                    ctc_lengths = ctc_inputs['lengths']  # (B,)

                    # The target for ctc is basically the original transcript (without SOS shift)
                    # If your 'golden' includes an eos token, remove it
                    # We also need the target lengths for ctc => transcript_lengths minus 1 if you appended EOS
                    # For simplicity, let's assume it is already correct
                    ctc_loss = self.ctc_criterion(
                        ctc_log_probs,  # (T', B, C)
                        targets_golden,  # shape (B, Ttext) flattened?
                        ctc_lengths,
                        transcript_lengths
                    )

                # Combine
                loss = ce_loss + self.ctc_weight * ctc_loss

            # Weighted average for logging
            batch_tokens = transcript_lengths.sum().item()
            total_tokens += batch_tokens
            running_ce_loss += ce_loss.item() * batch_tokens
            if self.ctc_weight > 0:
                running_ctc_loss += ctc_loss.item() * batch_tokens
            running_joint_loss += loss.item() * batch_tokens

            # Divide for gradient accumulation
            loss = loss / self.config['training']['gradient_accumulation_steps']

            # Backprop
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Step if needed
            if (i + 1) % self.config['training']['gradient_accumulation_steps'] == 0:
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step()
                    self.scaler.update()
                else:
                    self.optimizer.step()
                    if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step()

                self.optimizer.zero_grad()

            # Update progress bar
            avg_ce_loss = running_ce_loss / total_tokens
            avg_ctc_loss = running_ctc_loss / total_tokens
            avg_joint_loss = running_joint_loss / total_tokens
            perplexity = torch.exp(torch.tensor(avg_ce_loss))

            batch_bar.set_postfix(
                ce_loss=f"{avg_ce_loss:.4f}",
                ctc_loss=f"{avg_ctc_loss:.4f}",
                joint_loss=f"{avg_joint_loss:.4f}",
                perplexity=f"{perplexity:.4f}",
                acc_step=f"{(i % self.config['training']['gradient_accumulation_steps']) + 1}/"
                         f"{self.config['training']['gradient_accumulation_steps']}"
            )
            batch_bar.update()

            # Cleanup
            del feats, targets_shifted, targets_golden, feat_lengths, transcript_lengths
            del seq_out, ctc_inputs, loss
            torch.cuda.empty_cache()

        # leftover accumulation
        remainder = len(dataloader) % self.config['training']['gradient_accumulation_steps']
        if remainder != 0:
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step()
                self.scaler.update()
            else:
                self.optimizer.step()
                if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step()
            self.optimizer.zero_grad()

        avg_ce_loss = running_ce_loss / total_tokens
        avg_ctc_loss = running_ctc_loss / total_tokens
        avg_joint_loss = running_joint_loss / total_tokens
        avg_perplexity_token = torch.exp(torch.tensor(avg_ce_loss))
        # char-level perplexity
        avg_perplexity_char = torch.exp(torch.tensor(avg_ce_loss / dataloader.dataset.get_avg_chars_per_token()))
        batch_bar.close()

        metrics = {
            'ce_loss': avg_ce_loss,
            'ctc_loss': avg_ctc_loss,
            'joint_loss': avg_joint_loss,
            'perplexity_token': avg_perplexity_token.item(),
            'perplexity_char': avg_perplexity_char.item()
        }

        return metrics, last_batch_attn

    def recognize(self, dataloader, recognition_config: Optional[Dict[str, Any]] = None,
                  config_name: Optional[str] = None, max_length: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Evaluate the model by generating transcriptions from audio features.
        If references exist, store them too.
        """
        if max_length is None and not hasattr(self, 'text_max_len'):
            raise ValueError("text_max_len is not set. Please run training loop first or provide max_length")

        if recognition_config is None:
            # Default config -> greedy
            recognition_config = {
                'num_batches': None,
                'beam_width': 1,
                'temperature': 1.0,
                'repeat_penalty': 1.0,
                'lm_weight': 0.0,
                'lm_model': None
            }
            config_name = "greedy"

        # if shallow fusion LM is given
        if recognition_config.get('lm_model') is not None:
            recognition_config['lm_model'].eval()
            recognition_config['lm_model'].to(self.device)

        # We'll use the model in eval mode
        self.model.eval()
        results = []

        # Create progress bar
        desc_str = f"[Recognizing ASR]: {config_name}" if config_name else "[Recognizing ASR]"
        batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc=desc_str)

        # We'll create a SequenceGenerator with a dynamic score_fn
        generator = SequenceGenerator(
            score_fn=None,  # We'll define it for each batch
            tokenizer=self.tokenizer,
            max_length=max_length if max_length is not None else self.text_max_len,
            device=self.device
        )

        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                feats, _, targets_golden, feat_lengths, _ = batch
                feats = feats.to(self.device)
                feat_lengths = feat_lengths.to(self.device)
                if targets_golden is not None:
                    targets_golden = targets_golden.to(self.device)

                # Encode
                # The model typically does => self.model.encode(...)
                # returns => encoder_output, pad_mask_src, attn, ctc_inputs
                encoder_output, pad_mask_src, _, _ = self.model.encode(
                    padded_sources=feats,
                    source_lengths=feat_lengths
                )

                # define scoring function
                def get_score(prompts):
                    asr_logits = self.model.score(prompts, encoder_output, pad_mask_src)
                    if recognition_config['lm_model'] is not None and recognition_config['lm_weight'] > 0.0:
                        lm_logits = recognition_config['lm_model'].score(prompts)
                        return asr_logits + recognition_config['lm_weight'] * lm_logits
                    return asr_logits

                generator.score_fn = get_score

                # build initial prompts
                B = feats.size(0)
                # We'll create batch of shape (B, 1) each with [SOS]
                sos_tok = torch.full((B, 1), fill_value=self.tokenizer.sos_id, device=self.device, dtype=torch.long)

                # decode
                if recognition_config['beam_width'] > 1:
                    # beam search
                    seqs, scores = generator.generate_beam(
                        x=sos_tok,
                        beam_width=recognition_config['beam_width'],
                        temperature=recognition_config['temperature'],
                        repeat_penalty=recognition_config['repeat_penalty']
                    )
                    # beam => shape (B, beam_width, T)
                    # pick best beam => [0]
                    seqs = seqs[:, 0, :]
                    scores = scores[:, 0]
                else:
                    # greedy
                    seqs, scores = generator.generate_greedy(
                        x=sos_tok,
                        temperature=recognition_config['temperature'],
                        repeat_penalty=recognition_config['repeat_penalty']
                    )

                # post-process => remove everything after EOS
                post_processed_preds = generator.post_process_sequence(seqs, self.tokenizer)

                # store results
                if targets_golden is not None:
                    # post process references too
                    post_processed_refs = generator.post_process_sequence(targets_golden, self.tokenizer)
                    for j, (pred, ref) in enumerate(zip(post_processed_preds, post_processed_refs)):
                        results.append({
                            'target': self.tokenizer.decode(ref.tolist(), skip_special_tokens=True),
                            'generated': self.tokenizer.decode(pred.tolist(), skip_special_tokens=True),
                            'score': scores[j].item() if isinstance(scores[j], torch.Tensor) else scores[j]
                        })
                else:
                    # no references
                    for j, pred in enumerate(post_processed_preds):
                        results.append({
                            'generated': self.tokenizer.decode(pred.tolist(), skip_special_tokens=True),
                            'score': scores[j].item() if isinstance(scores[j], torch.Tensor) else scores[j]
                        })

                batch_bar.update()

                # if user-provided 'num_batches' => stop early
                if recognition_config['num_batches'] is not None:
                    if i >= recognition_config['num_batches'] - 1:
                        break

                del feats, feat_lengths, encoder_output, pad_mask_src, seqs, scores
                torch.cuda.empty_cache()

        batch_bar.close()
        return results

    def _validate_epoch(self, dataloader):
        """
        Validate for one epoch => we do recognition and compute CER/WER, etc.
        """
        self.model.eval()
        # Let's do an inference => produce predicted transcriptions
        val_config = {
            'num_batches': None,  # or small if you want partial
            'beam_width': 1,  # let's do greedy by default
            'temperature': 1.0,
            'repeat_penalty': 1.0,
            'lm_weight': 0.0,
            'lm_model': None
        }
        results = self.recognize(dataloader, recognition_config=val_config, config_name="val-greedy")

        # Extract references/hypotheses
        references = []
        hypotheses = []
        for r in results:
            if 'target' in r:
                references.append(r['target'])
            else:
                references.append("")
            hypotheses.append(r['generated'])

        # calculate metrics with e.g. torchmetrics
        wer_metric = tmt.WordErrorRate()
        cer_metric = tmt.CharErrorRate()
        wdist_metric = tmt.EditDistance(reduction='mean')  # for word-level distance

        # measure
        wer_val = wer_metric(hypotheses, references).item() * 100
        cer_val = cer_metric(hypotheses, references).item() * 100
        wdist = wdist_metric(hypotheses, references).item()

        metrics = {
            'wer': wer_val,
            'cer': cer_val,
            'word_dist': wdist
        }
        return metrics, results

    def train(self, train_dataloader, val_dataloader, epochs: int):
        """
        Full training loop for ASR.
        """
        if self.scheduler is None:
            raise ValueError("Scheduler not initialized!")
        if self.optimizer is None:
            raise ValueError("Optimizer not initialized!")

        # We'll pick a max transcript length for generation
        # E.g. from the dataset
        self.text_max_len = max(
            getattr(train_dataloader.dataset, 'text_max_len', 200),
            getattr(val_dataloader.dataset, 'text_max_len', 200)
        )

        best_val_cer = float('inf')

        for epoch in range(self.current_epoch, self.current_epoch + epochs):
            print(f"\n=== [Epoch {epoch}] Training ===")
            train_metrics, train_attn = self._train_epoch(train_dataloader)
            print(f"Train metrics: {train_metrics}")

            print(f"\n=== [Epoch {epoch}] Validating ===")
            val_metrics, val_results = self._validate_epoch(val_dataloader)
            print(f"Val metrics: {val_metrics}")

            # Step scheduler if it's a reduce-lr type
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics['cer'])

            # Log everything
            metrics = {
                'train': train_metrics,
                'val': val_metrics
            }
            self._log_metrics(metrics, epoch)

            # We can store attention from train for debugging
            if train_attn:
                attn_keys = list(train_attn.keys())
                if attn_keys:
                    # e.g. pick the first or last for plotting
                    key = attn_keys[0]
                    # self._save_attention_plot(train_attn[key][0], epoch, "train_self")

            # Save some recognized text
            self._save_generated_text(val_results, f'val_epoch_{epoch}')

            # Save checkpoint
            self.save_checkpoint('checkpoint-last-epoch-model.pth')

            # Check for best CER
            cer_val = val_metrics['cer']
            if cer_val < best_val_cer:
                best_val_cer = cer_val
                self.best_metric = cer_val
                self.save_checkpoint('checkpoint-best-metric-model.pth')

            self.current_epoch += 1

    def evaluate(self, dataloader, max_length: Optional[int] = None) -> Dict[str, Dict[str, float]]:
        """
        Evaluate on test set. We'll gather all recognition configs, run them, store results
        as dataframes or metrics.
        """
        recognition_configs = self._get_evaluation_recognition_configs()
        eval_results = {}

        for config_name, cfg in recognition_configs.items():
            print(f"Evaluating with {config_name} config")
            try:
                results = self.recognize(dataloader, cfg, config_name, max_length)
                # We just store them. If you want metrics, you can do references/hypotheses again
                # But test might not have references
                # Let's build a DF
                generated = [r['generated'] for r in results]
                df = pd.DataFrame({
                    'id': range(len(generated)),
                    'transcription': generated
                })
                eval_results[config_name] = df
                self._save_generated_text(results, f'test_{config_name}_results')
            except Exception as e:
                print(f"Error in {config_name} config: {e}")
                continue

        return eval_results

    def _get_evaluation_recognition_configs(self, lm_model: Optional[DecoderOnlyTransformer] = None,
                                            lm_weight: float = 0.0):
        """
        Return a dictionary of recognition configs for test-time eval.
        We'll do e.g. greedy, beam10, beam20
        """
        common_cfg = {
            'num_batches': None,
            'temperature': 1.0,
            'repeat_penalty': 1.0,
            'lm_weight': lm_weight,
            'lm_model': lm_model
        }
        beam10_cfg = common_cfg.copy()
        beam10_cfg.update({
            'beam_width': 10
        })
        beam20_cfg = common_cfg.copy()
        beam20_cfg.update({
            'beam_width': 20
        })
        greedy_cfg = common_cfg.copy()
        greedy_cfg.update({
            'beam_width': 1
        })
        return {
            'greedy': greedy_cfg,
            'beam_10': beam10_cfg,
            'beam_20': beam20_cfg
        }

    def _calculate_asr_metrics(self, references, hypotheses):
        """
        Calculate WER, CER, word_edit_distance
        references, hypotheses => lists of strings
        """
        wer_metric = tmt.WordErrorRate()  # returns fraction
        cer_metric = tmt.CharErrorRate()  # fraction
        wdist_metric = tmt.EditDistance(reduction='mean')

        wer_val = wer_metric(hypotheses, references).item() * 100
        cer_val = cer_metric(hypotheses, references).item() * 100
        wdist = wdist_metric(hypotheses, references).item()
        return {
            'wer': wer_val,
            'cer': cer_val,
            'word_dist': wdist
        }
