import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from ..decoding.sequence_generator import SequenceGenerator
from ..utils import create_scheduler, create_optimizer
from ..model import DecoderOnlyTransformer
import torchaudio.functional as aF
import json
import torchmetrics.text as tmt
from torch.utils.data import Subset
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Union

from .base_trainer import BaseTrainer


class ASRTrainer(BaseTrainer):
    """
    ASR (Automatic Speech Recognition) Trainer class that handles training, validation, and recognition loops.

    Implementation Tasks:
    - Initialize CE and CTC loss in __init__
    - Implement key parts of the training loop in _train_epoch
    - Implement recognition functionality in recognize
    - Implement key parts of the validation loop in _validate_epoch
    - Implement key parts of the full training loop in train
    """

    def __init__(self, model, tokenizer, config, run_name, config_file, device=None):
        super().__init__(model, tokenizer, config, run_name, config_file, device)

        # ----------------------------------------------------------------------------
        # TODO: Implement the __init__ method
        # ----------------------------------------------------------------------------

        # 1) Parse config for CE & CTC parameters
        self.pad_id = self.tokenizer.pad_id
        loss_config = self.config.get('loss', {})
        self.ctc_weight = loss_config.get('ctc_weight', 0.0)
        label_smoothing = loss_config.get('label_smoothing', 0.0)

        # 2) Initialize CE loss
        # Use the pad token as ignore_index
        self.ce_criterion = nn.CrossEntropyLoss(
            ignore_index=self.pad_id,
            label_smoothing=label_smoothing
        )

        # 3) Initialize CTC loss (only if ctc_weight > 0)
        # Again, use pad_id as the blank index
        self.ctc_criterion = None
        if self.ctc_weight > 0:
            self.ctc_criterion = nn.CTCLoss(
                blank=self.pad_id,
                zero_infinity=True
            )

        # 4) Use mixed precision GradScaler for training
        self.scaler = GradScaler()

        # (Optional) You might also want to do:
        # self.optimizer = create_optimizer(...)
        # self.scheduler = create_scheduler(...)
        # but that might be done outside or in your base trainer.

        # Remove the NotImplementedError
        # raise NotImplementedError # remove

    def _train_epoch(self, dataloader):
        """
        Train for one epoch.

        Args:
            dataloader: DataLoader for training data
        Returns:
            Tuple[Dict[str, float], Dict[str, torch.Tensor]]: Training metrics and attention weights
        """
        # ----------------------------------------------------------------------------
        # TODO: In-fill the _train_epoch method
        # ----------------------------------------------------------------------------

        self.model.train()
        batch_bar = tqdm(
            total=len(dataloader),
            dynamic_ncols=True,
            leave=False,
            position=0,
            desc="[Training ASR]"
        )

        running_ce_loss = 0.0
        running_ctc_loss = 0.0
        running_joint_loss = 0.0
        total_tokens = 0

        # Some dictionary for storing the last attention heads (optional)
        running_att = {}

        # Zero gradients initially
        self.optimizer.zero_grad()

        for i, batch in enumerate(dataloader):
            # Unpack batch
            feats, targets_shifted, targets_golden, feat_lengths, transcript_lengths = batch

            # Move to device
            feats = feats.to(self.device)
            feat_lengths = feat_lengths.to(self.device)
            if targets_shifted is not None:
                targets_shifted = targets_shifted.to(self.device)
            if targets_golden is not None:
                targets_golden = targets_golden.to(self.device)
            if transcript_lengths is not None:
                transcript_lengths = transcript_lengths.to(self.device)

            # Forward pass under autocast (mixed precision)
            with autocast(device_type=self.device, dtype=torch.float16):
                # Our model forward => (seq_out, curr_att, ctc_inputs)
                seq_out, curr_att, ctc_inputs = self.model(
                    padded_sources=feats,
                    padded_targets=targets_shifted,
                    source_lengths=feat_lengths,
                    target_lengths=transcript_lengths
                )
                # seq_out: (B, U, num_classes)
                # ctc_inputs: { 'log_probs': (T', B, num_classes), 'lengths': (B,) }
                # curr_att: attention dict

                # Keep track of last attention for debugging/plots
                running_att = curr_att

                # ----- Compute CE loss -----
                if targets_golden is not None:
                    # Flatten outputs => (B*U, num_classes)
                    ce_loss = self.ce_criterion(
                        seq_out.reshape(-1, seq_out.size(-1)),
                        targets_golden.reshape(-1)
                    )
                else:
                    ce_loss = torch.tensor(0.0, device=self.device)

                # ----- Compute CTC loss if applicable -----
                if self.ctc_weight > 0 and targets_golden is not None:
                    ctc_log_probs = ctc_inputs["log_probs"]   # (T', B, num_classes)
                    input_lengths = ctc_inputs["lengths"]     # (B,)

                    # Build a 1D target for all batch items
                    target_list = []
                    target_len_list = []
                    for b_idx in range(targets_golden.size(0)):
                        length = transcript_lengths[b_idx].item()
                        # slice out the non-pad
                        unpadded = targets_golden[b_idx, :length]
                        target_list.append(unpadded)
                        target_len_list.append(length)

                    targets_1d = torch.cat(target_list)  # shape: sum of lengths
                    target_lengths_1d = torch.tensor(target_len_list, device=self.device)

                    ctc_loss = self.ctc_criterion(
                        ctc_log_probs,   # (T', B, num_classes)
                        targets_1d,      # (total_tokens,)
                        input_lengths,   # (B,)
                        target_lengths_1d
                    )
                    loss = ce_loss + self.ctc_weight * ctc_loss
                else:
                    ctc_loss = torch.tensor(0.0, device=self.device)
                    loss = ce_loss

            # Calculate metrics
            # number of tokens used for weighting
            if transcript_lengths is not None:
                batch_tokens = transcript_lengths.sum().item()
            else:
                batch_tokens = 1  # fallback if no transcripts
            total_tokens += batch_tokens

            running_ce_loss += ce_loss.item() * batch_tokens
            running_ctc_loss += ctc_loss.item() * batch_tokens
            running_joint_loss += loss.item() * batch_tokens

            # Scale the loss by gradient accumulation steps
            accum_steps = self.config['training'].get('gradient_accumulation_steps', 1)
            scaled_loss = loss / accum_steps

            # Backprop with scaler
            self.scaler.scale(scaled_loss).backward()

            # Only step optimizer after enough accumulation
            if (i + 1) % accum_steps == 0:
                self.scaler.step(self.optimizer)
                if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step()
                self.scaler.update()
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
                acc_step=f"{(i % accum_steps) + 1}/{accum_steps}"
            )
            batch_bar.update()

            # Cleanup
            del feats, targets_shifted, targets_golden, feat_lengths, transcript_lengths
            del seq_out, curr_att, ctc_inputs, loss
            torch.cuda.empty_cache()

        # Handle leftover grads if the dataloader size isn't divisible by accum_steps
        if (len(dataloader) % accum_steps) != 0:
            self.scaler.step(self.optimizer)
            if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()
            self.scaler.update()
            self.optimizer.zero_grad()

        batch_bar.close()

        # Compute final metrics
        avg_ce_loss = running_ce_loss / total_tokens if total_tokens > 0 else 0
        avg_ctc_loss = running_ctc_loss / total_tokens if total_tokens > 0 else 0
        avg_joint_loss = running_joint_loss / total_tokens if total_tokens > 0 else 0

        avg_perplexity_token = torch.exp(torch.tensor(avg_ce_loss)) if total_tokens > 0 else 1.0
        # Convert from token-level to char-level if your dataset has a ratio
        chars_per_token = dataloader.dataset.get_avg_chars_per_token()
        if chars_per_token == 0:
            avg_perplexity_char = avg_perplexity_token
        else:
            avg_perplexity_char = torch.exp(torch.tensor(avg_ce_loss / chars_per_token))

        metrics = {
            'ce_loss': float(avg_ce_loss),
            'ctc_loss': float(avg_ctc_loss),
            'joint_loss': float(avg_joint_loss),
            'perplexity_token': float(avg_perplexity_token.item()),
            'perplexity_char': float(avg_perplexity_char.item())
        }

        return metrics, running_att

    def _validate_epoch(self, dataloader):
        """
        Validate for one epoch.

        Args:
            dataloader: DataLoader for validation data
        Returns:
            Tuple[Dict[str, float], List[Dict[str, Any]]]: Validation metrics and recognition results
        """
        # ----------------------------------------------------------------------------
        # TODO: In-fill the _validate_epoch method
        # ----------------------------------------------------------------------------

        self.model.eval()

        # 1) Use your .recognize(...) method to get results for the entire dataloader
        #    Typically we do 'greedy' decode or a small beam here.
        #    We'll do a single pass for the entire set to measure CER/WER.

        # For large dev sets, you might decode in smaller chunks or use fewer batches for speed.
        # For the example, we’ll decode all:

        results = self.recognize(
            dataloader,
            recognition_config={
                'num_batches': None,   # means decode entire loader
                'beam_width': 1,      # do greedy
                'temperature': 1.0,
                'repeat_penalty': 1.0,
                'lm_weight': 0.0,
                'lm_model': None
            },
            config_name='val',
            max_length=self.text_max_len
        )

        # 2) Extract references & hypotheses
        references, hypotheses = [], []
        for r in results:
            if 'target' in r:  # If we have ground-truth
                references.append(r['target'])
            else:
                references.append("")  # or skip
            hypotheses.append(r['generated'])

        # 3) Calculate metrics
        metrics = self._calculate_asr_metrics(references, hypotheses)

        # This method returns metrics plus the raw decode results if you want to log them
        return metrics, results

    def train(self, train_dataloader, val_dataloader, epochs: int):
        """
        Full training loop for ASR training.

        Args:
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            epochs: int, number of epochs to train
        """
        if self.scheduler is None:
            raise ValueError("Scheduler is not initialized, initialize it first!")
        if self.optimizer is None:
            raise ValueError("Optimizer is not initialized, initialize it first!")

        # ----------------------------------------------------------------------------
        # TODO: In-fill the train method
        # ----------------------------------------------------------------------------

        # Set max transcript length if needed
        self.text_max_len = max(
            val_dataloader.dataset.text_max_len,
            train_dataloader.dataset.text_max_len
        )

        best_val_loss = float('inf')
        best_val_wer = float('inf')
        best_val_cer = float('inf')
        best_val_dist = float('inf')  # you might store word_dist or similar

        for epoch in range(self.current_epoch, self.current_epoch + epochs):
            # 1) Train for one epoch
            train_metrics, train_attn = self._train_epoch(train_dataloader)

            # 2) Validate
            val_metrics, val_results = self._validate_epoch(val_dataloader)

            # If using ReduceLROnPlateau, step with some validation metric
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics['cer'])  # or 'wer', or something

            # Combine to log
            metrics = {
                'train': train_metrics,
                'val': val_metrics
            }
            self._log_metrics(metrics, epoch)

            # Optionally plot or save attention
            train_attn_keys = list(train_attn.keys()) if train_attn else []
            if train_attn_keys:
                # Just pick one for illustration
                first_key = train_attn_keys[0]
                self._save_attention_plot(train_attn[first_key][0], epoch, "decoder_self")

            # Save partial results
            self._save_generated_text(val_results, f'val_epoch_{epoch}')

            # Save a "last epoch" checkpoint
            self.save_checkpoint('checkpoint-last-epoch-model.pth')

            # Check for improved CER
            if val_metrics['cer'] < best_val_cer:
                best_val_cer = val_metrics['cer']
                self.best_metric = best_val_cer
                self.save_checkpoint('checkpoint-best-metric-model.pth')

            self.current_epoch += 1

    def recognize(self, dataloader, recognition_config: Optional[Dict[str, Any]] = None,
                  config_name: Optional[str] = None, max_length: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Evaluate the model by generating transcriptions from audio features.

        Args:
            dataloader: DataLoader containing the evaluation data
            recognition_config: Optional dictionary containing decode params
            config_name: Optional name for logging
            max_length: Optional maximum length of the generated sequence
        Returns:
            List of dicts with {'target':..., 'generated':..., 'score':...}
        """

        if max_length is None and not hasattr(self, 'text_max_len'):
            raise ValueError("text_max_len is not set. Provide a max_length or run training first.")

        if recognition_config is None:
            recognition_config = {
                'num_batches': None,
                'beam_width': 1,
                'temperature': 1.0,
                'repeat_penalty': 1.0,
                'lm_weight': 0.0,
                'lm_model': None
            }
            config_name = config_name or 'greedy'

        # If shallow fusion with LM
        lm_model = recognition_config.get('lm_model', None)
        lm_weight = recognition_config.get('lm_weight', 0.0)
        if lm_model is not None:
            lm_model.eval()
            lm_model.to(self.device)

        # Prepare a SequenceGenerator
        generator = SequenceGenerator(
            score_fn=None,  # We'll define it dynamically per batch
            tokenizer=self.tokenizer,
            max_length=max_length if max_length else self.text_max_len,
            device=self.device
        )

        beam_width = recognition_config.get('beam_width', 1)
        temperature = recognition_config.get('temperature', 1.0)
        repeat_penalty = recognition_config.get('repeat_penalty', 1.0)
        num_batches = recognition_config.get('num_batches', None)

        results = []
        self.model.eval()
        batch_bar = tqdm(
            total=len(dataloader),
            dynamic_ncols=True,
            leave=False,
            position=0,
            desc=f"[Recognizing ASR] : {config_name}"
        )

        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                feats, _, targets_golden, feat_lengths, _ = batch

                # Move to device
                feats = feats.to(self.device)
                feat_lengths = feat_lengths.to(self.device)
                if targets_golden is not None:
                    targets_golden = targets_golden.to(self.device)

                # 1) Encode
                encoder_output, pad_mask_src, _, _ = self.model.encode(feats, feat_lengths)

                # 2) Build a scoring fn
                def get_score(batch_prompts: torch.Tensor) -> torch.Tensor:
                    # shape => (B, seq_len)
                    asr_logits = self.model.score(batch_prompts, encoder_output, pad_mask_src)
                    # If shallow fusion is used
                    if lm_model is not None:
                        lm_logits = lm_model.score(batch_prompts)
                        return asr_logits + lm_weight * lm_logits
                    return asr_logits

                generator.score_fn = get_score

                # 3) Initialize prompts (SOS token)
                batch_size = feats.size(0)
                prompts = torch.full(
                    (batch_size, 1),
                    self.tokenizer.sos_id,
                    dtype=torch.long,
                    device=self.device
                )

                # 4) Decode
                if beam_width > 1:
                    seqs, scores = generator.generate_beam(
                        prompts,
                        beam_width=beam_width,
                        temperature=temperature,
                        repeat_penalty=repeat_penalty
                    )
                    # pick best beam => (B, seq_len)
                    seqs = seqs[:, 0, :]
                    scores = scores[:, 0]
                else:
                    seqs, scores = generator.generate_greedy(
                        prompts,
                        temperature=temperature,
                        repeat_penalty=repeat_penalty
                    )

                # 5) Post-process sequences
                post_processed_preds = generator.post_process_sequence(seqs, self.tokenizer)

                # 6) Store results
                if targets_golden is not None:
                    # post-process targets to strip extra pads, etc.
                    target_proc = generator.post_process_sequence(targets_golden, self.tokenizer)
                    for j, (pred, tgt) in enumerate(zip(post_processed_preds, target_proc)):
                        results.append({
                            'target': self.tokenizer.decode(tgt.tolist(), skip_special_tokens=True),
                            'generated': self.tokenizer.decode(pred.tolist(), skip_special_tokens=True),
                            'score': scores[j].item()
                        })
                else:
                    for j, pred in enumerate(post_processed_preds):
                        results.append({
                            'generated': self.tokenizer.decode(pred.tolist(), skip_special_tokens=True),
                            'score': scores[j].item()
                        })

                batch_bar.update()

                # If user wants to limit #batches
                if num_batches is not None and i >= num_batches - 1:
                    break

                # Cleanup
                del feats, feat_lengths, encoder_output, pad_mask_src, prompts, seqs, scores
                torch.cuda.empty_cache()

        batch_bar.close()
        return results
