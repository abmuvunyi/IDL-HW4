import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, Tuple, Any, Optional, List
from ..utils import create_scheduler
from ..decoding.sequence_generator import SequenceGenerator
from .base_trainer import BaseTrainer


class LMTrainer(BaseTrainer):
    """
    Language Model Trainer class that handles the training, validation, and generation loops.

    Implementation Notes:
    1. For __init__:
       - Initialize CrossEntropyLoss with appropriate padding index and label smoothing

    2. For _train_epoch:
       - Unpack the batch (shifted inputs, golden targets, lengths)
       - Get model predictions and attention weights
       - Calculate loss
       - Accumulate and update (handle gradient accumulation)

    3. For _validate_epoch:
       - Similar to _train_epoch but without gradient updates
       - Use torch.inference_mode() for validation

    4. For train:
       - Implement the epoch loop with training, validation, generation, checkpointing

    5. For generate:
       - Use the greedy decoding method from SequenceGenerator
       - Post-process sequences
       - Return the generated results
    """

    def __init__(self, model, tokenizer, config, run_name, config_file, device=None):
        super().__init__(model, tokenizer, config, run_name, config_file, device)
        # 1) Initialize the criterion (CrossEntropyLoss) with:
        #    - ignore_index = tokenizer.pad_id (so pad tokens aren't counted in the loss)
        #    - label_smoothing from config['training'].get('label_smoothing', 0.0)
        pad_id = getattr(self.tokenizer, "pad_id", None)
        label_smoothing = self.config["training"].get("label_smoothing", 0.0)
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=pad_id,
            label_smoothing=label_smoothing
        )

    def _train_epoch(self, dataloader) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """
        Train for one epoch.

        Returns:
            metrics, attention_weights
        """
        self.model.train()
        batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc="[Training LM]")
        running_ce_loss = 0.0
        total_tokens = 0

        # Zero gradients at the start
        self.optimizer.zero_grad()

        attn_weights = {}  # We'll store the last batch's attn weights here if needed

        for i, batch in enumerate(dataloader):
            # The batch typically: (shifted, golden, lengths)
            targets_shifted, targets_golden, lengths = batch

            # Move to device
            targets_shifted = targets_shifted.to(self.device, non_blocking=True)
            targets_golden = targets_golden.to(self.device, non_blocking=True)
            lengths = lengths.to(self.device, non_blocking=True)

            # Automatic Mixed Precision context
            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.scaler is not None):

                # Forward pass: model returns logits of shape (B, T, num_classes), plus attention
                raw_preds, attn = self.model(targets_shifted, target_lengths=lengths)
                attn_weights = attn  # Storing last batch's attention for diagnostic

                # Flatten for CE loss
                # raw_preds => (B*T, num_classes), targets_golden => (B*T)
                B, T, C = raw_preds.shape
                raw_preds_2d = raw_preds.view(B * T, C)
                gold_1d = targets_golden.view(B * T)

                # Cross-entropy
                raw_loss = self.criterion(raw_preds_2d, gold_1d)

            # Weighted by number of tokens for average
            batch_tokens = lengths.sum().item()
            total_tokens += batch_tokens
            running_ce_loss += raw_loss.item() * batch_tokens

            # Gradient accumulation
            loss = raw_loss / self.config['training']['gradient_accumulation_steps']

            if self.scaler is not None:
                self.scaler.scale(loss).backward()

                # clipping gradient norm
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.scaler.step(self.optimizer)
                if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step()
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step()

            # Update after enough accumulation
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
            perplexity_token = torch.exp(torch.tensor(avg_ce_loss))
            batch_bar.set_postfix(
                ce_loss_token=f"{avg_ce_loss:.4f}",
                perplexity_token=f"{perplexity_token:.4f}",
                acc_step=f"{(i % self.config['training']['gradient_accumulation_steps']) + 1}/"
                         f"{self.config['training']['gradient_accumulation_steps']}"
            )
            batch_bar.update()

            # Clean up
            del targets_shifted, targets_golden, lengths, raw_preds, loss
            torch.cuda.empty_cache()

        # If the dataloader size isn't a multiple of accumulation steps:
        remainder = len(dataloader) % self.config['training']['gradient_accumulation_steps']
        if remainder != 0:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step()
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step()
            self.optimizer.zero_grad()

        avg_ce_loss = running_ce_loss / total_tokens
        avg_ce_loss_char = avg_ce_loss / dataloader.dataset.get_avg_chars_per_token()
        avg_perplexity_token = torch.exp(torch.tensor(avg_ce_loss))
        avg_perplexity_char = torch.exp(torch.tensor(avg_ce_loss_char))
        batch_bar.close()

        metrics = {
            'ce_loss_token': avg_ce_loss,
            'ce_loss_char': avg_ce_loss_char,
            'perplexity_token': avg_perplexity_token.item(),
            'perplexity_char': avg_perplexity_char.item()
        }

        return metrics, attn_weights

    def _validate_epoch(self, dataloader):
        """
        Validate for one epoch.
        """
        self.model.eval()
        batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc="[Validating LM]")
        running_ce_loss = 0.0
        total_tokens = 0
        attn_weights = {}

        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                targets_shifted, targets_golden, lengths = batch
                targets_shifted = targets_shifted.to(self.device, non_blocking=True)
                targets_golden = targets_golden.to(self.device, non_blocking=True)
                lengths = lengths.to(self.device, non_blocking=True)

                # Forward pass
                raw_preds, attn = self.model(targets_shifted, target_lengths=lengths)
                attn_weights = attn  # Keep last batch's attention weights

                # Flatten for CE
                B, T, C = raw_preds.shape
                raw_preds_2d = raw_preds.view(B * T, C)
                gold_1d = targets_golden.view(B * T)

                # Cross-entropy
                loss = self.criterion(raw_preds_2d, gold_1d)

                batch_tokens = lengths.sum().item()
                total_tokens += batch_tokens
                running_ce_loss += loss.item() * batch_tokens

                # Update progress bar
                avg_ce_loss = running_ce_loss / total_tokens
                perplexity_token = torch.exp(torch.tensor(avg_ce_loss))
                batch_bar.set_postfix(
                    ce_loss_token=f"{avg_ce_loss:.4f}",
                    perplexity_token=f"{perplexity_token:.4f}",
                )
                batch_bar.update()

                # Clean up
                del targets_shifted, targets_golden, lengths, raw_preds, loss
                torch.cuda.empty_cache()

        avg_ce_loss = running_ce_loss / total_tokens
        avg_ce_loss_char = avg_ce_loss / dataloader.dataset.get_avg_chars_per_token()
        avg_perplexity_token = torch.exp(torch.tensor(avg_ce_loss))
        avg_perplexity_char = torch.exp(torch.tensor(avg_ce_loss_char))
        batch_bar.close()

        metrics = {
            'ce_loss_token': avg_ce_loss,
            'ce_loss_char': avg_ce_loss_char,
            'perplexity_token': avg_perplexity_token.item(),
            'perplexity_char': avg_perplexity_char.item()
        }
        return metrics, attn_weights

    def train(self, train_dataloader, val_dataloader, epochs: int):
        """
        Full training loop for language model training.
        """
        if self.scheduler is None:
            raise ValueError("Scheduler is not initialized, initialize it first!")
        if self.optimizer is None:
            raise ValueError("Optimizer is not initialized, initialize it first!")

        best_val_loss = float('inf')

        for epoch in range(self.current_epoch, self.current_epoch + epochs):
            print(f"\n=== [Epoch {epoch}] Training ===")
            train_metrics, train_attn = self._train_epoch(train_dataloader)
            print(f"Train metrics: {train_metrics}")

            print(f"\n=== [Epoch {epoch}] Validating ===")
            val_metrics, val_attn = self._validate_epoch(val_dataloader)
            print(f"Val metrics: {val_metrics}")

            # Optionally generate some text from validation set
            print("\n=== Generating on validation set ===")
            gen_results = self.generate(val_dataloader)
            print(f"Sample generation: {gen_results[:1]}")  # Show first generation result

            # If using a reduce-on-plateau scheduler, step with validation loss
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics['ce_loss_char'])

            # Log metrics
            metrics = {
                'train': train_metrics,
                'val': val_metrics
            }
            self._log_metrics(metrics, epoch)

            # Save some attention plots (just an example with the first layer's attention)
            train_attn_keys = list(train_attn.keys())
            val_attn_keys = list(val_attn.keys())
            if train_attn_keys:
                self._save_attention_plot(train_attn[train_attn_keys[0]][0], epoch, "train_self")
            if val_attn_keys:
                self._save_attention_plot(val_attn[val_attn_keys[0]][0], epoch, "val_self")

            # Save generated text
            self._save_generated_text(gen_results, f'val_epoch_{epoch}')

            # Save checkpoint
            self.save_checkpoint('checkpoint-last-epoch-model.pth')

            # Track best validation
            val_loss = val_metrics['ce_loss_char']
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_metric = val_loss
                self.save_checkpoint('checkpoint-best-metric-model.pth')

            self.current_epoch += 1

    def evaluate(self, test_dataloader):
        """
        Evaluate the model on the test set.
        """
        test_metrics, test_attn = self._validate_epoch(test_dataloader)
        print("[Test] Metrics:", test_metrics)

        # Log metrics
        metrics = {'test': test_metrics}
        self._log_metrics(metrics, self.current_epoch)

        # Optionally save attention plot
        test_attn_keys = list(test_attn.keys())
        if test_attn_keys:
            self._save_attention_plot(test_attn[test_attn_keys[0]][0], self.current_epoch, "test_self")

        # Generate with various configs
        generation_results = {}
        eval_configs = self._get_evaluation_generation_configs()
        for config_name, cfg in eval_configs.items():
            try:
                gen_results = self.generate(test_dataloader, generation_config=cfg)
                generation_results[config_name] = gen_results
                self._save_generated_text(gen_results, f'test_epoch_{self.current_epoch}_{config_name}')
            except Exception as e:
                print(f"Could not generate results for {config_name}: {e}")
        return test_metrics, generation_results

    def generate(self, dataloader, generation_config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Evaluate the model by generating sequences from prompts.
        By default uses greedy search if no config is provided.
        """
        if generation_config is None:
            generation_config = {
                'num_samples': 10,
                'prompt_length': 20,
                'seed': 11785,
                'max_length': self.model.max_len,
                'temperature': 1.0,
                'beam_width': 1,
                'repeat_penalty': 1.0,
                'top_k': 0,
                'top_p': 0.0
            }

        # Create SequenceGenerator
        generator = SequenceGenerator(
            score_fn=lambda x: self.model.score(x),
            tokenizer=self.tokenizer,
            max_length=self.model.max_len,
            device=self.device
        )

        # Sample prompts
        prompts, originals = dataloader.dataset.sample_prompts(
            num_samples=generation_config['num_samples'],
            prompt_length=generation_config['prompt_length'],
            seed=generation_config['seed']
        )
        prompts = prompts.to(self.device)

        self.model.eval()
        with torch.inference_mode():
            # If top_k or top_p -> we do sampling
            if generation_config['top_k'] > 0 or generation_config['top_p'] > 0.0:
                print("Generating with sampling...")
                seqs, scores = generator.generate_sample(
                    x=prompts,
                    temperature=generation_config.get('temperature', 1.0),
                    top_k=generation_config.get('top_k', 0),
                    top_p=generation_config.get('top_p', 1.0)
                )
            # If beam_width > 1 -> beam search
            elif generation_config['beam_width'] > 1:
                print("Generating with beam search...")
                seqs, scores = generator.generate_beam(
                    x=prompts,
                    beam_width=generation_config['beam_width'],
                    temperature=generation_config['temperature'],
                    repeat_penalty=generation_config['repeat_penalty']
                )
                # beam search returns shape (B, beam_width, final_length) and (B, beam_width)
                # let's take best beam in [0]
                seqs = seqs[:, 0, :]
                scores = scores[:, 0]
            else:
                # Default: Greedy
                print("Generating with greedy search...")
                seqs, scores = generator.generate_greedy(
                    x=prompts,
                    temperature=generation_config['temperature'],
                    repeat_penalty=generation_config['repeat_penalty']
                )

        # Post-process
        processed_seqs = generator.post_process_sequence(seqs, self.tokenizer)

        # Build results
        results = []
        for prompt_toks, seq, score, orig_toks in zip(prompts, processed_seqs, scores, originals):
            # prompt, generated, original
            # decode
            prompt_str = self.tokenizer.decode(prompt_toks.tolist())
            # We'll treat 'orig_toks' as a torch tensor of the entire sequence, so we remove the length of the prompt:
            gen_str = self.tokenizer.decode(seq[len(prompt_toks):].tolist())
            original_str = self.tokenizer.decode(orig_toks[len(prompt_toks):].tolist())

            results.append({
                'prompt': prompt_str,
                'original': original_str,
                'generated': gen_str,
                'score': score.item() if isinstance(score, torch.Tensor) else score
            })

        return results

    def _get_evaluation_generation_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a list of generation configurations for evaluation.

        Returns:
            Dictionary containing generation configurations
        """
        common_config = {
            'num_samples': 50,
            'prompt_length': 10,
            'seed': 11785,
            'max_length': self.model.max_len,
        }

        greedy_config = common_config.copy()
        greedy_config.update({
            'temperature': 1.0,
            'beam_width': 1,
            'repeat_penalty': 1.0,
            'top_k': 0,
            'top_p': 0.0
        })

        beam_config = common_config.copy()
        beam_config.update({
            'temperature': 1.0,
            'beam_width': 10,
            'repeat_penalty': 1.2,
            'top_k': 0,
            'top_p': 0.0
        })

        sample_config = common_config.copy()
        sample_config.update({
            'temperature': 1.0,
            'beam_width': 1,
            'repeat_penalty': 1.0,
            'top_k': 10,
            'top_p': 0.95
        })

        return {
            'greedy': greedy_config,
            'beam': beam_config,
            'sample': sample_config
        }