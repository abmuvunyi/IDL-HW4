import torch
import torch.nn as nn
from typing import Tuple, Optional, List, Callable
from ..data import H4Tokenizer


class SequenceGenerator:
    """
    A class for generating sequences using various decoding strategies:
    - Greedy Search
    - Beam Search
    - Sampling with top-k / top-p (already in your snippet)
    """

    def __init__(
            self,
            score_fn: Callable,
            tokenizer: H4Tokenizer,
            max_length: int,
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the sequence generator.

        Args:
            score_fn: A function that takes the current tokens (shape: (batch_size, seq_len))
                      and returns logits for the next token (shape: (batch_size, vocab_size)).
            tokenizer: Tokenizer for ID-to-token conversions, which includes .eos_id.
            max_length: Maximum total sequence length to generate.
            device: Device to run generation on.
        """
        self.score_fn = score_fn
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    def _apply_repeat_penalty(
            self,
            logits: torch.Tensor,
            sequences: torch.Tensor,
            penalty: float = 1.0
    ) -> torch.Tensor:
        """
        Apply repetition penalty to logits based on tokens in sequences.
        This reduces the probability of tokens that have already appeared in the sequence
        if penalty > 1.0. For tokens that have appeared, we divide their logits by 'penalty'
        if the logit > 0, or multiply by penalty if the logit < 0.

        Args:
            logits: shape (batch_size, vocab_size) or (batch_size, beam_width, vocab_size)
            sequences: shape (batch_size, seq_len) or (batch_size, beam_width, seq_len)
            penalty: Repetition penalty value (>1 means more penalty)

        Returns:
            logits with repetition penalty applied in-place
        """
        if penalty == 1.0:
            return logits

        if logits.dim() == 2:
            # shape => (batch_size, vocab_size)
            batch_size = logits.size(0)
            for b in range(batch_size):
                unique_tokens = torch.unique(sequences[b])
                # For each token that appears, apply penalty
                for t in unique_tokens:
                    if logits[b, t] > 0:
                        logits[b, t] = logits[b, t] / penalty
                    else:
                        logits[b, t] = logits[b, t] * penalty
        else:
            # shape => (batch_size, beam_width, vocab_size)
            bsz, beam_w, vocab_sz = logits.shape
            for b in range(bsz):
                for beam_idx in range(beam_w):
                    unique_tokens = torch.unique(sequences[b, beam_idx])
                    for t in unique_tokens:
                        if logits[b, beam_idx, t] > 0:
                            logits[b, beam_idx, t] = logits[b, beam_idx, t] / penalty
                        else:
                            logits[b, beam_idx, t] = logits[b, beam_idx, t] * penalty

        return logits

    def _filter_logits(
            self,
            logits: torch.Tensor,
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 1.0
    ) -> torch.Tensor:
        """
        Apply temperature, top-k, and top-p filtering to logits (for sampling).
        You already have this code in your snippet, included here for completeness.
        """
        logits = logits / temperature

        if top_k > 0:
            top_k_logits, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            indices_to_remove = logits < top_k_logits[..., -1:]
            logits[indices_to_remove] = float('-inf')

        if top_p < 1.0:
            log_probs = torch.log_softmax(logits, dim=-1)
            sorted_log_probs, sorted_indices = torch.sort(log_probs, descending=True)
            cumulative_probs = torch.cumsum(torch.exp(sorted_log_probs), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift everything one step to the right
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')

        return logits

    def generate_greedy(
            self,
            x: torch.Tensor,
            temperature: float = 1.0,
            repeat_penalty: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using greedy search.

        Args:
            x: shape (batch_size, seq_len) - starting tokens
            temperature: float, >0
            repeat_penalty: float, >=1.0 means stronger penalty for repeated tokens

        Returns:
            sequences: (batch_size, final_seq_len)
            scores: (batch_size,) log probability sums
        """
        # Basic checks
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Input x must be 2D (batch_size, seq_len)")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")

        batch_size = x.size(0)
        scores = torch.zeros(batch_size, device=x.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
        seq_len = x.size(1)

        # Greedy decoding
        for _ in range(self.max_length - seq_len):
            if finished.all():
                break

            # 1) Get next-token logits
            logits = self.score_fn(x)  # shape (B, vocab_size)

            # 2) Apply repeat penalty
            logits = self._apply_repeat_penalty(logits, x, repeat_penalty)

            # 3) Temperature scaling + choose highest logit
            logits = logits / temperature

            log_probs = torch.log_softmax(logits, dim=-1)
            next_tokens = torch.argmax(log_probs, dim=-1)  # shape (B,)
            token_scores = log_probs.gather(1, next_tokens.unsqueeze(1)).squeeze(1)

            # 4) Update scores only for unfinished sequences
            scores = torch.where(finished, scores, scores + token_scores)

            # 5) Append next tokens
            x = torch.cat([x, next_tokens.unsqueeze(1)], dim=1)

            # 6) Check for EOS
            is_eos = (next_tokens == self.tokenizer.eos_id)
            finished = finished | is_eos

        return x, scores

    def generate_beam(
            self,
            x: torch.Tensor,
            beam_width: int,
            temperature: float = 1.0,
            repeat_penalty: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using beam search.

        Args:
            x: shape (batch_size, seq_len) - starting tokens
            beam_width: int, how many beams to keep
            temperature: scaling factor
            repeat_penalty: repetition penalty factor

        Returns:
            sequences: shape (batch_size, beam_width, final_seq_len)
            scores: shape (batch_size, beam_width)  (log-prob sums)
        """
        # Basic checks
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Input x must be 2D (batch_size, seq_len)")
        if beam_width < 1:
            raise ValueError("beam_width must be >= 1")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")

        batch_size = x.size(0)
        seq_len = x.size(1)

        # We'll keep track of beams in "sequences" shape => (B, beam_w, seq_len_so_far)
        # as well as "scores" shape => (B, beam_w)
        # Start by replicating the initial prompt across beams:
        sequences = x.unsqueeze(1).repeat(1, beam_width, 1)  # (B, beam_w, seq_len)
        scores = torch.zeros(batch_size, beam_width, device=x.device)  # log-probs
        finished = torch.zeros(batch_size, beam_width, dtype=torch.bool, device=x.device)

        # We track "alive" beams for each batch item (some beams might be finished)
        for _ in range(self.max_length - seq_len):
            if finished.all():
                break

            # (B, beam_w, seq_len) => we flatten beams for scoring => (B*beam_w, seq_len)
            # flat_seqs = sequences.view(batch_size * beam_width, -1)

            # 1) get next-token logits => shape (B*beam_w, vocab_size)
            # logits = self.score_fn(flat_seqs)
            # score_fn must be called separately per batch item, each with (beam_width, seq_len)
            logits = []
            for b in range(batch_size):
                logits_b = self.score_fn(sequences[b])  # shape: (beam_width, vocab_size)
                logits.append(logits_b.unsqueeze(0))
            logits = torch.cat(logits, dim=0)  # shape: (batch_size, beam_width, vocab_size)

            # 2) apply repetition penalty
            logits = self._apply_repeat_penalty(
                logits.view(batch_size, beam_width, -1),
                sequences,
                penalty=repeat_penalty
            ).view(batch_size * beam_width, -1)

            # 3) temperature scaling
            logits = logits / temperature

            log_probs = torch.log_softmax(logits, dim=-1)  # (B*beam_w, vocab_size)

            # 4) expand dimension => (B, beam_w, vocab_size)
            vocab_size = log_probs.size(-1)
            log_probs = log_probs.view(batch_size, beam_width, vocab_size)

            # 5) combine old beam scores with new log_probs
            # shape => (B, beam_w, vocab_size)
            expanded_scores = scores.unsqueeze(-1) + log_probs  # (B, beam_w, vocab_size)

            # 6) For beams that are finished, keep them unchanged
            # If beam i is finished, we want to ensure that beam stays as-is
            # We'll do that by setting the new scores for those beams to -inf except for the current token
            # But an easier approach is to mask them out of the topk
            inf_mask = finished.unsqueeze(-1).expand_as(expanded_scores)
            expanded_scores[inf_mask] = float('-inf')

            # 7) pick top beam_width from all beam_width*vocab_size
            # shape => (B, beam_w * vocab_size)
            expanded_scores = expanded_scores.view(batch_size, beam_width * vocab_size)
            topk_scores, topk_indices = torch.topk(expanded_scores, k=beam_width, dim=-1)

            # 8) we now map topk_indices back to old beam + token
            beam_indices = topk_indices // vocab_size
            token_indices = topk_indices % vocab_size

            # 9) create new sequences
            new_sequences = []
            new_finished = []
            for b in range(batch_size):
                seq_batch = []
                fin_batch = []
                for beam_i in range(beam_width):
                    old_beam_idx = beam_indices[b, beam_i]
                    token_idx = token_indices[b, beam_i]

                    # gather old sequence
                    old_seq = sequences[b, old_beam_idx]
                    # append the new token
                    new_seq = torch.cat([old_seq, token_idx.unsqueeze(0)], dim=0)
                    seq_batch.append(new_seq)

                    # check if the new token is eos or was previously finished
                    was_finished = finished[b, old_beam_idx]
                    is_eos = (token_idx == self.tokenizer.eos_id)
                    fin_batch.append(was_finished or is_eos)

                new_sequences.append(torch.stack(seq_batch, dim=0))
                new_finished.append(torch.tensor(fin_batch, device=x.device))

            # shape => (B, beam_w, seq_len+1)
            sequences = torch.stack(new_sequences, dim=0)
            finished = torch.stack(new_finished, dim=0)

            # 10) update scores
            scores = topk_scores

        # At the end, sequences => (B, beam_w, final_seq_len), scores => (B, beam_w)
        return sequences, scores

    def generate_sample(
            self,
            x: torch.Tensor,
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using sampling with top-k and nucleus filtering.
        Provided in your snippet, but you can add repeat-penalty if desired.
        """
        # The body is the same as your snippet, but we could add _apply_repeat_penalty if we like.
        # We'll keep your snippet intact for clarity.

        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Input x must be 2-dimensional (batch_size, seq_len)")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not 0 < top_p <= 1.0:
            raise ValueError("top_p must be > 0 and <= 1.0")

        batch_size = x.size(0)
        scores = torch.zeros(batch_size, device=x.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=x.device)

        for _ in range(self.max_length - x.size(1)):
            if finished.all():
                break

            next_scores = self.score_fn(x)  # shape (B, vocab_size)

            # Optionally apply repeat penalty here if you want:
            # next_scores = self._apply_repeat_penalty(next_scores, x, penalty=1.05)

            filtered_logits = self._filter_logits(next_scores, temperature, top_k, top_p)
            log_probs = torch.log_softmax(filtered_logits, dim=-1)

            probs = torch.exp(log_probs)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            token_scores = log_probs.gather(1, next_tokens.unsqueeze(1)).squeeze(1)

            scores = torch.where(finished, scores, scores + token_scores)
            x = torch.cat([x, next_tokens.unsqueeze(1)], dim=1)

            is_eos = (next_tokens == self.tokenizer.eos_id)
            finished = finished | is_eos

        return x, scores

    @staticmethod
    def post_process_sequence(seq: torch.Tensor, tokenizer: H4Tokenizer) -> torch.Tensor:
        """
        Provided in your snippet. We leave it as is.
        """
        if seq.dim() == 1:
            eos_indices = (seq == tokenizer.eos_id).nonzero()
            if len(eos_indices) > 0:
                end_idx = eos_indices[0].item() + 1
                return seq[:end_idx]
            return seq

        eos_mask = seq == tokenizer.eos_id
        eos_indices = eos_mask.float().cumsum(dim=1).eq(1) & eos_mask
        seq_mask = eos_indices.cumsum(dim=1).eq(0) | eos_indices
        return [s[:m.sum()] for s, m in zip(seq, seq_mask)]
