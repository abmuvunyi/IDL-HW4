from typing import Tuple, List
import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from .tokenizer import H4Tokenizer

'''
TODO: Implement this class.

Specification:
- Dataset for training and evaluating language models
- Loads text data from files and tokenizes them
- Handles data subsetting based on configuration
- Creates shifted and golden (target) versions of sequences
- Tracks dataset statistics (chars, tokens, lengths)
- Provides collation function for batching
- Supports random prompt sampling for generation

Key Requirements:
- Each sequence should start with SOS token in shifted version
- Each sequence should end with EOS token in golden version
- Padding should use the designated pad token
- Sequences within a batch must be padded to same length
- Must track character and token counts for perplexity calculation
- Must verify alignment between shifted and golden sequences
'''


class LMDataset(Dataset):
    """
    Dataset for Language Model training/evaluation.
    """

    def __init__(
            self,
            partition: str,
            config: dict,
            tokenizer: H4Tokenizer
    ):
        """
        Initializes the Language Model Dataset for training language models on text data.

        Args:
            partition (str): Data partition subdirectory under root (e.g., 'train', 'test')
            config (dict): Configuration dictionary containing dataset settings
            tokenizer (H4Tokenizer): Tokenizer for encoding/decoding text
        """
        # DO NOT MODIFY
        self.config = config
        self.partition = partition
        self.tokenizer = tokenizer

        # Load special token IDs from the tokenizer
        self.eos_token = self.tokenizer.eos_id
        self.sos_token = self.tokenizer.sos_id
        self.pad_token = self.tokenizer.pad_id

        # Join root and partition to get directory
        self.text_dir = os.path.join(self.config["root"], self.partition)

        # Gather .npy files
        all_files = sorted(os.listdir(self.text_dir))
        self.text_files = [
            os.path.join(self.text_dir, f)
            for f in all_files
            if f.endswith(".npy")
        ]

        # Cast subset_size to int if it's not None
        subset_size = self.config.get("subset", None)
        if subset_size is not None:
            subset_size = int(subset_size)  # <-- fix here
            if subset_size < len(self.text_files):
                self.text_files = self.text_files[:subset_size]

        self.transcripts_shifted = []
        self.transcripts_golden = []

        # Tracking variables (DO NOT MODIFY)
        self.total_chars = 0
        self.total_tokens = 0
        self.text_max_len = 0

        print(f"Loading transcripts for {partition} partition...")
        for file in tqdm(self.text_files):
            loaded_arr = np.load(file, allow_pickle=True)

            if loaded_arr.ndim == 0:
                # Just one string (or single item) stored
                transcript = loaded_arr.item()
            else:
                # An array of strings/chars – join them
                transcript = "".join(list(loaded_arr))

            self.total_chars += len(transcript)

            # Tokenize
            tokenized = self.tokenizer.encode(transcript)

            # Track token count & max length (with 1 extra for SOS/EOS)
            self.total_tokens += len(tokenized)
            self.text_max_len = max(self.text_max_len, len(tokenized) + 1)

            # Build shifted & golden sequences
            shifted_seq = [self.sos_token] + tokenized
            golden_seq = tokenized + [self.eos_token]

            self.transcripts_shifted.append(shifted_seq)
            self.transcripts_golden.append(golden_seq)

        # Average chars per token
        self.avg_chars_per_token = (
            self.total_chars / self.total_tokens if self.total_tokens > 0 else 0
        )

        # Ensure transcripts match
        if not (len(self.transcripts_shifted) == len(self.transcripts_golden)):
            raise ValueError("Shifted and golden transcripts are misaligned")

        # Store dataset length
        self.length = len(self.transcripts_shifted)

    def get_avg_chars_per_token(self) -> float:
        """
        Get the average number of characters per token
        (used for character-level perplexity). DO NOT MODIFY
        """
        return self.avg_chars_per_token

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """
        Get a single sample from the dataset.

        Returns:
            (shifted_transcript, golden_transcript)
        """
        shifted = torch.LongTensor(self.transcripts_shifted[idx])
        golden = torch.LongTensor(self.transcripts_golden[idx])
        return (shifted, golden)

    def collate_fn(self, batch: List[Tuple[torch.LongTensor, torch.LongTensor]]) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Collate and pad a batch of transcripts.
        Returns padded_shifted, padded_golden, and lengths.
        """
        # Unzip
        shifted_transcripts, golden_transcripts = zip(*batch)

        # Original lengths (all are same for shifted and golden)
        lengths = torch.LongTensor([len(seq) for seq in golden_transcripts])

        # Pad with pad_token
        padded_shifted = pad_sequence(
            shifted_transcripts,
            batch_first=True,
            padding_value=self.pad_token
        )
        padded_golden = pad_sequence(
            golden_transcripts,
            batch_first=True,
            padding_value=self.pad_token
        )

        return padded_shifted, padded_golden, lengths
