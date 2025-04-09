import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torchaudio.transforms as tat
from typing import Literal, Tuple, Optional
from .tokenizer import H4Tokenizer


class ASRDataset(Dataset):
    def __init__(
            self,
            partition: Literal['train-clean-100', 'dev-clean', 'test-clean'],
            config: dict,
            tokenizer: H4Tokenizer,
            isTrainPartition: bool,
            global_stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ):
        """
        Initialize the ASRDataset for ASR training/validation/testing.

        Args:
            partition (str): Dataset partition ('train-clean-100', 'dev-clean', or 'test-clean')
            config (dict): Configuration dictionary containing dataset settings
            tokenizer (H4Tokenizer): Tokenizer for encoding/decoding text
            isTrainPartition (bool): Whether this is the training partition (for SpecAugment, etc.)
            global_stats (tuple, optional): (mean, std) computed from training set for global MVN.
        """
        # Store basic configuration
        self.config = config
        self.partition = partition
        self.isTrainPartition = isTrainPartition
        self.tokenizer = tokenizer

        # Token IDs
        self.eos_token = self.tokenizer.eos_id
        self.sos_token = self.tokenizer.sos_id
        self.pad_token = self.tokenizer.pad_id

        # Set up data directories
        # e.g. root might be hw4_data_subset/hw4p2_data
        root = self.config["root"]  # e.g. "hw4_data_subset/hw4p2_data"
        self.fbank_dir = os.path.join(root, partition, "fbank")

        # Gather feature files
        all_fbank_files = sorted([
            f for f in os.listdir(self.fbank_dir) if f.endswith(".npy")
        ])
        self.fbank_files = [os.path.join(self.fbank_dir, f) for f in all_fbank_files]

        # Possibly subset
        subset_size = self.config.get("subset", None)
        if subset_size is not None and subset_size < len(self.fbank_files):
            self.fbank_files = self.fbank_files[:subset_size]

        # Count how many samples
        self.length = len(self.fbank_files)

        # If not test-clean, gather text files too
        if self.partition != "test-clean":
            self.text_dir = os.path.join(root, partition, "text")
            all_text_files = sorted([
                f for f in os.listdir(self.text_dir) if f.endswith(".npy")
            ])
            self.text_files = [os.path.join(self.text_dir, f) for f in all_text_files]

            # Possibly subset text the same way
            if subset_size is not None and subset_size < len(self.text_files):
                self.text_files = self.text_files[:subset_size]

            # Check alignment
            if len(self.fbank_files) != len(self.text_files):
                raise ValueError("Number of feature files and transcript files must match")

        # We'll store loaded features/transcripts
        self.feats = []
        self.transcripts_shifted = []
        self.transcripts_golden = []

        # Keep track of dataset-level stats
        self.total_chars = 0
        self.total_tokens = 0
        self.feat_max_len = 0
        self.text_max_len = 0

        # Possibly keep track for global MVN
        self.global_mean = None
        self.global_std = None

        # If we are doing global MVN and no global_stats provided, we compute them here
        if self.config['norm'] == 'global_mvn' and global_stats is None:
            if not isTrainPartition:
                raise ValueError("global_stats must be provided for non-training partitions in global_mvn")
            count = 0
            mean = torch.zeros(self.config['num_feats'], dtype=torch.float64)
            M2 = torch.zeros(self.config['num_feats'], dtype=torch.float64)
        else:
            # We might not do anything if norm != global_mvn or if global_stats is provided
            pass

        print(f"Loading data for {partition} partition...")
        for i in tqdm(range(self.length)):
            # Load feature of shape (num_feats, time)
            feat_np = np.load(self.fbank_files[i])
            # truncate to config['num_feats'] if needed
            # e.g. if the .npy might have more or fewer freq bins
            feat_np = feat_np[:self.config['num_feats'], :]  # e.g. (num_feats, time)
            self.feat_max_len = max(self.feat_max_len, feat_np.shape[1])

            # Possibly update global stats
            if self.config['norm'] == 'global_mvn' and global_stats is None:
                feat_tensor = torch.FloatTensor(feat_np)
                batch_count = feat_tensor.shape[1]
                count += batch_count

                # Welford
                delta = feat_tensor - mean.unsqueeze(1)
                mean += delta.mean(dim=1)
                delta2 = feat_tensor - mean.unsqueeze(1)
                M2 += (delta * delta2).sum(dim=1)

            # Store the raw feature (still as np, or we can convert to torch in __getitem__)
            self.feats.append(feat_np)

            if self.partition != "test-clean":
                # Load transcript
                text_np = np.load(self.text_files[i], allow_pickle=True)
                transcript_str = "".join(list(text_np))  # or adapt if array is already a single string
                self.total_chars += len(transcript_str)

                # Tokenize
                tokenized = self.tokenizer.encode(transcript_str)
                self.total_tokens += len(tokenized)
                self.text_max_len = max(self.text_max_len, len(tokenized) + 1)

                # Build shifted + golden
                shifted_seq = [self.sos_token] + tokenized
                golden_seq = tokenized + [self.eos_token]
                self.transcripts_shifted.append(shifted_seq)
                self.transcripts_golden.append(golden_seq)

        # final average chars/token
        self.avg_chars_per_token = self.total_chars / self.total_tokens if self.total_tokens > 0 else 0

        if self.partition != "test-clean":
            if not (len(self.feats) == len(self.transcripts_shifted) == len(self.transcripts_golden)):
                raise ValueError("Features and transcripts are misaligned")

        # if we must compute global stats
        if self.config['norm'] == 'global_mvn':
            if global_stats is not None:
                self.global_mean, self.global_std = global_stats
            else:
                # finalize Welford
                variance = M2 / (count - 1)
                self.global_std = torch.sqrt(variance + 1e-8).float()
                self.global_mean = mean.float()

        # create specaug transforms
        self.time_mask = tat.TimeMasking(
            time_mask_param=config['specaug_conf']['time_mask_width_range'],
            iid_masks=True
        )
        self.freq_mask = tat.FrequencyMasking(
            freq_mask_param=config['specaug_conf']['freq_mask_width_range'],
            iid_masks=True
        )

    def get_avg_chars_per_token(self):
        """
        For perplexity calculations
        """
        return self.avg_chars_per_token

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Return (feat, shifted_transcript, golden_transcript) for training/val
        or (feat, None, None) for test-clean
        """
        feat_np = self.feats[idx]  # shape (num_feats, time)
        feat = torch.FloatTensor(feat_np)  # convert to torch

        # normalization
        if self.config['norm'] == 'global_mvn':
            # (num_feats, time)
            feat = (feat - self.global_mean.unsqueeze(1)) / (self.global_std.unsqueeze(1) + 1e-8)
        elif self.config['norm'] == 'cepstral':
            # per-utt
            m = feat.mean(dim=1, keepdim=True)
            s = feat.std(dim=1, keepdim=True) + 1e-8
            feat = (feat - m) / s
        # else 'none': do nothing

        shifted, golden = None, None
        if self.partition != "test-clean":
            shifted_seq = self.transcripts_shifted[idx]
            golden_seq = self.transcripts_golden[idx]
            shifted = torch.LongTensor(shifted_seq)
            golden = torch.LongTensor(golden_seq)

        return feat, shifted, golden

    def collate_fn(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        returns:
         padded_feats  (B, max_time, num_feats)
         padded_shifted (B, max_text_len) or None
         padded_golden  (B, max_text_len) or None
         feat_lengths (B,)
         transcript_lengths (B,) or None
        """

        # separate out
        feat_list, shifted_list, golden_list = zip(*batch)  # each is a tuple

        # lengths
        feat_lengths = [f.shape[1] for f in feat_list]  # number of time frames
        feat_lengths = torch.LongTensor(feat_lengths)

        # pad features (list of shape (num_feats, time)) => we want (B, T, F)
        # first we transpose to shape (time, num_feats), then pad, then transpose back
        transposed = [f.transpose(0, 1) for f in feat_list]  # each => (time, num_feats)
        padded_feats = pad_sequence(transposed, batch_first=True, padding_value=0.0)  # (B, max_time, num_feats)
        padded_feats = padded_feats  # shape => (B, T, F)

        # handle transcripts
        padded_shifted, padded_golden, transcript_lengths = None, None, None
        if self.partition != "test-clean":
            # gather only non-None transcripts
            shift_lens = [s.shape[0] for s in shifted_list]
            golden_lens = [g.shape[0] for g in golden_list]
            shift_lens = torch.LongTensor(shift_lens)
            # or we can confirm shift_lens == golden_lens
            transcript_lengths = shift_lens  # we can store the length of golden or shifted, both same size

            padded_shifted = pad_sequence(shifted_list, batch_first=True, padding_value=self.pad_token)
            padded_golden = pad_sequence(golden_list, batch_first=True, padding_value=self.pad_token)

        # specaug only for training
        if self.config.get("specaug", False) and self.isTrainPartition:
            # permute to (B, F, T)
            padded_feats = padded_feats.transpose(1, 2)  # (B, F, T)

            # freq mask
            if self.config["specaug_conf"].get("apply_freq_mask", False):
                num_freq_mask = self.config["specaug_conf"].get("num_freq_mask", 2)
                for _ in range(num_freq_mask):
                    padded_feats = self.freq_mask(padded_feats)

            # time mask
            if self.config["specaug_conf"].get("apply_time_mask", False):
                num_time_mask = self.config["specaug_conf"].get("num_time_mask", 2)
                for _ in range(num_time_mask):
                    padded_feats = self.time_mask(padded_feats)

            # back to (B, T, F)
            padded_feats = padded_feats.transpose(1, 2)

        return padded_feats, padded_shifted, padded_golden, feat_lengths, transcript_lengths
