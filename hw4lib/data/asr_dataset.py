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
            isTrainPartition (bool): Whether this is the training partition
            global_stats (tuple, optional): (mean, std) computed from training set for global MVN.
        """
        super().__init__()

        self.config = config
        self.partition = partition
        self.isTrainPartition = isTrainPartition
        self.tokenizer = tokenizer

        # Token IDs
        self.eos_token = self.tokenizer.eos_id
        self.sos_token = self.tokenizer.sos_id
        self.pad_token = self.tokenizer.pad_id

        # Directories
        root = self.config["root"]  # e.g. "hw4_data_subset/hw4p2_data"
        self.fbank_dir = os.path.join(root, partition, "fbank")

        # 1) Gather all fbank files and build basenames
        fbank_files_all = sorted([f for f in os.listdir(self.fbank_dir) if f.endswith(".npy")])
        fbank_basenames = {os.path.splitext(fname)[0] for fname in fbank_files_all}

        self.text_files = None
        if self.partition != "test-clean":
            # gather text dir
            self.text_dir = os.path.join(root, partition, "text")
            text_files_all = sorted([f for f in os.listdir(self.text_dir) if f.endswith(".npy")])
            text_basenames = {os.path.splitext(fname)[0] for fname in text_files_all}

            # Intersect so we only keep aligned pairs
            common_basenames = sorted(fbank_basenames.intersection(text_basenames))

            # Build the final aligned fbank + text lists
            self.fbank_files = [os.path.join(self.fbank_dir, b + ".npy") for b in common_basenames]
            self.text_files = [os.path.join(self.text_dir, b + ".npy") for b in common_basenames]
        else:
            # test-clean => no text
            common_basenames = sorted(fbank_basenames)
            self.fbank_files = [os.path.join(self.fbank_dir, b + ".npy") for b in common_basenames]

        self.length = len(self.fbank_files)
        if self.partition != "test-clean" and self.length == 0:
            raise ValueError("No matched fbank/text files found, cannot proceed with training or dev set")

        # 2) Possibly subset
        subset_size = self.config.get('subset', None)
        if subset_size is not None:
            subset_size = int(subset_size)  # ensure int
            if subset_size < self.length:
                self.fbank_files = self.fbank_files[:subset_size]
                if self.text_files is not None:
                    self.text_files = self.text_files[:subset_size]
                self.length = len(self.fbank_files)

        # We'll store loaded features/transcripts
        self.feats = []
        self.transcripts_shifted = []
        self.transcripts_golden = []

        # Stats
        self.total_chars = 0
        self.total_tokens = 0
        self.feat_max_len = 0
        self.text_max_len = 0

        self.global_mean = None
        self.global_std = None

        # If config says global_mvn, we might compute stats with Welford
        if self.config['norm'] == 'global_mvn' and global_stats is None:
            if not isTrainPartition:
                raise ValueError("global_stats must be provided for non-training partitions if using global_mvn")
            count = 0
            mean = torch.zeros(self.config['num_feats'], dtype=torch.float64)
            M2 = torch.zeros(self.config['num_feats'], dtype=torch.float64)
        else:
            # or set from global_stats
            pass

        print(f"Loading data for {partition} partition...")
        for i in tqdm(range(self.length)):
            # load feature
            feat_np = np.load(self.fbank_files[i])  # shape (any, time)
            feat_np = feat_np[:self.config['num_feats'], :]  # truncate to exact num_feats
            self.feat_max_len = max(self.feat_max_len, feat_np.shape[1])

            # Welford if needed
            if self.config['norm'] == 'global_mvn' and global_stats is None:
                feat_t = torch.FloatTensor(feat_np)  # (num_feats, time)
                batch_count = feat_t.shape[1]
                count += batch_count
                delta = feat_t - mean.unsqueeze(1)
                mean += delta.mean(dim=1)
                delta2 = feat_t - mean.unsqueeze(1)
                M2 += (delta * delta2).sum(dim=1)

            self.feats.append(feat_np)

            # If not test-clean => load transcripts
            if self.partition != "test-clean":
                text_np = np.load(self.text_files[i], allow_pickle=True)
                # Convert to string
                transcript = "".join(list(text_np))
                self.total_chars += len(transcript)

                tokenized = self.tokenizer.encode(transcript)
                self.total_tokens += len(tokenized)
                self.text_max_len = max(self.text_max_len, len(tokenized) + 1)

                shifted_seq = [self.sos_token] + tokenized
                golden_seq = tokenized + [self.eos_token]
                self.transcripts_shifted.append(shifted_seq)
                self.transcripts_golden.append(golden_seq)

        self.avg_chars_per_token = self.total_chars / self.total_tokens if self.total_tokens > 0 else 0

        if self.partition != "test-clean":
            # check alignment if we want to be extra sure
            if not (len(self.feats) == len(self.transcripts_shifted) == len(self.transcripts_golden)):
                raise ValueError("Loaded features and transcripts are misaligned in length after building lists")

        # finalize global stats
        if self.config['norm'] == 'global_mvn':
            if global_stats is not None:
                self.global_mean, self.global_std = global_stats
            else:
                variance = M2 / (count - 1)
                self.global_std = torch.sqrt(variance + 1e-8).float()
                self.global_mean = mean.float()

        # create specaug transforms
        self.time_mask = tat.TimeMasking(
            time_mask_param=self.config['specaug_conf']['time_mask_width_range'],
            iid_masks=True
        )
        self.freq_mask = tat.FrequencyMasking(
            freq_mask_param=self.config['specaug_conf']['freq_mask_width_range'],
            iid_masks=True
        )

    def get_avg_chars_per_token(self):
        return self.avg_chars_per_token

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        feat_np = self.feats[idx]
        feat = torch.FloatTensor(feat_np)  # (num_feats, time)

        # normalization
        if self.config['norm'] == 'global_mvn' and self.global_mean is not None and self.global_std is not None:
            # apply
            feat = (feat - self.global_mean.unsqueeze(1)) / (self.global_std.unsqueeze(1) + 1e-8)
        elif self.config['norm'] == 'cepstral':
            # per-utterance
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

    def collate_fn(self, batch):
        # Unzip
        feats_list, shifted_list, golden_list = zip(*batch)  # each is a tuple

        # feature lengths
        feat_lengths = [f.shape[1] for f in feats_list]
        feat_lengths = torch.LongTensor(feat_lengths)

        # we want to pad features to (B, max_time, num_feats)
        # but each is (num_feats, time) => transpose to (time, num_feats)
        feats_transposed = [f.transpose(0, 1) for f in feats_list]  # (time, num_feats)
        padded_feats = pad_sequence(feats_transposed, batch_first=True, padding_value=0.0)
        # now shape => (B, max_time, num_feats)

        # handle transcripts
        padded_shifted, padded_golden, transcript_lengths = None, None, None
        if self.partition != "test-clean":
            shift_lens = [s.shape[0] for s in shifted_list]
            shift_lens = torch.LongTensor(shift_lens)
            padded_shifted = pad_sequence(shifted_list, batch_first=True, padding_value=self.pad_token)
            padded_golden = pad_sequence(golden_list, batch_first=True, padding_value=self.pad_token)
            transcript_lengths = shift_lens

        # specaug if training + config says so
        if self.config.get("specaug", False) and self.isTrainPartition:
            padded_feats = padded_feats.transpose(1, 2)  # (B, F, T)
            if self.config["specaug_conf"].get("apply_freq_mask", False):
                num_freq_mask = self.config["specaug_conf"].get("num_freq_mask", 2)
                for _ in range(num_freq_mask):
                    padded_feats = self.freq_mask(padded_feats)
            if self.config["specaug_conf"].get("apply_time_mask", False):
                num_time_mask = self.config["specaug_conf"].get("num_time_mask", 2)
                for _ in range(num_time_mask):
                    padded_feats = self.time_mask(padded_feats)
            padded_feats = padded_feats.transpose(1, 2)  # back to (B, T, F)

        return padded_feats, padded_shifted, padded_golden, feat_lengths, transcript_lengths
