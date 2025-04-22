import wandb
import json
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
from hw4lib.data.tokenizer import H4Tokenizer
from hw4lib.utils import create_optimizer
from hw4lib.model import DecoderOnlyTransformer, EncoderDecoderTransformer
import os
import shutil
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from torchinfo import summary


class BaseTrainer(ABC):
    """
    Base Trainer class that provides common functionality for all trainers.

    ***UPDATED 2025‑04‑22***
    ─────────────────────────────────────────────────────────────────────────────
    • keeps a *global* step counter that is saved/loaded in checkpoints
    • synchronises wandb's internal counter with that global step when resuming
      so warnings of the form "step 0 < 60" disappear
    • _log_metrics() now uses that counter (unless caller overrides)
    """
    def __init__(
            self,
            model: nn.Module,
            tokenizer: H4Tokenizer,
            config: dict,
            run_name: str,
            config_file: str,
            device: Optional[str] = None
    ):
        # ── Device ────────────────────────────────────────────────────────────
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        self.device = device

        # ── Core objects ─────────────────────────────────────────────────────
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.config = config

        # ── Optim / sched scaffolding (child classes fill in) ────────────────
        self.optimizer = None
        self.scheduler = None
        self.scaler = torch.amp.GradScaler(device=self.device)

        # ── WandB flag ───────────────────────────────────────────────────────
        self.use_wandb = config['training'].get('use_wandb', False)

        # ── Experiment folders ───────────────────────────────────────────────
        paths = self._init_experiment(run_name, config_file)
        (self.expt_root, self.checkpoint_dir, self.attn_dir, self.text_dir,
         self.best_model_path, self.last_model_path) = paths

        # ── Training state ───────────────────────────────────────────────────
        self.current_epoch: int = 0
        self.global_step: int = 0            # <‑‑ NEW global counter
        self.best_metric: float = float('inf')
        self.training_history = []

    # ───────────────────────────────────────────────────────────────────────────
    # ABSTRACT METHODS (implemented by subclasses)
    # ───────────────────────────────────────────────────────────────────────────
    @abstractmethod
    def _train_epoch(self, dataloader) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        pass

    @abstractmethod
    def _validate_epoch(self, dataloader) -> Dict[str, float]:
        pass

    @abstractmethod
    def train(self, train_dataloader, val_dataloader):
        pass

    @abstractmethod
    def evaluate(self, dataloader) -> Dict[str, float]:
        pass

    # ───────────────────────────────────────────────────────────────────────────
    # INTERNAL helpers
    # ───────────────────────────────────────────────────────────────────────────
    def _init_experiment(self, run_name: str, config_file: str):
        """Creates directories, saves configs & model arch, initialises wandb."""
        expt_root = Path(os.getcwd()) / 'expts' / run_name
        expt_root.mkdir(parents=True, exist_ok=True)

        # save YAML
        shutil.copy2(config_file, expt_root / "config.yaml")

        # save model architecture summary --------------------------------------------------
        with open(expt_root / "model_arch.txt", "w") as f:
            if isinstance(self.model, DecoderOnlyTransformer):
                batch_size = self.config['data'].get('batch_size', 8)
                max_len = self.model.max_len
                model_summary = summary(self.model,
                                        input_size=[(batch_size, max_len), (batch_size,)],
                                        dtypes=[torch.long, torch.long])
                f.write(str(model_summary))
            elif isinstance(self.model, EncoderDecoderTransformer):
                batch_size = self.config['data'].get('batch_size', 8)
                max_len = 1000
                num_feats = self.config['data']['num_feats']
                dummy_inputs = [torch.randn(batch_size, max_len, num_feats).to(self.device),
                                torch.randint(0, self.model.num_classes, (batch_size, max_len // 10)).to(self.device),
                                torch.randint(max_len // 2, max_len, (batch_size,)).to(self.device),
                                torch.randint(max_len // 20, max_len // 10, (batch_size,)).to(self.device)]
                dtypes = [torch.float32, torch.long, torch.long, torch.long]
                model_summary = summary(self.model, input_data=dummy_inputs, dtypes=dtypes)
                f.write(str(model_summary))
            else:
                raise NotImplementedError("Model architecture summary not implemented for this class")

        # create sub‑dirs
        checkpoint_dir = expt_root / 'checkpoints'; checkpoint_dir.mkdir(exist_ok=True)
        attn_dir = expt_root / 'attn';             attn_dir.mkdir(exist_ok=True)
        text_dir = expt_root / 'text';             text_dir.mkdir(exist_ok=True)

        best_model_path = checkpoint_dir / 'checkpoint-best-metric-model.pth'
        last_model_path = checkpoint_dir / 'checkpoint-last-epoch-model.pth'

        # ── WandB init ────────────────────────────────────────────────────
        if self.use_wandb:
            run_id = self.config['training'].get('wandb_run_id', None)
            project = self.config['training'].get('wandb_project', 'default-project')

            if run_id and run_id.lower() != "none":
                self.wandb_run = wandb.init(project=project,
                                             id=run_id,
                                             resume="must",
                                             config=self.config,
                                             name=run_name)
            else:
                self.wandb_run = wandb.init(project=project,
                                             config=self.config,
                                             name=run_name)
        else:
            self.wandb_run = None

        return expt_root, checkpoint_dir, attn_dir, text_dir, best_model_path, last_model_path

    # ───────────────────────────────────────────────────────────────────────────
    # LOGGING helpers
    # ───────────────────────────────────────────────────────────────────────────
    def _log_metrics(self, metrics: Dict[str, Dict[str, float]], step: Optional[int] = None):
        """Log metrics to history, stdout, and wandb."""
        # default to internal counter
        if step is None:
            step = self.global_step
        self.training_history.append({'epoch': step, **metrics, 'lr': self.optimizer.param_groups[0]['lr']})

        # wandb
        if self.use_wandb:
            flat = {f"{split}/{k}": v for split, m in metrics.items() for k, v in m.items()}
            flat['learning_rate'] = self.optimizer.param_groups[0]['lr']
            wandb.log(flat, step=step)

        # console pretty print -------------------------------------------------
        print(f"\n📊 Metrics (Epoch {step}):")
        splits = sorted(metrics.keys())
        for i, split in enumerate(splits):
            print(f"{'└──' if i == len(splits)-1 else '├──'} {split.upper()}:" )
            split_metrics = sorted(metrics[split].items())
            for j, (k, v) in enumerate(split_metrics):
                bars = '    ' if i == len(splits)-1 else '│   '
                print(f"{bars}{'└──' if j == len(split_metrics)-1 else '├──'} {k}: {v:.4f}")
        print("└── TRAINING:\n    └── learning_rate: {:.6f}".format(self.optimizer.param_groups[0]['lr']))

        # advance internal counter exactly once per call
        self.global_step += 1
        if self.use_wandb and self.wandb_run:
            self.wandb_run.step = self.global_step

    # ───────────────────────────────────────────────────────────────────────────
    # CHECKPOINTS
    # ───────────────────────────────────────────────────────────────────────────
    def save_checkpoint(self, filename: str):
        path = self.checkpoint_dir / filename
        chk = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,        # <‑‑ save new counter
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict(),
            'best_metric': self.best_metric,
            'training_history': self.training_history,
            'config': self.config
        }
        torch.save(chk, path)
        if self.use_wandb:
            wandb.save(str(path))

    def load_checkpoint(self, filename: str):
        path = self.checkpoint_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"No checkpoint found at {path}")
        chk = torch.load(path, map_location=self.device)

        # load states (best‑effort)
        self.model.load_state_dict(chk['model_state_dict'])
        try:
            self.optimizer.load_state_dict(chk['optimizer_state_dict'])
        except Exception as e:
            print("Warning: couldn't load optimizer state:", e)
        if chk.get('scheduler_state_dict') and self.scheduler:
            try:
                self.scheduler.load_state_dict(chk['scheduler_state_dict'])
            except Exception as e:
                print("Warning: couldn't load scheduler state:", e)
        self.scaler.load_state_dict(chk['scaler_state_dict'])

        # restore counters / history ------------------------------------------------
        self.current_epoch = chk.get('epoch', 0) + 1
        self.global_step  = chk.get('global_step', 0) + 1
        self.best_metric  = chk.get('best_metric', float('inf'))
        self.training_history = chk.get('training_history', [])

        # --------------------------------------------------------
# W&B step is read‑only in the new SDK; we just rely on the
# explicit `step=` argument already passed in `_log_metrics`.
# --------------------------------------------------------

# print(f"Checkpoint loaded – resuming from epoch {self.current_epoch} (step {self.global_step})")
# f"Checkpoint loaded – resuming from epoch {self.current_epoch} (step {self.global_step})")

    # ───────────────────────────────────────────────────────────────────────────
    # VISUALISATIONS & TEXT OUTPUT (unchanged)
    # ───────────────────────────────────────────────────────────────────────────
    def _save_attention_plot(self, attn_weights: torch.Tensor, epoch: int, attn_type: str = "self"):
        if isinstance(attn_weights, torch.Tensor):
            attn_weights = attn_weights.cpu().detach().numpy()
        plt.figure(figsize=(10, 8)); sns.heatmap(attn_weights, cmap="viridis", cbar=True)
        plt.title(f"Attention Weights - Epoch {epoch}"); plt.xlabel("Source Sequence"); plt.ylabel("Target Sequence")
        plot_path = os.path.join(self.attn_dir, f"{attn_type}_attention_epoch{epoch}.png"); plt.savefig(plot_path); plt.close()
        if self.use_wandb:
            wandb.log({f"{attn_type}_attention": wandb.Image(plot_path)}, step=epoch)

    def _save_generated_text(self, text: dict, suffix: str):
        path = os.path.join(self.text_dir, f"text_{suffix}.json")
        with open(path, "w") as f:
            json.dump(text, f, indent=4)
        if self.use_wandb:
            wandb.save(path)

    # ───────────────────────────────────────────────────────────────────────────
    # CLEANUP
    # ───────────────────────────────────────────────────────────────────────────
    def cleanup(self):
        if self.use_wandb and self.wandb_run:
            wandb.finish()
