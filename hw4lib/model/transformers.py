import torch.nn as nn
import torch
import random
from typing import Tuple, Optional, Literal
from .masks import PadMask, CausalMask
from .positional_encoding import PositionalEncoding
from .decoder_layers import SelfAttentionDecoderLayer, CrossAttentionDecoderLayer
from .encoder_layers import SelfAttentionEncoderLayer
from .speech_embedding import SpeechEmbedding
import warnings
from torchinfo import summary


## -------------------------------------------------------------------------------------------------
## Decoder-Only Transformer
## -------------------------------------------------------------------------------------------------
class DecoderOnlyTransformer(nn.Module):
    '''
    A Pre-LN Decoder-Only Transformer model.
    '''

    def __init__(
            self,
            num_layers: int,
            d_model: int,
            num_heads: int,
            d_ff: int,
            dropout: float,
            max_len: int,
            num_classes: int,
            weight_tying: bool = False,
            layer_drop_rate: float = 0.0,
    ):
        '''
        Initialize the Decoder-Only Transformer model.

        Args:
            num_layers: int, number of decoder layers
            d_model: int, model dimension
            num_heads: int, number of attention heads
            d_ff: int, feed-forward dimension
            dropout: float, dropout rate
            max_len: int, maximum sequence length this model can handle
            num_classes: int, number of classes
            weight_tying: bool, whether to use weight tying (default: False)
            layer_drop_rate: float, layer drop rate (default: 0.0)
        '''
        super().__init__()

        # DO NOT MODIFY THESE ATTRIBUTES
        self.max_len = max_len
        self.layer_drop_rate = layer_drop_rate
        self.num_classes = num_classes
        self.num_layers = num_layers

        # (1) Create the decoder layers
        self.dec_layers = nn.ModuleList([
            SelfAttentionDecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # (2) Create target embedding (token embedding)
        self.target_embedding = nn.Embedding(num_classes, d_model)

        # (3) Create positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        # (4) Create final linear layer (projection)
        self.final_linear = nn.Linear(d_model, num_classes)

        # (5) Create dropout + final norm
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        # Weight tying
        if weight_tying:
            # Re-use the same weight matrix for embedding + final linear
            # i.e. the row dimension of final_linear's weights must match embedding dimension
            if self.final_linear.weight.shape != self.target_embedding.weight.shape:
                raise ValueError("Weight tying failed: shape mismatch between embedding and final linear.")
            self.final_linear.weight = self.target_embedding.weight

    def forward(self, padded_targets: torch.Tensor, target_lengths: Optional[torch.Tensor] = None) -> Tuple[
        torch.Tensor, dict]:
        '''
        Forward pass for the decoder. Used for Training only. Tokens are assumed to be right-padded.
        Args:
            padded_targets (torch.Tensor): shape (batch_size, seq_len)
            target_lengths (Optional[torch.Tensor]): shape (batch_size,)
        Returns:
            seq_out (torch.Tensor): shape (batch_size, seq_len, num_classes)
            running_att (dict): attention weights
        '''
        # Ensure target_lengths is provided in training mode
        if self.training and target_lengths is None:
            raise ValueError("target_lengths must be provided during training")

        # (1) Create padding mask if we have target_lengths
        pad_mask_dec = None
        if target_lengths is not None:
            # shape => (batch_size, seq_len)
            pad_mask_dec = PadMask(padded_targets, target_lengths)

        # (2) Create causal mask => shape (seq_len, seq_len)
        causal_mask = CausalMask(padded_targets)

        # (3) Embedding + positional encoding + dropout
        # shape => (B, T) -> (B, T, d_model)
        x = self.target_embedding(padded_targets)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        # (4) Pass through decoder layers
        running_att = {}
        for i, layer in enumerate(self.dec_layers, start=1):
            # LayerDrop
            if self.training and self.layer_drop_rate > 0 and random.random() < self.layer_drop_rate:
                # skip entire layer
                continue

            # Self-attention
            x, attn_weights = layer(
                x,
                key_padding_mask=pad_mask_dec,
                attn_mask=causal_mask
            )
            # Save attention
            running_att[f'layer{i}_dec_self'] = attn_weights

        # (5) Final layer norm
        x = self.norm(x)

        # (6) Final linear => shape (B, T, num_classes)
        seq_out = self.final_linear(x)

        return seq_out, running_att

    def score(self, batch_prompts: torch.Tensor) -> torch.Tensor:
        '''
        Score the tokens for the decoder.
        This is used for scoring the next token for a given prompt.
        Padding mask is not applied so ensure that the prompts are not padded.
        This method can only handle batch_size=1 or same-length un-padded sequences.
        Args:
            batch_prompts (torch.Tensor): shape (batch_size, seq_len)
        Returns:
            logits (torch.Tensor): shape (batch_size, num_classes)
        '''
        if self.training:
            raise ValueError("score method is not supported during training, use forward method instead")

        # Forward with no target_lengths => no pad_mask
        seq_out, _ = self.forward(batch_prompts, target_lengths=None)

        # Return last token's logits
        # shape => (batch_size, num_classes)
        logits = seq_out[:, -1, :]
        return logits


## -------------------------------------------------------------------------------------------------
## Encoder-Decoder Transformer
## -------------------------------------------------------------------------------------------------
class EncoderDecoderTransformer(nn.Module):
    '''
    A Pre-LN Encoder-Decoder Transformer model for ASR tasks.
    '''

    def __init__(
            self,
            input_dim: int,
            time_reduction: int,
            reduction_method: Literal['lstm', 'conv', 'both'],
            num_encoder_layers: int,
            num_encoder_heads: int,
            d_ff_encoder: int,
            num_decoder_layers: int,
            num_decoder_heads: int,
            d_ff_decoder: int,
            d_model: int,
            dropout: float,
            max_len: int,
            num_classes: int,
            weight_tying: bool = False,
            layer_drop_rate: float = 0.0,
            skip_encoder_pe: bool = False,
            skip_decoder_pe: bool = False,
    ):
        '''
        Initialize the Encoder-Decoder Transformer model.

        Args:
            input_dim: dimension of input speech features
            time_reduction: stride along time dimension
            reduction_method: 'lstm', 'conv', or 'both'
            num_encoder_layers: # of encoder layers
            num_encoder_heads: # of encoder heads
            d_ff_encoder: feed-forward size for encoder
            num_decoder_layers: # of decoder layers
            num_decoder_heads: # of decoder heads
            d_ff_decoder: feed-forward size for decoder
            d_model: model dimension
            dropout: dropout rate
            max_len: maximum sequence length
            num_classes: number of output classes
            weight_tying: whether to tie weights
            layer_drop_rate: layer drop rate
            skip_encoder_pe: if True, skip positional encoding for encoder
            skip_decoder_pe: if True, skip positional encoding for decoder
        '''
        super().__init__()

        # DO NOT MODIFY THESE
        self.max_len = max_len
        self.layer_drop_rate = layer_drop_rate
        self.num_classes = num_classes
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.skip_encoder_pe = skip_encoder_pe
        self.skip_decoder_pe = skip_decoder_pe

        # (1) Create encoder layers
        self.enc_layers = nn.ModuleList([
            SelfAttentionEncoderLayer(
                d_model=d_model,
                num_heads=num_encoder_heads,
                d_ff=d_ff_encoder,
                dropout=dropout
            )
            for _ in range(num_encoder_layers)
        ])

        # (2) Create decoder layers
        self.dec_layers = nn.ModuleList([
            CrossAttentionDecoderLayer(
                d_model=d_model,
                num_heads=num_decoder_heads,
                d_ff=d_ff_decoder,
                dropout=dropout
            )
            for _ in range(num_decoder_layers)
        ])

        # (3) Create source embedding for speech
        self.source_embedding = SpeechEmbedding(
            input_dim=input_dim,
            output_dim=d_model,
            time_reduction=time_reduction,
            reduction_method=reduction_method,
            dropout=dropout
        )

        # (4) Create target embedding
        self.target_embedding = nn.Embedding(num_classes, d_model)

        # (5) Create one positional encoding module
        # We'll use the same class for both encoder/decoder if not skipped
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        # (6) Final linear
        self.final_linear = nn.Linear(d_model, num_classes)

        # (7) Dropout
        self.dropout = nn.Dropout(dropout)

        # (8) Norm layers for final encoder/decoder outputs
        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_norm = nn.LayerNorm(d_model)

        # (9) CTC head: project from d_model -> num_classes, then log_softmax
        self.ctc_head = nn.Sequential(
            nn.Linear(d_model, num_classes),
            nn.LogSoftmax(dim=-1)
        )

        # Weight tying if enabled
        if weight_tying:
            if self.final_linear.weight.shape != self.target_embedding.weight.shape:
                raise ValueError("Weight tying failed: mismatch between final_linear and target_embedding shapes.")
            self.final_linear.weight = self.target_embedding.weight

    def encode(self, padded_sources: torch.Tensor, source_lengths: torch.Tensor):
        '''
        Encodes the source features into a sequence of hidden states.
        Args:
            padded_sources: (B, src_len, input_dim)
            source_lengths: (B,)
        Returns:
            x_enc: (B, src_len_reduced, d_model)
            pad_mask_src: (B, src_len_reduced) -> True for padded
            running_att: dict of attention weights
            ctc_inputs: dict with:
                'log_probs': shape (src_len_reduced, B, num_classes), the CTC logits
                'lengths': shape (B,), the updated source_lengths after time reduction
        '''
        # 1) Apply speech embedding -> (B, T', d_model), T' <= src_len
        x_enc, x_enc_lengths = self.source_embedding(padded_sources, source_lengths)

        # 2) Optionally apply positional encoding if not skip_encoder_pe
        if not self.skip_encoder_pe:
            x_enc = self.positional_encoding(x_enc)

        # 3) Dropout
        x_enc = self.dropout(x_enc)

        # 4) Create source padding mask
        pad_mask_src = PadMask(x_enc, x_enc_lengths)  # shape (B, T')

        # 5) Pass through encoder layers
        running_att = {}
        for i, layer in enumerate(self.enc_layers, start=1):
            if self.training and self.layer_drop_rate > 0 and random.random() < self.layer_drop_rate:
                # skip entire layer
                continue
            x_enc, attn = layer(x_enc, key_padding_mask=pad_mask_src)
            running_att[f'layer{i}_enc_self'] = attn

        # 6) Final normalization
        x_enc = self.encoder_norm(x_enc)

        # 7) Project to CTC logits => shape (B, T', num_classes)
        ctc_out = self.ctc_head(x_enc)  # (B, T', num_classes)

        # For CTCLoss, we want shape (T', B, num_classes)
        ctc_out_tbc = ctc_out.transpose(0, 1)  # (T', B, num_classes)
        ctc_inputs = {
            'log_probs': ctc_out_tbc,
            'lengths': x_enc_lengths
        }

        return x_enc, pad_mask_src, running_att, ctc_inputs

    def decode(
            self,
            padded_targets: torch.Tensor,
            encoder_output: torch.Tensor,
            target_lengths: Optional[torch.Tensor] = None,
            pad_mask_src: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        '''
        Decode the target sequence conditioned on the encoder output.
        Args:
            padded_targets: (B, tgt_len)
            encoder_output: (B, src_len_reduced, d_model)
            target_lengths: (B,) or None
            pad_mask_src: (B, src_len_reduced)
        Returns:
            seq_out: (B, tgt_len, num_classes)
            running_att: dict of attention weights
        '''
        # 1) Create target padding mask
        pad_mask_tgt = None
        if target_lengths is not None:
            pad_mask_tgt = PadMask(padded_targets, target_lengths)

        # 2) Create causal mask
        causal_mask = CausalMask(padded_targets)

        # 3) Embedding
        x_dec = self.target_embedding(padded_targets)

        # 4) Optionally apply positional encoding if not skip_decoder_pe
        if not self.skip_decoder_pe:
            x_dec = self.positional_encoding(x_dec)

        # 5) Dropout
        x_dec = self.dropout(x_dec)

        # 6) Pass through decoder layers
        running_att = {}
        for i, layer in enumerate(self.dec_layers, start=1):
            if self.training and self.layer_drop_rate > 0 and random.random() < self.layer_drop_rate:
                # skip entire layer
                continue
            # CrossAttentionDecoderLayer => returns (x, self_attn, cross_attn)
            x_dec, self_attn, cross_attn = layer(
                x_dec,
                enc_output=encoder_output,
                dec_key_padding_mask=pad_mask_tgt,
                enc_key_padding_mask=pad_mask_src,
                attn_mask=causal_mask
            )
            running_att[f'layer{i}_dec_self'] = self_attn
            running_att[f'layer{i}_dec_cross'] = cross_attn

        # 7) Final normalization
        x_dec = self.decoder_norm(x_dec)

        # 8) Final projection
        seq_out = self.final_linear(x_dec)

        return seq_out, running_att

    def forward(
            self,
            padded_sources: torch.Tensor,
            padded_targets: torch.Tensor,
            source_lengths: Optional[torch.Tensor] = None,
            target_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict, dict]:
        '''
        Forward pass for the encoder-decoder transformer.

        Args:
            padded_sources: (B, src_len, input_dim)
            padded_targets: (B, tgt_len)
            source_lengths: (B,)
            target_lengths: (B,)
        Returns:
            seq_out: (B, tgt_len, num_classes)
            running_att: dict of attention weights (encoder + decoder)
            ctc_inputs: dict for CTC:
                { 'log_probs': (T', B, num_classes), 'lengths': (B,) }
        '''
        # During training, must have source_lengths + target_lengths
        if self.training:
            if target_lengths is None:
                raise ValueError("target_lengths must be provided during training")
            if source_lengths is None:
                raise ValueError("source_lengths must be provided during training")

        # 1) Encode
        encoder_output, pad_mask_src, enc_running_att, ctc_inputs = self.encode(
            padded_sources,
            source_lengths
        )

        # 2) Decode
        seq_out, dec_running_att = self.decode(
            padded_targets=padded_targets,
            encoder_output=encoder_output,
            target_lengths=target_lengths,
            pad_mask_src=pad_mask_src
        )

        # Combine attention dict
        running_att = {**enc_running_att, **dec_running_att}

        return seq_out, running_att, ctc_inputs

    def score(self, batch_prompts: torch.Tensor, encoder_output: torch.Tensor,
              pad_mask_src: torch.Tensor) -> torch.Tensor:
        '''
        Score the next token for given encoder output and prompt.
        Args:
            batch_prompts: (B, tgt_len)
            encoder_output: (B, src_len_reduced, d_model)
            pad_mask_src: (B, src_len_reduced)
        Returns:
            logits: (B, num_classes)
        '''
        if self.training:
            raise ValueError("score method is not supported during training")

        # decode w/ no target lengths
        seq_out, _ = self.decode(batch_prompts, encoder_output, None, pad_mask_src)

        # Return the last token's logits
        return seq_out[:, -1, :]

    @classmethod
    def from_pretrained_decoder(
            cls,
            decoder_checkpoint_path: str,
            config: dict,
    ):
        """
        Helper function to initialize an encoder-decoder transformer with decoder weights
        from a pretrained decoder-only model.
        Returns: (model, param_info)
        """
        print("\n=== Initializing Encoder-Decoder from Pretrained Decoder ===")
        print(f"Loading checkpoint from: {decoder_checkpoint_path}")

        # Create new encoder-decoder model
        print("\nCreating new encoder-decoder model...")
        model = cls(**config)

        # Load decoder checkpoint
        print("Loading pretrained decoder weights...")
        checkpoint = torch.load(decoder_checkpoint_path, map_location='cpu')
        decoder_state_dict = checkpoint['model_state_dict']

        # Track named parameters
        transferred_params = []
        new_params = []

        def transfer_module_weights(target_module, prefix):
            module_state_dict = {
                k.replace(prefix, ''): v
                for k, v in decoder_state_dict.items()
                if k.startswith(prefix)
            }
            target_module.load_state_dict(module_state_dict)
            for name, param in target_module.named_parameters():
                transferred_params.append((f"{prefix}{name}", param))

        print("\nTransferring shared components:")
        transfer_module_weights(model.target_embedding, 'target_embedding.')
        transfer_module_weights(model.final_linear, 'final_linear.')
        transfer_module_weights(model.decoder_norm, 'norm.')

        # Transfer decoder layers
        num_layers = min(
            len([k for k in decoder_state_dict.keys() if k.startswith('dec_layers.')]) // 2,
            model.num_decoder_layers
        )
        print(f"\nTransferring decoder layers (found {num_layers} layers):")
        for i in range(num_layers):
            print(f" Layer {i + 1}/{num_layers} ...")
            transfer_module_weights(model.dec_layers[i].self_attn, f'dec_layers.{i}.self_attn.')
            transfer_module_weights(model.dec_layers[i].ffn, f'dec_layers.{i}.ffn.')

        # Collect new parameters
        for name, param in model.named_parameters():
            is_new = True
            for transferred_name, transferred_param in transferred_params:
                if param is transferred_param:
                    is_new = False
                    break
            if is_new:
                new_params.append((name, param))

        print("\n=== Initialization Complete ===")
        return model, {'transferred': transferred_params, 'new': new_params}

    def log_param_groups(self, param_groups: list) -> None:
        """Log information about parameter groups."""
        print("\nParameter groups:")
        total_params = 0
        total_trainable = 0

        for group in param_groups:
            num_params = sum(p.numel() for p in group['params'])
            trainable = sum(p.numel() for p in group['params'] if p.requires_grad)
            total_params += num_params
            total_trainable += trainable

            print(f"\n{group['name']}:")
            print(f"  Parameters: {num_params:,}")
            print(f"  Trainable: {trainable:,}")
            print(f"  LR factor: {group['lr_factor']}")

        print(f"\nTotal parameters: {total_params:,}")
        print(f"Total trainable: {total_trainable:,}")
