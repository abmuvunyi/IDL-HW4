import torch.nn as nn
import torch
from typing import Tuple, Optional
from .sublayers import SelfAttentionLayer, FeedForwardLayer


class SelfAttentionEncoderLayer(nn.Module):
    '''
    Pre-LN Encoder Layer with self-attention mechanism.
    Used in the encoder part of transformer architectures.
    '''

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        '''
        Initialize the SelfAttentionEncoderLayer.
        Args:
            d_model   (int): The dimension of the model.
            num_heads (int): The number of attention heads.
            d_ff      (int): The dimension of the feedforward network.
            dropout (float): The dropout rate.
        '''
        super().__init__()

        # Sublayers
        # We reuse the same SelfAttentionLayer from sublayers.py
        # but won't pass a causal attn_mask for the forward
        self.self_attn = SelfAttentionLayer(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )
        self.ffn = FeedForwardLayer(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[
        torch.Tensor, torch.Tensor]:
        '''
        Forward pass for the EncoderLayer.
        Args:
            x (torch.Tensor): shape: (batch_size, seq_len, d_model)
            key_padding_mask (torch.Tensor): shape: (batch_size, seq_len)

        Returns:
            x (torch.Tensor): shape: (batch_size, seq_len, d_model)
            mha_attn_weights (torch.Tensor): shape: (batch_size, seq_len, seq_len)
        '''
        # 1) Self-attention with no causal mask
        #    We only pass key_padding_mask to handle padded tokens
        x, mha_attn_weights = self.self_attn(
            x,
            key_padding_mask=key_padding_mask,
            attn_mask=None  # no causal mask for encoder
        )

        # 2) Feed-forward
        x = self.ffn(x)

        return x, mha_attn_weights
