import torch
import torch.nn as nn
from typing import Tuple, Optional


class SelfAttentionLayer(nn.Module):
    '''
    Pre-LN Decoder Sub-Layer 1:
    Implements causally-masked self-attention with a pre-layernorm structure.
    '''

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        '''
        Initialize the SelfAttentionLayer.
        Args:
            d_model   (int): The dimension of the model.
            num_heads (int): The number of attention heads.
            dropout (float): The dropout rate.
        '''
        super().__init__()

        # 1) Multi-head attention (batch_first for shape (B, T, d_model))
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 2) Layer norm for pre-normalization
        self.norm = nn.LayerNorm(d_model)

        # 3) Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            x: torch.Tensor,
            key_padding_mask: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward pass for the SelfAttentionLayer.
        Args:
            x: shape (batch_size, seq_len, d_model)
            key_padding_mask: shape (batch_size, seq_len), True => pad
            attn_mask: shape (seq_len, seq_len), True => blocked

        Returns:
            x: shape (batch_size, seq_len, d_model)
            attn_weights: shape (batch_size, seq_len, seq_len)
        '''
        # Residual
        residual = x

        # Pre-LN
        x_normed = self.norm(x)

        # MHA: need_weights=True, average_attn_weights=True
        out, attn_weights = self.mha(
            x_normed, x_normed, x_normed,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=True,
            average_attn_weights=True
        )

        # Dropout + residual
        out = self.dropout(out)
        x = residual + out

        return x, attn_weights


class CrossAttentionLayer(nn.Module):
    '''
    Pre-LN Decoder Sub-Layer 2:
    Implements cross-attention (decoder queries, encoder keys/values).
    '''

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        '''
        Initialize the CrossAttentionLayer.
        Args:
            d_model   (int): The dimension of the model.
            num_heads (int): The number of attention heads.
            dropout (float): The dropout rate.
        '''
        super().__init__()

        # Multi-head attention for cross-attention
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer norm for pre-normalization
        self.norm = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            key_padding_mask: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward pass for the CrossAttentionLayer.
        Args:
            x: shape (batch_size, dec_len, d_model) => Decoder states
            y: shape (batch_size, enc_len, d_model) => Encoder outputs
            key_padding_mask: shape (batch_size, enc_len), True => pad
            attn_mask: shape (dec_len, enc_len), True => blocked

        Returns:
            x: shape (batch_size, dec_len, d_model)
            attn_weights: shape (batch_size, dec_len, enc_len)
        '''
        # Residual
        residual = x

        # Pre-LN
        x_normed = self.norm(x)

        # MHA: query=x, key=y, value=y
        out, attn_weights = self.mha(
            query=x_normed,
            key=y,
            value=y,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=True,
            average_attn_weights=True
        )

        # Dropout + residual
        out = self.dropout(out)
        x = residual + out

        return x, attn_weights


class FeedForwardLayer(nn.Module):
    '''
    Pre-LN Decoder Sub-Layer 3:
    Implements the position-wise feed-forward network with dropout and residual.
    '''

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        '''
        Initialize the FeedForwardLayer.
        Args:
            d_model (int): The dimension of the model.
            d_ff (int): The dimension of the feedforward network.
            dropout (float): The dropout rate.
        '''
        super().__init__()

        # 1) Position-wise feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

        # 2) Pre-layernorm
        self.norm = nn.LayerNorm(d_model)

        # 3) Extra dropout for the final residual connection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass for the FeedForwardLayer.
        Args:
            x: shape (batch_size, seq_len, d_model)

        Returns:
            x: shape (batch_size, seq_len, d_model)
        '''
        # Residual
        residual = x

        # Pre-LN
        x_normed = self.norm(x)

        # FFN
        out = self.ffn(x_normed)  # shape (B, T, d_model)

        # Dropout + residual
        out = self.dropout(out)
        x = residual + out
        return x
