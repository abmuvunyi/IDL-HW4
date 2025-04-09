import torch.nn as nn
import torch
from typing import Tuple, Optional
from .sublayers import SelfAttentionLayer, CrossAttentionLayer, FeedForwardLayer

'''
TODO: Implement these Modules.

1) SelfAttentionDecoderLayer: 
   - Has masked self-attention + feed-forward sublayers 
   - Used in decoder-only Transformers (e.g., GPT)

2) CrossAttentionDecoderLayer:
   - Has masked self-attention, cross-attention, and feed-forward sublayers
   - Used in encoder-decoder Transformers (e.g., BART)

We assume you're using the Pre-LN approach as described in the docstrings:
   - Each sublayer is pre-normalized,
   - Then a residual connection is added.
'''


class SelfAttentionDecoderLayer(nn.Module):
    '''
    Pre-LN Decoder Layer with masked self-attention and feed-forward sublayers.
    Used in the decoder-only Transformer architecture.
    '''

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        '''
        Initialize the SelfAttentionDecoderLayer.
        Args:
            d_model   (int): The dimension of the model.
            num_heads (int): The number of attention heads.
            d_ff      (int): The dimension of the feedforward network.
            dropout (float): The dropout rate.
        '''
        super().__init__()
        # Sublayers:
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

    def forward(
            self,
            x: torch.Tensor,
            key_padding_mask: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward pass for the SelfAttentionDecoderLayer.
        Args:
            x (torch.Tensor): shape: (batch_size, seq_len, d_model)
            key_padding_mask (Optional[torch.Tensor]): shape: (batch_size, seq_len)
            attn_mask (Optional[torch.Tensor]): shape: (seq_len, seq_len)

        Returns:
            x (torch.Tensor): shape: (batch_size, seq_len, d_model)
            mha_attn_weights (torch.Tensor): shape: (batch_size, seq_len, seq_len)
        '''
        # 1) Masked self-attention
        x, mha_attn_weights = self.self_attn(
            x,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask
        )

        # 2) Feed-forward
        x = self.ffn(x)

        return x, mha_attn_weights


class CrossAttentionDecoderLayer(nn.Module):
    '''
    Pre-LN Decoder Layer with masked self-attention, cross-attention, and feed-forward sublayers.
    Used in the encoder-decoder Transformer architecture.
    '''

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        '''
        Initialize the CrossAttentionDecoderLayer.
        Args:
            d_model   (int): The dimension of the model.
            num_heads (int): The number of attention heads.
            d_ff      (int): The dimension of the feedforward network.
            dropout (float): The dropout rate.
        '''
        super().__init__()
        # Sublayers:
        self.self_attn = SelfAttentionLayer(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )
        self.cross_attn = CrossAttentionLayer(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )
        self.ffn = FeedForwardLayer(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )

    def forward(
            self,
            x: torch.Tensor,
            enc_output: torch.Tensor,
            dec_key_padding_mask: Optional[torch.Tensor] = None,
            enc_key_padding_mask: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Forward pass for the CrossAttentionDecoderLayer.
        Args:
            x (torch.Tensor): shape: (batch_size, dec_len, d_model)
            enc_output (torch.Tensor): shape: (batch_size, enc_len, d_model)
            dec_key_padding_mask (Optional[torch.Tensor]): shape: (batch_size, dec_len)
            enc_key_padding_mask (Optional[torch.Tensor]): shape: (batch_size, enc_len)
            attn_mask (Optional[torch.Tensor]): shape: (dec_len, dec_len)
        Returns:
            x (torch.Tensor): shape: (batch_size, dec_len, d_model)
            self_attn_weights (torch.Tensor): shape: (batch_size, dec_len, dec_len)
            cross_attn_weights (torch.Tensor): shape: (batch_size, dec_len, enc_len)
        '''
        # 1) Masked self-attention
        x, self_attn_weights = self.self_attn(
            x,
            key_padding_mask=dec_key_padding_mask,
            attn_mask=attn_mask
        )

        # 2) Cross-attention (decoder queries = x, encoder keys/values = enc_output)
        x, cross_attn_weights = self.cross_attn(
            x,
            enc_output,
            key_padding_mask=enc_key_padding_mask
        )

        # 3) Feed-forward
        x = self.ffn(x)

        return x, self_attn_weights, cross_attn_weights
