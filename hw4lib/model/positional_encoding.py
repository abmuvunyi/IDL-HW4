import torch
from torch import nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        """
        Initialize the PositionalEncoding.
        Args:
            d_model (int): The dimension of the model.
            max_len (int): The maximum length of the input sequence.

        Steps:
        1. Call parent class constructor using super().__init__()
        2. Call create_pe_table to initialize positional encoding matrix
        """
        super().__init__()
        self.create_pe_table(d_model, max_len)

    def create_pe_table(self, d_model, max_len):
        """
        Create the positional encoding table.

        Args:
            d_model (int): The dimension of the model.
            max_len (int): The maximum length of the input sequence.

        Side Effects:
            - Initializes the positional encoding buffer 'pe'
              of shape (1, max_len, d_model)
        """
        # Prepare a tensor of shape (max_len, d_model) to store the encodings
        pe = torch.zeros(max_len, d_model)

        # position: [0..max_len-1] shape (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # div_term: shape (d_model/2,)
        # This is the standard formula from "Attention is All You Need"
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )

        # For even i (i = 0,2,4,...), use sine
        pe[:, 0::2] = torch.sin(position * div_term)
        # For odd i (i = 1,3,5,...), use cosine
        pe[:, 1::2] = torch.cos(position * div_term)

        # Reshape to (1, max_len, d_model) for broadcasting
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)

        # Register as buffer to save with model
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the PositionalEncoding.
        Args:
            x (torch.Tensor): The input tensor of shape (B x T x d_model)
        Returns:
            torch.Tensor: Input with positional encoding added (B x T x d_model)
        Errors:
            - ValueError: If sequence length exceeds maximum length
        """
        seq_len = x.size(1)
        # Check against self.pe.size(1) (which is max_len)
        if seq_len > self.pe.size(1):
            raise ValueError(
                f"Sequence length {seq_len} exceeds the maximum length {self.pe.size(1)}"
            )

        # Add positional encodings to input
        # x shape:  (B, T, d_model)
        # pe shape: (1, max_len, d_model)
        # We'll take the first `seq_len` positions from pe
        x = x + self.pe[:, :seq_len, :]
        return x
