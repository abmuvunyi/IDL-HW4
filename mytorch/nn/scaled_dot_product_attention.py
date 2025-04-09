import numpy as np
from .activation import Softmax


class ScaledDotProductAttention:
    """
    Scaled Dot Product Attention
    """

    def __init__(self):
        self.eps = 1e-4  # DO NOT MODIFY
        self.softmax = Softmax(dim=-1)  # Softmax along the last dimension (S)

        # We'll store Q, K, V for backward
        self.Q = None
        self.K = None
        self.V = None
        self.attention_scores = None

    def forward(self, Q, K, V, mask=None):
        """
        :param Q: shape (N, ..., H, L, E)
        :param K: shape (N, ..., H, S, E)
        :param V: shape (N, ..., H, S, Ev)
        :param mask: shape (N, ..., H, L, S) or broadcastable shape
        :return: Output of shape (N, ..., H, L, Ev)
        """
        # Store Q, K, V for backward
        self.Q = Q
        self.K = K
        self.V = V

        # 1) Compute scaled dot product: (Q @ K^T) / sqrt(E)
        E = K.shape[-1]
        scaled_dot_product = np.matmul(Q, K.swapaxes(-2, -1)) / np.sqrt(E)

        # 2) Apply mask before softmax (if any)
        if mask is not None:
            scaled_dot_product += (mask * -self.eps)

        # 3) Softmax to get attention scores
        self.attention_scores = self.softmax.forward(scaled_dot_product)  # shape (N, ..., H, L, S)

        # 4) Multiply by V: (N, ..., H, L, S) @ (N, ..., H, S, Ev) -> (N, ..., H, L, Ev)
        output = np.matmul(self.attention_scores, V)
        return output

    def backward(self, d_output, mask=None):
        """
        :param d_output: shape (N, ..., H, L, Ev)
        :param mask: shape (N, ..., H, L, S) or broadcastable shape
        :return: (dQ, dK, dV)
        """
        E = self.K.shape[-1]

        # --- 1) Gradient wrt V ---
        # attention_scores: (N, ..., H, L, S)
        # d_output:         (N, ..., H, L, Ev)
        # => dV:            (N, ..., H, S, Ev)
        d_V = np.matmul(self.attention_scores.swapaxes(-2, -1), d_output)

        # --- 2) Gradient wrt attention_scores ---
        # d_output:   (N, ..., H, L, Ev)
        # V^T:        (N, ..., H, Ev, S)
        # => d_attention_scores: (N, ..., H, L, S)
        d_attention_scores = np.matmul(d_output, self.V.swapaxes(-2, -1))

        # --- 3) Backprop through softmax ---
        # We must apply the derivative of the softmax to d_attention_scores
        d_attention_scores = self.softmax.backward(d_attention_scores)

        # --- 4) Scale by sqrt(E) for the dot product derivative ---
        d_scaled_dot_product = d_attention_scores / np.sqrt(E)

        # --- 5) Apply mask if present ---
        if mask is not None:
            d_scaled_dot_product += (mask * -self.eps)

        # --- 6) Gradients wrt Q and K ---
        # dQ: (N, ..., H, L, S) @ (N, ..., H, S, E) -> (N, ..., H, L, E)
        d_Q = np.matmul(d_scaled_dot_product, self.K)
        # dK: (N, ..., H, L, S)^T @ (N, ..., H, L, E) -> (N, ..., H, S, E)
        d_K = np.matmul(d_scaled_dot_product.swapaxes(-2, -1), self.Q)

        return d_Q, d_K, d_V
