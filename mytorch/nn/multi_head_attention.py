import numpy as np
from .linear import Linear
from .scaled_dot_product_attention import ScaledDotProductAttention


class MultiHeadAttention:
    """
    Multi Head Attention
    """

    def __init__(self, embed_dim, num_heads):
        """
        :param embed_dim: Embedding dimension
        :param num_heads: Number of attention heads
        """
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        # Initialize parameters and layers
        # DO NOT MODIFY
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Dimension per head
        self.head_dim = embed_dim // num_heads

        # Initialize your scaled dot product attention layer
        self.attention = ScaledDotProductAttention()  # Use your existing ScaledDotProductAttention

        # Initialize your linear layers
        #  embed_dim -> embed_dim
        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)

        # We'll store these in forward for use in backward
        self.q = None
        self.k = None
        self.v = None
        self.q_split = None
        self.k_split = None
        self.v_split = None
        self.attn_outputs = None
        self.attn_output = None
        self.output = None
        self.mask = None

        # Shapes for backward
        self.N = None
        self.L = None
        self.S = None
        self.E = None

    def init_weights(self, Wq, bq, Wk, bk, Wv, bv, Wo, bo):
        """
        Initialize the weights and biases with the given values.
        """
        # Initialize your linear layers (DO NOT MODIFY)
        self.q_proj.init_weights(Wq, bq)
        self.k_proj.init_weights(Wk, bk)
        self.v_proj.init_weights(Wv, bv)
        self.out_proj.init_weights(Wo, bo)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        """
        :param query: (N, L, E)
        :param key: (N, S, E)
        :param value: (N, S, E)
        :param key_padding_mask: (N, S) where 1/True indicates positions to ignore
        :param attn_mask: (L, S) where 1/True indicates positions to ignore
        :return: (N, L, E)
        """
        # TODO: Implement forward pass

        # Store shapes
        self.N = query.shape[0]
        self.L = query.shape[1]
        self.S = key.shape[1]
        self.E = query.shape[2]

        # 1) Project the query, key, and value
        # (N, L, E) -> (N, L, embed_dim)
        self.q = self.q_proj.forward(query)
        # (N, S, E) -> (N, S, embed_dim)
        self.k = self.k_proj.forward(key)
        # (N, S, E) -> (N, S, embed_dim)
        self.v = self.v_proj.forward(value)

        # 2) Split the query, key, and value into multiple heads
        # (N, L, embed_dim) -> (N, num_heads, L, embed_dim // num_heads)
        self.q_split = self._split_heads(self.q)  # shape: (N, num_heads, L, head_dim)
        self.k_split = self._split_heads(self.k)  # (N, num_heads, S, head_dim)
        self.v_split = self._split_heads(self.v)  # (N, num_heads, S, head_dim)

        # 3) Merge the masks
        # (N, S) + (L, S) -> (N, num_heads, L, S)
        self.mask = self._merge_masks(key_padding_mask, attn_mask)  # or None if no masks

        # 4) Apply the attention mechanism
        # We pass Q=(N, num_heads, L, head_dim), K=(N, num_heads, S, head_dim), V=(N, num_heads, S, head_dim)
        # result: (N, num_heads, L, head_dim)
        self.attn_outputs = self.attention.forward(
            Q=self.q_split,
            K=self.k_split,
            V=self.v_split,
            mask=self.mask
        )

        # 5) Merge (concatenate) the attention outputs
        # (N, num_heads, L, head_dim) -> (N, L, embed_dim)
        self.attn_output = self._concat_heads(self.attn_outputs)

        # 6) Project the attention outputs
        # (N, L, embed_dim) -> (N, L, embed_dim)
        self.output = self.out_proj.forward(self.attn_output)

        # Return output
        return self.output

    def backward(self, d_output):
        """
        :param d_output: Gradient of loss wrt output of shape (N, L, E)
        :return: Gradient of loss wrt input query, key, value of shapes (N, L, E), (N, S, E), (N, S, E)
        """
        # TODO: Implement backward pass

        # 1) Backprop through the output projection
        # (N, L, embed_dim) -> (N, L, embed_dim)
        d_attn_output = self.out_proj.backward(d_output)  # shape (N, L, embed_dim)

        # 2) Split the gradients into multiple heads
        # (N, L, embed_dim) -> (N, num_heads, L, head_dim)
        d_attn_outputs = self._split_heads(d_attn_output)  # same shape as self.attn_outputs

        # 3) Backprop through the attention mechanism
        # We'll get d_q_split, d_k_split, d_v_split
        d_q_split, d_k_split, d_v_split = self.attention.backward(d_attn_outputs, mask=self.mask)

        # 4) Merge (concatenate) the gradients w.r.t. Q, K, V heads
        # (N, num_heads, L, head_dim) -> (N, L, embed_dim)
        d_q_merged = self._concat_heads(d_q_split)  # shape (N, L, embed_dim)
        d_k_merged = self._concat_heads(d_k_split)  # shape (N, S, embed_dim)
        d_v_merged = self._concat_heads(d_v_split)  # shape (N, S, embed_dim)

        # 5) Backprop through the input projections
        # (N, L, embed_dim) -> (N, L, E)
        d_q = self.q_proj.backward(d_q_merged)
        # (N, S, embed_dim) -> (N, S, E)
        d_k = self.k_proj.backward(d_k_merged)
        # (N, S, embed_dim) -> (N, S, E)
        d_v = self.v_proj.backward(d_v_merged)

        # Return gradients d_q, d_k, d_v
        return d_q, d_k, d_v

    def _merge_masks(self, key_padding_mask, attn_mask):
        """
        Merge key_padding_mask and attn_mask into a single mask.
        :param key_padding_mask: (N, S)
        :param attn_mask: (L, S)
        :return: (N, H, L, S)
        """
        # TODO: Implement merge masks

        # If neither mask is provided, return None
        if key_padding_mask is None and attn_mask is None:
            return None

        # We'll build boolean masks and then combine them via logical OR
        # Expand key_padding_mask to (N, 1, 1, S) -> (N, self.num_heads, L, S)
        if key_padding_mask is not None:
            # Convert to boolean if needed. 1/True means ignore.
            key_mask = key_padding_mask.astype(bool)
            key_mask = np.expand_dims(key_mask, axis=1)  # (N, 1, S)
            key_mask = np.expand_dims(key_mask, axis=2)  # (N, 1, 1, S)
            # Broadcast to (N, H, L, S)
            key_mask = np.broadcast_to(key_mask, (self.N, self.num_heads, self.L, self.S))
        else:
            key_mask = None

        # Expand attn_mask to (1, 1, L, S) -> (N, self.num_heads, L, S)
        if attn_mask is not None:
            # Convert to boolean if needed
            attn_mask_bool = attn_mask.astype(bool)
            attn_mask_bool = np.expand_dims(attn_mask_bool, axis=0)  # (1, L, S)
            attn_mask_bool = np.expand_dims(attn_mask_bool, axis=0)  # (1, 1, L, S)
            # Broadcast to (N, H, L, S)
            attn_mask_bool = np.broadcast_to(attn_mask_bool, (self.N, self.num_heads, self.L, self.S))
        else:
            attn_mask_bool = None

        # Combine masks
        if key_mask is not None and attn_mask_bool is not None:
            combined_mask = np.logical_or(key_mask, attn_mask_bool)
        elif key_mask is not None:
            combined_mask = key_mask
        elif attn_mask_bool is not None:
            combined_mask = attn_mask_bool
        else:
            combined_mask = None

        return combined_mask

    def _split_heads(self, x):
        """
        Split the last dimension into (num_heads, d_k).
        Transpose to move num_heads dimension to the front.
        :param x: (N, L, embed_dim)
        :return: (N, num_heads, L, embed_dim // num_heads)
        """
        # TODO: Implement split heads

        # Reshape: (N, L, embed_dim) -> (N, L, num_heads, head_dim)
        new_shape = (x.shape[0], x.shape[1], self.num_heads, self.head_dim)
        x = x.reshape(new_shape)

        # Transpose to (N, num_heads, L, head_dim)
        x = np.transpose(x, (0, 2, 1, 3))

        # Return x
        return x

    def _concat_heads(self, x):
        """
        Concatenate the last dimension into (num_heads, d_k).
        Transpose to move num_heads dimension to the back.
        :param x: (N, num_heads, L, embed_dim // num_heads)
        :return: (N, L, embed_dim)
        """
        # TODO: Implement concat heads

        # Transpose: (N, num_heads, L, head_dim) -> (N, L, num_heads, head_dim)
        x = np.transpose(x, (0, 2, 1, 3))

        # Reshape: (N, L, num_heads, head_dim) -> (N, L, embed_dim)
        new_shape = (x.shape[0], x.shape[1], self.num_heads * self.head_dim)
        x = x.reshape(new_shape)

        # Return x
        return x
