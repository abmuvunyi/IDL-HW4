import numpy as np


class Linear:
    def __init__(self, in_features, out_features):
        """
        Initialize the weights and biases with zeros
        W shape: (out_features, in_features)
        b shape: (out_features,)  # Changed from (out_features, 1) to match PyTorch
        """
        # DO NOT MODIFY
        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros(out_features)

    def init_weights(self, W, b):
        """
        Initialize the weights and biases with the given values.
        """
        # DO NOT MODIFY
        self.W = W
        self.b = b

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (*, in_features)
        :return: Output Z with shape (*, out_features)

        Handles arbitrary batch dimensions like PyTorch
        """
        # TODO: Implement forward pass

        # Store input for backward pass
        self.A = A

        # Reshape the input to be 2D (batch_size, in_features)
        A_flat = A.reshape(-1, self.W.shape[1])

        # Compute the output: Z = A * W.T + b
        Z = np.dot(A_flat, self.W.T) + self.b

        # Reshape back to the original input shape with the new last dimension
        Z = Z.reshape(*A.shape[:-1], self.W.shape[0])

        return Z

    def backward(self, dLdZ):
        """
        :param dLdZ: Gradient of loss wrt output Z (*, out_features)
        :return: Gradient of loss wrt input A (*, in_features)
        """
        # Compute gradients (refer to the equations in the writeup)

        # Gradient of loss w.r.t. input A: dLdA = dLdZ * W
        self.dLdA = np.dot(dLdZ, self.W)

        # Gradient of loss w.r.t. W: dLdW = (dLdZ.T * A).sum(axis=0)
        # Flatten A to ensure the dot product is valid for all batch dimensions
        A_flat = self.A.reshape(-1, self.W.shape[1])

        # Fix the issue in gradient computation: reshape dLdZ to match the shape of A_flat
        self.dLdW = np.dot(dLdZ.reshape(-1, self.W.shape[0]).T, A_flat)

        # Gradient of loss w.r.t. b: dLdb = sum(dLdZ, axis=0)
        self.dLdb = np.sum(dLdZ, axis=(0, 1))  # Sum over batch and seq_len

        # Return gradient of loss w.r.t input A
        return self.dLdA
