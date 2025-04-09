import numpy as np


class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """

    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        DO NOT MODIFY
        """
        self.dim = dim

    def forward(self, Z):
        """
        :param Z: Data Z (*) to apply activation function to input Z.
        :return: Output returns the computed output A (*).
        """
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError("Dimension to apply softmax to is greater than the number of dimensions in Z")

        # TODO: Implement forward pass
        # Compute the softmax in a numerically stable way
        Z_max = np.max(Z, axis=self.dim, keepdims=True)  # To prevent overflow in exp
        Z_exp = np.exp(Z - Z_max)  # Subtract max to maintain numerical stability

        # Compute softmax
        softmax = Z_exp / np.sum(Z_exp, axis=self.dim, keepdims=True)

        self.A = softmax  # Store the result for backward pass
        return self.A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        # TODO: Implement backward pass

        # Get the shape of the input
        shape = self.A.shape
        # Find the dimension along which softmax was applied
        C = shape[self.dim]

        # Reshape input to 2D
        if len(shape) > 2:
            A_flat = self.A.reshape(-1, C)
            dLdA_flat = dLdA.reshape(-1, C)
        else:
            A_flat = self.A
            dLdA_flat = dLdA

        # Compute the gradient with respect to input
        # Gradient of the softmax function:
        # dL/dZ = dL/dA * softmax'(Z)
        dLdZ = np.zeros_like(A_flat)

        for i in range(A_flat.shape[0]):
            jacobian = np.diag(A_flat[i]) - np.outer(A_flat[i], A_flat[i])  # Softmax jacobian
            dLdZ[i] = np.dot(jacobian, dLdA_flat[i])  # Apply chain rule

        # Reshape back to original dimensions if necessary
        if len(shape) > 2:
            dLdZ = dLdZ.reshape(*shape)

        return dLdZ
