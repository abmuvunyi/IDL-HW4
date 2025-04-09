import torch


def PadMask(padded_input, input_lengths):
    """
    Create a mask to identify non-padding positions.
    Args:
        padded_input: The input tensor with padding, shape (N, T, ...) or (N, T).
        input_lengths: The actual lengths of each sequence before padding, shape (N,).
    Returns:
        A boolean mask tensor with shape (N, T), where:
            - padding positions are marked with True
            - non-padding positions are marked with False.
    """
    # We assume the first dimension is batch (N) and the second dimension is time (T)
    N, T = padded_input.shape[0], padded_input.shape[1]
    device = padded_input.device

    # For each batch index n, positions >= input_lengths[n] are padded
    # Compare an arange [0..T-1] to each length
    # If >= length => True (i.e., mask out / ignore)
    mask = torch.arange(T, device=device).unsqueeze(0) >= input_lengths.unsqueeze(1)
    return mask


def CausalMask(padded_input):
    """
    Create a mask to identify non-causal positions.
    Args:
        padded_input: The input tensor with padding, shape (N, T, ...) or (N, T).

    Returns:
        A boolean mask tensor with shape (T, T), where:
            - non-causal positions (don't attend to) are marked with True
            - causal positions (can attend to) are marked with False.
    """
    # We assume the second dimension is time (T)
    T = padded_input.shape[1]
    device = padded_input.device

    # A standard approach is to create an upper-triangular matrix (excluding diagonal)
    # If diagonal=1, the diagonal is False and positions above it are True
    # e.g. for i < j => True
    mask = torch.triu(
        torch.ones(T, T, dtype=torch.bool, device=device),
        diagonal=1
    )
    return mask
