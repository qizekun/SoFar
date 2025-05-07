import numpy as np
import random
import torch
from collections import abc


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def set_random_seed(seed, deterministic=False):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.

    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def is_seq_of(seq, expected_type, seq_type=None):
    """Check whether it is a sequence of some type.
    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type.
    Returns:
        bool: Whether the sequence is valid.
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def fps(xyz, num_points):
    """
    Farthest Point Sampling (FPS) implementation in PyTorch.

    Args:
        xyz (torch.Tensor): Input point cloud of shape (B, N, 3).
        num_points (int): Number of points to sample.

    Returns:
        torch.Tensor: Sampled points of shape (B, num_points, 3).
    """
    B, N, _ = xyz.shape
    device = xyz.device

    centroids = torch.zeros(B, num_points, dtype=torch.long, device=device)  # Indices of sampled points
    distances = torch.ones(B, N, device=device) * 1e10  # Initialize distances as large values
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)  # Random initial farthest point

    batch_indices = torch.arange(B, device=device)

    for i in range(num_points):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].unsqueeze(1)  # Shape: (B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)  # Squared Euclidean distance
        distances = torch.min(distances, dist)  # Update distances
        farthest = torch.max(distances, dim=1)[1]  # Next farthest point

    return xyz[batch_indices.unsqueeze(-1), centroids]  # Return sampled points
