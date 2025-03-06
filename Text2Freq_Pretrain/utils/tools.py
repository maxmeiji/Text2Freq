import numpy as np
import torch
from fastdtw import fastdtw  
from scipy.spatial.distance import euclidean
class EarlyStopping:
    def __init__(self, patience=100):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def crps(
    target: torch.Tensor,
    samples: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the CRPS-like score for a single sample per instance in the batch.
    No sorting required because we only have one forecast sample.

    Parameters:
    -----------
    target: torch.Tensor
        The target values with shape (batch_size, series_length).
    samples: torch.Tensor
        The forecast samples with shape (batch_size, series_length), where only one forecast per instance.

    Returns:
    --------
    crps: torch.Tensor
        The average CRPS for the batch.
    """
    assert target.shape == samples.shape, f"Shape mismatch: {target.shape} vs {samples.shape}"

    batch_size = target.shape[0]
    series_length = target.shape[1]

    # Calculate the absolute difference between target and samples
    abs_diff = torch.abs(target - samples).sum(dim=1) / series_length  # shape: (batch_size,)

    # Compute beta0 (mean of samples)
    beta0 = samples.sum(dim=1) / series_length  # shape: (batch_size,)

    # Compute beta1 (weighted mean)
    # No sorting required, just use the sample values as they are
    i_array = torch.arange(series_length, device=samples.device).float().unsqueeze(0)  # shape: (1, series_length)
    beta1 = (i_array * samples).sum(dim=1) / (series_length * (series_length - 1))  # shape: (batch_size,)

    # Calculate CRPS for each instance in the batch
    crps = abs_diff + beta0  # shape: (batch_size,)

    # Return the average CRPS score across the batch
    return crps.mean()


def DTW(
    series1: torch.Tensor,
    series2: torch.Tensor,
) -> float:  # Return type updated to `float`
    # Ensure the tensors are on the CPU and convert to numpy arrays
    
    series1 = series1.cpu().numpy() if isinstance(series1, torch.Tensor) else np.asarray(series1)
    series2 = series2.cpu().numpy() if isinstance(series2, torch.Tensor) else np.asarray(series2)
    # Compute DTW Distance using fastdtw
    batch_size = series1.shape[0]
    dtw_score = 0.0
    for i in range(batch_size):
        distance, _ = fastdtw(series1[i], series2[i], dist=2)
        dtw_score += distance
    dtw_score = dtw_score / batch_size
    return dtw_score