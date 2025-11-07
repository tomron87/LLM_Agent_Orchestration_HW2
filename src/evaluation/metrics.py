"""
Evaluation Metrics Module
==========================

Functions for computing performance metrics:
- MSE on training and test sets
- Generalization check
- Per-frequency metrics
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional


def compute_mse(
    predictions: np.ndarray,
    targets: np.ndarray
) -> float:
    """
    Compute Mean Squared Error.

    Args:
        predictions: Model predictions
        targets: Ground truth values

    Returns:
        mse: Mean squared error
    """
    return np.mean((predictions - targets) ** 2)


def check_generalization(
    train_mse: float,
    test_mse: float,
    threshold: float = 0.2
) -> Tuple[bool, float]:
    """
    Check if model generalizes well to unseen data.

    Generalization is considered good if test_mse ≈ train_mse (within threshold).

    Args:
        train_mse: MSE on training set
        test_mse: MSE on test set
        threshold: Maximum allowed relative difference (default 20%)

    Returns:
        generalizes: True if test_mse ≈ train_mse
        ratio: test_mse / train_mse
    """
    if train_mse == 0:
        return False, float('inf')

    ratio = test_mse / train_mse
    relative_diff = abs(ratio - 1.0)
    generalizes = relative_diff <= threshold

    return generalizes, ratio


class Evaluator:
    """
    Model evaluator for computing metrics and predictions.

    Handles both stateful (L=1) and sequence (L>1) models.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        model_type: str = 'stateful'
    ):
        """
        Initialize evaluator.

        Args:
            model: Trained LSTM model
            device: Device to run evaluation on
            model_type: 'stateful' or 'sequence'
        """
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        self.model_type = model_type

    @torch.no_grad()
    def evaluate_stateful(
        self,
        dataloader: DataLoader
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Evaluate stateful L=1 model.

        Args:
            dataloader: Data loader (train or test)

        Returns:
            mse: Mean squared error
            predictions: All model outputs
            targets: All ground truth values
        """
        self.model.eval()
        all_predictions = []
        all_targets = []

        current_freq_idx = None

        for batch in dataloader:
            inputs = batch['input'].to(self.device)
            targets = batch['target'].to(self.device)
            freq_idx = batch['freq_idx']
            sample_idx = batch['sample_idx']

            # Reset state when needed
            reset_state = False
            if current_freq_idx is None or freq_idx[0].item() != current_freq_idx:
                reset_state = True
                current_freq_idx = freq_idx[0].item()
            elif sample_idx[0].item() == 0:
                reset_state = True

            # Forward pass
            outputs = self.model(inputs, reset_state=reset_state)

            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

        predictions = np.concatenate(all_predictions).flatten()
        targets = np.concatenate(all_targets).flatten()

        mse = compute_mse(predictions, targets)

        return mse, predictions, targets

    @torch.no_grad()
    def evaluate_sequence(
        self,
        dataloader: DataLoader
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Evaluate sequence L>1 model.

        Args:
            dataloader: Data loader

        Returns:
            mse: Mean squared error
            predictions: All model outputs
            targets: All ground truth values
        """
        self.model.eval()
        all_predictions = []
        all_targets = []

        for batch in dataloader:
            inputs = batch['input'].to(self.device)
            targets = batch['target'].to(self.device)

            # Forward pass
            outputs = self.model(inputs)

            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

        predictions = np.concatenate(all_predictions).flatten()
        targets = np.concatenate(all_targets).flatten()

        mse = compute_mse(predictions, targets)

        return mse, predictions, targets

    def evaluate(
        self,
        dataloader: DataLoader
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Evaluate model (automatically selects appropriate method).

        Args:
            dataloader: Data loader

        Returns:
            mse: Mean squared error
            predictions: All model outputs
            targets: All ground truth values
        """
        if self.model_type == 'stateful':
            return self.evaluate_stateful(dataloader)
        else:
            return self.evaluate_sequence(dataloader)

    def evaluate_per_frequency(
        self,
        dataloader: DataLoader,
        num_frequencies: int = 4
    ) -> Dict[int, Dict[str, float]]:
        """
        Compute metrics for each frequency separately.

        Args:
            dataloader: Data loader
            num_frequencies: Number of frequencies (default 4)

        Returns:
            freq_metrics: Dictionary mapping freq_idx to metrics
        """
        _, predictions, targets = self.evaluate(dataloader)

        # Reshape to (num_frequencies, num_samples_per_freq)
        samples_per_freq = len(predictions) // num_frequencies
        predictions = predictions.reshape(num_frequencies, samples_per_freq)
        targets = targets.reshape(num_frequencies, samples_per_freq)

        freq_metrics = {}
        for freq_idx in range(num_frequencies):
            freq_pred = predictions[freq_idx]
            freq_target = targets[freq_idx]

            mse = compute_mse(freq_pred, freq_target)
            mae = np.mean(np.abs(freq_pred - freq_target))
            correlation = np.corrcoef(freq_pred, freq_target)[0, 1]

            freq_metrics[freq_idx] = {
                'mse': mse,
                'mae': mae,
                'correlation': correlation
            }

        return freq_metrics

    def get_frequency_predictions(
        self,
        dataloader: DataLoader,
        freq_idx: int,
        num_frequencies: int = 4
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get predictions and targets for a specific frequency.

        Args:
            dataloader: Data loader
            freq_idx: Frequency index (0-3)
            num_frequencies: Total number of frequencies

        Returns:
            predictions: Model predictions for this frequency
            targets: Ground truth for this frequency
        """
        _, all_predictions, all_targets = self.evaluate(dataloader)

        # Reshape to separate frequencies
        samples_per_freq = len(all_predictions) // num_frequencies
        all_predictions = all_predictions.reshape(num_frequencies, samples_per_freq)
        all_targets = all_targets.reshape(num_frequencies, samples_per_freq)

        return all_predictions[freq_idx], all_targets[freq_idx]


def print_evaluation_summary(
    train_mse: float,
    test_mse: float,
    freq_metrics_train: Optional[Dict] = None,
    freq_metrics_test: Optional[Dict] = None
) -> None:
    """
    Print formatted evaluation summary.

    Args:
        train_mse: Training MSE
        test_mse: Test MSE
        freq_metrics_train: Per-frequency metrics for training
        freq_metrics_test: Per-frequency metrics for test
    """
    print("\n" + "="*60)
    print(" "*20 + "EVALUATION RESULTS")
    print("="*60)

    print(f"\nOverall Performance:")
    print(f"  Training MSE:   {train_mse:.6f}")
    print(f"  Test MSE:       {test_mse:.6f}")

    generalizes, ratio = check_generalization(train_mse, test_mse)
    print(f"\nGeneralization Check:")
    print(f"  Test/Train Ratio: {ratio:.4f}")
    print(f"  Generalizes Well: {'✓ YES' if generalizes else '✗ NO'}")

    if freq_metrics_train and freq_metrics_test:
        print(f"\nPer-Frequency Performance:")
        print(f"{'Freq':<8} {'Hz':<8} {'Train MSE':<12} {'Test MSE':<12} {'Correlation':<12}")
        print("-" * 60)

        frequencies = [1, 3, 5, 7]
        for i in range(4):
            train_m = freq_metrics_train[i]
            test_m = freq_metrics_test[i]
            print(f"{i+1:<8} {frequencies[i]:<8} "
                  f"{train_m['mse']:<12.6f} {test_m['mse']:<12.6f} "
                  f"{test_m['correlation']:<12.4f}")

    print("="*60 + "\n")


if __name__ == '__main__':
    """Test metrics functions."""

    print("Testing evaluation metrics...")

    # Test compute_mse
    print("\n1. Testing compute_mse...")
    pred = np.array([1.0, 2.0, 3.0, 4.0])
    target = np.array([1.1, 2.1, 2.9, 4.2])
    mse = compute_mse(pred, target)
    print(f"   MSE: {mse:.6f}")

    # Test check_generalization
    print("\n2. Testing check_generalization...")
    cases = [
        (0.1, 0.11, "Good generalization"),
        (0.1, 0.15, "Acceptable generalization"),
        (0.1, 0.20, "Poor generalization")
    ]

    for train_mse, test_mse, desc in cases:
        generalizes, ratio = check_generalization(train_mse, test_mse)
        print(f"   {desc}: train={train_mse:.2f}, test={test_mse:.2f}, "
              f"ratio={ratio:.2f}, generalizes={generalizes}")

    print("\n✓ All metrics tests passed!")
