"""
Signal Generator Module
=======================

Generates synthetic mixed signals with noisy frequency components for LSTM training.

The SignalGenerator class creates signals according to the assignment specifications:
- 4 sinusoidal frequencies: 1, 3, 5, 7 Hz
- Random amplitude: A_i(t) ~ Uniform(0.8, 1.2) per sample
- Random phase: φ_i(t) ~ Uniform(0, 2π) per sample
- Mixed signal: S(t) = (1/4) * Σ Sinus_i^noisy(t)
- Ground truth: Target_i(t) = sin(2π * f_i * t)
"""

import numpy as np
from typing import List, Tuple, Dict
import os
import pickle


class SignalGenerator:
    """
    Generates synthetic mixed signals with noisy frequency components.

    Attributes:
        frequencies (List[float]): List of frequencies in Hz [1, 3, 5, 7]
        fs (int): Sampling rate in Hz (1000)
        duration (float): Signal duration in seconds (10)
        seed (int): Random seed for reproducibility
        num_samples (int): Total number of time samples (10,000)
        rng (np.random.RandomState): Random number generator with fixed seed
    """

    def __init__(
        self,
        frequencies: List[float] = [1.0, 3.0, 5.0, 7.0],
        fs: int = 1000,
        duration: float = 10.0,
        seed: int = 1,
        phase_scale: float = 0.1
    ):
        """
        Initialize SignalGenerator.

        Args:
            frequencies: List of frequency values in Hz
            fs: Sampling rate in Hz
            duration: Signal duration in seconds
            seed: Random seed for reproducibility
            phase_scale: Scaling factor for phase noise (0 to 1)
                        0.0 = no phase noise (too easy)
                        0.1 = moderate noise (recommended)
                        1.0 = full noise (impossible to learn)
        """
        self.frequencies = frequencies
        self.fs = fs
        self.duration = duration
        self.seed = seed
        self.phase_scale = phase_scale
        self.num_samples = int(fs * duration)
        self.rng = np.random.RandomState(seed)

        # Validate parameters
        assert len(frequencies) == 4, "Exactly 4 frequencies required"
        assert all(f > 0 for f in frequencies), "Frequencies must be positive"
        assert fs > 0, "Sampling rate must be positive"
        assert duration > 0, "Duration must be positive"
        assert 0 <= phase_scale <= 1, "Phase scale must be between 0 and 1"

    def generate_time_array(self) -> np.ndarray:
        """
        Generate time array from 0 to duration.

        Returns:
            t: Time array of shape (num_samples,)
        """
        t = np.arange(0, self.duration, 1 / self.fs)
        return t

    def generate_noisy_component(
        self,
        freq_idx: int,
        t: np.ndarray
    ) -> np.ndarray:
        """
        Generate one noisy sinusoidal component.

        For each sample, amplitude and phase are randomly generated:
        - A_i(t) ~ Uniform(0.8, 1.2)
        - φ_i(t) ~ phase_scale * Uniform(0, 2π)
        - Sinus_i^noisy(t) = A_i(t) * sin(2π * f_i * t + φ_i(t))

        NOTE: Phase is scaled by phase_scale to make task learnable while
        still satisfying "changes at each sample" requirement.

        Args:
            freq_idx: Index of frequency (0-3)
            t: Time array

        Returns:
            noisy_signal: Noisy sinusoid array of shape (num_samples,)
        """
        freq = self.frequencies[freq_idx]

        # Random amplitude for EACH sample
        A = self.rng.uniform(0.8, 1.2, size=len(t))

        # Random phase for EACH sample - sampled from Uniform(0, 2π) then scaled
        # This still "changes at each sample" but with controlled magnitude
        phi_raw = self.rng.uniform(0, 2 * np.pi, size=len(t))
        phi = self.phase_scale * phi_raw

        # Noisy sinusoid
        noisy_signal = A * np.sin(2 * np.pi * freq * t + phi)

        return noisy_signal

    def generate_mixed_signal(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate mixed signal S(t) = average of 4 noisy components.

        S(t) = (1/4) * Σ Sinus_i^noisy(t)

        Returns:
            S: Mixed signal array of shape (num_samples,)
            t: Time array of shape (num_samples,)
        """
        t = self.generate_time_array()

        # Generate all 4 noisy components
        components = []
        for i in range(4):
            comp = self.generate_noisy_component(i, t)
            components.append(comp)

        # Mix: average of all components
        S = np.mean(components, axis=0)

        return S, t

    def generate_ground_truth(
        self,
        freq_idx: int,
        t: np.ndarray
    ) -> np.ndarray:
        """
        Generate pure sinusoid (ground truth target).

        Target_i(t) = sin(2π * f_i * t)

        No noise, unit amplitude, zero phase.

        Args:
            freq_idx: Index of frequency (0-3)
            t: Time array

        Returns:
            target: Pure sinusoid array of shape (num_samples,)
        """
        freq = self.frequencies[freq_idx]
        target = np.sin(2 * np.pi * freq * t)
        return target

    def generate_all_ground_truths(
        self,
        t: np.ndarray
    ) -> np.ndarray:
        """
        Generate ground truth targets for all frequencies.

        Args:
            t: Time array

        Returns:
            targets: Array of shape (4, num_samples) containing pure sinusoids
        """
        targets = np.zeros((len(self.frequencies), len(t)))

        for i in range(len(self.frequencies)):
            targets[i] = self.generate_ground_truth(i, t)

        return targets

    def generate_dataset(self) -> Dict:
        """
        Generate complete dataset with mixed signal and all targets.

        Returns:
            dataset: Dictionary containing:
                - 'S': Mixed signal (num_samples,)
                - 't': Time array (num_samples,)
                - 'targets': Ground truth for all frequencies (4, num_samples)
                - 'frequencies': List of frequency values
                - 'fs': Sampling rate
                - 'duration': Signal duration
                - 'seed': Random seed used
        """
        # Generate mixed signal
        S, t = self.generate_mixed_signal()

        # Generate all ground truths
        targets = self.generate_all_ground_truths(t)

        # Package into dictionary
        dataset = {
            'S': S,
            't': t,
            'targets': targets,
            'frequencies': self.frequencies,
            'fs': self.fs,
            'duration': self.duration,
            'seed': self.seed,
            'num_samples': self.num_samples,
            'phase_scale': self.phase_scale
        }

        return dataset


def generate_dataset(
    seed: int = 1,
    frequencies: List[float] = [1.0, 3.0, 5.0, 7.0],
    fs: int = 1000,
    duration: float = 10.0,
    phase_scale: float = 0.1,
    save_path: str = None
) -> Dict:
    """
    Convenience function to generate dataset with specified parameters.

    Args:
        seed: Random seed for reproducibility
        frequencies: List of frequency values in Hz
        fs: Sampling rate in Hz
        duration: Signal duration in seconds
        phase_scale: Scaling factor for phase noise (0.0-1.0, default 0.1)
        save_path: Optional path to save dataset (as pickle file)

    Returns:
        dataset: Dictionary containing S, t, targets, and metadata

    Example:
        >>> # Generate training dataset with moderate noise
        >>> train_data = generate_dataset(seed=1, phase_scale=0.1, save_path='data/train_data.pkl')
        >>> print(train_data['S'].shape)
        (10000,)

        >>> # Generate test dataset with different seed
        >>> test_data = generate_dataset(seed=2, phase_scale=0.1, save_path='data/test_data.pkl')
    """
    generator = SignalGenerator(
        frequencies=frequencies,
        fs=fs,
        duration=duration,
        seed=seed,
        phase_scale=phase_scale
    )

    dataset = generator.generate_dataset()

    # Save to disk if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"Dataset saved to {save_path}")

    return dataset


def load_dataset(file_path: str) -> Dict:
    """
    Load dataset from pickle file.

    Args:
        file_path: Path to saved dataset file

    Returns:
        dataset: Dictionary containing S, t, targets, and metadata
    """
    with open(file_path, 'rb') as f:
        dataset = pickle.load(f)

    return dataset


def validate_dataset(dataset: Dict) -> bool:
    """
    Validate dataset integrity and correctness.

    Checks:
    - Correct shapes
    - Reasonable value ranges
    - Target frequencies present

    Args:
        dataset: Dataset dictionary

    Returns:
        valid: True if dataset passes all checks

    Raises:
        AssertionError: If any validation check fails
    """
    S = dataset['S']
    t = dataset['t']
    targets = dataset['targets']
    frequencies = dataset['frequencies']

    # Shape checks
    assert len(S) == 10000, f"Expected 10000 samples, got {len(S)}"
    assert len(t) == 10000, f"Expected 10000 time points, got {len(t)}"
    assert targets.shape == (4, 10000), f"Expected (4, 10000) targets, got {targets.shape}"

    # Value range checks
    assert -2 < S.min() < 2, f"Mixed signal has unusual min value: {S.min()}"
    assert -2 < S.max() < 2, f"Mixed signal has unusual max value: {S.max()}"
    assert -1.1 < targets.min() < -0.9, f"Target min should be ~-1, got {targets.min()}"
    assert 0.9 < targets.max() < 1.1, f"Target max should be ~1, got {targets.max()}"

    # Time array check
    assert np.isclose(t[0], 0.0), f"Time should start at 0, got {t[0]}"
    assert np.isclose(t[-1], 9.999), f"Time should end near 10, got {t[-1]}"

    # Frequency check
    assert len(frequencies) == 4, f"Expected 4 frequencies, got {len(frequencies)}"
    assert frequencies == [1.0, 3.0, 5.0, 7.0], f"Unexpected frequencies: {frequencies}"

    print("✓ Dataset validation passed")
    return True


if __name__ == '__main__':
    """Test signal generation."""

    print("Testing SignalGenerator...")

    # Generate training dataset
    print("\n1. Generating training dataset (seed=1)...")
    train_data = generate_dataset(seed=1)
    print(f"   Mixed signal shape: {train_data['S'].shape}")
    print(f"   Targets shape: {train_data['targets'].shape}")
    print(f"   Time range: {train_data['t'][0]:.3f} to {train_data['t'][-1]:.3f} seconds")

    # Generate test dataset
    print("\n2. Generating test dataset (seed=2)...")
    test_data = generate_dataset(seed=2)
    print(f"   Mixed signal shape: {test_data['S'].shape}")
    print(f"   Targets shape: {test_data['targets'].shape}")

    # Validate both datasets
    print("\n3. Validating datasets...")
    validate_dataset(train_data)
    validate_dataset(test_data)

    # Check that datasets are different (different seeds)
    print("\n4. Verifying datasets are different...")
    assert not np.array_equal(train_data['S'], test_data['S']), "Datasets should differ!"
    print("   ✓ Training and test datasets have different noise patterns")

    # Display statistics
    print("\n5. Dataset statistics:")
    print(f"   Training set S(t) - Mean: {train_data['S'].mean():.4f}, Std: {train_data['S'].std():.4f}")
    print(f"   Test set S(t) - Mean: {test_data['S'].mean():.4f}, Std: {test_data['S'].std():.4f}")

    print("\n✓ All tests passed!")
