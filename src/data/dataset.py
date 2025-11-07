"""
PyTorch Dataset Module
======================

Custom Dataset classes for frequency extraction task.

The dataset provides samples in the format:
- Input: [S[t], C1, C2, C3, C4] where C is one-hot frequency selector
- Target: Pure sinusoid value at selected frequency
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, Tuple


class FrequencyExtractionDataset(Dataset):
    """
    PyTorch Dataset for LSTM frequency extraction.

    Dataset structure:
    - Total samples: 40,000 (10,000 time samples × 4 frequencies)
    - Each sample contains:
        * S[t]: Scalar mixed signal value
        * C: 4-dimensional one-hot frequency selector vector
        * target: Ground truth pure sinusoid value
        * metadata: freq_idx, sample_idx, time value

    The dataset is organized such that:
    - Rows 0-9999: Frequency 1 (1 Hz) for all time points
    - Rows 10000-19999: Frequency 2 (3 Hz) for all time points
    - Rows 20000-29999: Frequency 3 (5 Hz) for all time points
    - Rows 30000-39999: Frequency 4 (7 Hz) for all time points

    This organization ensures that consecutive samples within a frequency
    block maintain temporal continuity, which is critical for the L=1
    stateful LSTM implementation.
    """

    def __init__(
        self,
        dataset_dict: Dict,
        transform=None
    ):
        """
        Initialize dataset from generated data dictionary.

        Args:
            dataset_dict: Dictionary from SignalGenerator containing:
                - 'S': Mixed signal (num_samples,)
                - 't': Time array (num_samples,)
                - 'targets': Ground truth (4, num_samples)
                - 'frequencies': List of frequency values
            transform: Optional transform to apply to samples
        """
        self.S = dataset_dict['S']
        self.t = dataset_dict['t']
        self.targets = dataset_dict['targets']
        self.frequencies = dataset_dict['frequencies']
        self.num_frequencies = len(self.frequencies)
        self.num_samples = len(self.S)
        self.transform = transform

        # Validate shapes
        assert self.S.shape == (self.num_samples,), \
            f"S should have shape ({self.num_samples},), got {self.S.shape}"
        assert self.targets.shape == (self.num_frequencies, self.num_samples), \
            f"Targets should have shape ({self.num_frequencies}, {self.num_samples}), got {self.targets.shape}"

    def __len__(self) -> int:
        """
        Return total number of samples.

        Returns:
            length: Total samples = num_frequencies × num_time_samples = 40,000
        """
        return self.num_frequencies * self.num_samples

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample from the dataset.

        The index is mapped to (freq_idx, sample_idx) as follows:
        - freq_idx = idx // num_samples (which frequency: 0-3)
        - sample_idx = idx % num_samples (which time point: 0-9999)

        Args:
            idx: Global index (0 to 39,999)

        Returns:
            sample: Dictionary containing:
                - 'input': FloatTensor of shape (5,) = [S[t], C1, C2, C3, C4]
                - 'target': FloatTensor of shape (1,) = ground truth value
                - 'freq_idx': int, frequency index (0-3)
                - 'sample_idx': int, time sample index (0-9999)
                - 't': float, time value in seconds

        Example:
            >>> dataset[0]  # First sample, freq 1, time 0
            {
                'input': tensor([0.8124, 1., 0., 0., 0.]),  # S[0], C=[1,0,0,0]
                'target': tensor([0.0000]),
                'freq_idx': 0,
                'sample_idx': 0,
                't': 0.0
            }

            >>> dataset[10000]  # First sample, freq 2, time 0
            {
                'input': tensor([0.8124, 0., 1., 0., 0.]),  # S[0], C=[0,1,0,0]
                'target': tensor([0.0000]),
                'freq_idx': 1,
                'sample_idx': 0,
                't': 0.0
            }
        """
        # Decompose global index into frequency and sample indices
        freq_idx = idx // self.num_samples
        sample_idx = idx % self.num_samples

        # Create one-hot encoding for frequency selection
        C = np.zeros(self.num_frequencies, dtype=np.float32)
        C[freq_idx] = 1.0

        # Concatenate S[t] and C to form input vector
        input_vec = np.concatenate([
            [self.S[sample_idx]],  # Scalar S[t]
            C                       # One-hot vector
        ]).astype(np.float32)

        # Get target value
        target_val = self.targets[freq_idx, sample_idx]

        # Get time value
        time_val = self.t[sample_idx]

        # Create sample dictionary
        sample = {
            'input': torch.from_numpy(input_vec),
            'target': torch.tensor([target_val], dtype=torch.float32),
            'freq_idx': freq_idx,
            'sample_idx': sample_idx,
            't': time_val
        }

        # Apply transform if provided
        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_frequency_slice(self, freq_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get all samples for a specific frequency.

        Useful for sequential processing in L=1 stateful mode.

        Args:
            freq_idx: Frequency index (0-3)

        Returns:
            S: Mixed signal array (num_samples,)
            targets: Ground truth array (num_samples,)
            t: Time array (num_samples,)
        """
        assert 0 <= freq_idx < self.num_frequencies, \
            f"freq_idx must be 0-{self.num_frequencies-1}, got {freq_idx}"

        return self.S, self.targets[freq_idx], self.t

    def get_sample_info(self, idx: int) -> str:
        """
        Get human-readable information about a sample.

        Args:
            idx: Global index

        Returns:
            info: String description of the sample
        """
        freq_idx = idx // self.num_samples
        sample_idx = idx % self.num_samples
        freq_val = self.frequencies[freq_idx]
        time_val = self.t[sample_idx]

        info = (
            f"Sample {idx}: "
            f"Frequency {freq_idx+1} ({freq_val} Hz), "
            f"Time sample {sample_idx} (t={time_val:.3f}s)"
        )

        return info


class SequenceDataset(Dataset):
    """
    PyTorch Dataset for L>1 sequence-based LSTM.

    Creates sliding window sequences of length L from the time series.
    Each sequence contains L consecutive samples with the same frequency selection.

    This dataset is used for the alternative L>1 implementation.
    """

    def __init__(
        self,
        dataset_dict: Dict,
        sequence_length: int = 10,
        stride: int = 1,
        transform=None
    ):
        """
        Initialize sequence dataset.

        Args:
            dataset_dict: Dictionary from SignalGenerator
            sequence_length: Length of each sequence (L)
            stride: Step size between consecutive sequences
            transform: Optional transform to apply
        """
        self.S = dataset_dict['S']
        self.t = dataset_dict['t']
        self.targets = dataset_dict['targets']
        self.frequencies = dataset_dict['frequencies']
        self.num_frequencies = len(self.frequencies)
        self.num_samples = len(self.S)
        self.sequence_length = sequence_length
        self.stride = stride
        self.transform = transform

        # Calculate number of sequences per frequency
        self.num_sequences_per_freq = (
            self.num_samples - sequence_length
        ) // stride + 1

    def __len__(self) -> int:
        """
        Return total number of sequences.

        Returns:
            length: num_frequencies × num_sequences_per_freq
        """
        return self.num_frequencies * self.num_sequences_per_freq

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sequence from the dataset.

        Args:
            idx: Global sequence index

        Returns:
            sample: Dictionary containing:
                - 'input': FloatTensor of shape (L, 5)
                - 'target': FloatTensor of shape (L, 1)
                - 'freq_idx': int
                - 'start_idx': int, starting time sample index
        """
        freq_idx = idx // self.num_sequences_per_freq
        seq_idx = idx % self.num_sequences_per_freq

        # Calculate start index for this sequence
        start_idx = seq_idx * self.stride

        # Create one-hot encoding for this frequency
        C = np.zeros(self.num_frequencies, dtype=np.float32)
        C[freq_idx] = 1.0

        # Build sequence
        input_seq = []
        target_seq = []

        for i in range(self.sequence_length):
            sample_idx = start_idx + i

            # Input: [S[t], C1, C2, C3, C4]
            input_vec = np.concatenate([
                [self.S[sample_idx]],
                C
            ]).astype(np.float32)

            input_seq.append(input_vec)

            # Target
            target_val = self.targets[freq_idx, sample_idx]
            target_seq.append([target_val])

        # Convert to tensors
        input_tensor = torch.from_numpy(np.array(input_seq))  # (L, 5)
        target_tensor = torch.tensor(target_seq, dtype=torch.float32)  # (L, 1)

        sample = {
            'input': input_tensor,
            'target': target_tensor,
            'freq_idx': freq_idx,
            'start_idx': start_idx
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == '__main__':
    """Test dataset functionality."""

    print("Testing FrequencyExtractionDataset...")

    # Import signal generator
    from signal_generator import generate_dataset

    # Generate test data
    print("\n1. Generating test data...")
    data = generate_dataset(seed=1)

    # Create dataset
    print("\n2. Creating FrequencyExtractionDataset...")
    dataset = FrequencyExtractionDataset(data)
    print(f"   Total samples: {len(dataset)}")
    assert len(dataset) == 40000, "Expected 40,000 samples"

    # Test sample access
    print("\n3. Testing sample access...")
    sample_0 = dataset[0]
    print(f"   Sample 0 info: {dataset.get_sample_info(0)}")
    print(f"   Input shape: {sample_0['input'].shape}")
    print(f"   Input values: {sample_0['input']}")
    print(f"   Target shape: {sample_0['target'].shape}")
    print(f"   Target value: {sample_0['target'].item():.4f}")

    sample_10000 = dataset[10000]
    print(f"\n   Sample 10000 info: {dataset.get_sample_info(10000)}")
    print(f"   Input values: {sample_10000['input']}")
    print(f"   Target value: {sample_10000['target'].item():.4f}")

    # Test frequency slice
    print("\n4. Testing frequency slice...")
    S, targets, t = dataset.get_frequency_slice(freq_idx=0)
    print(f"   Frequency 0 slice shapes: S={S.shape}, targets={targets.shape}, t={t.shape}")

    # Test SequenceDataset
    print("\n5. Testing SequenceDataset (L=10)...")
    seq_dataset = SequenceDataset(data, sequence_length=10, stride=1)
    print(f"   Total sequences: {len(seq_dataset)}")

    seq_sample = seq_dataset[0]
    print(f"   Sequence input shape: {seq_sample['input'].shape}")
    print(f"   Sequence target shape: {seq_sample['target'].shape}")

    print("\n✓ All dataset tests passed!")
