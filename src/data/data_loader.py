"""
DataLoader Utilities
====================

Functions for creating PyTorch DataLoaders with appropriate configurations
for both stateful (L=1) and sequence-based (L>1) training.
"""

import torch
from torch.utils.data import DataLoader, Subset
from typing import Dict, Tuple
import numpy as np

from .dataset import FrequencyExtractionDataset, SequenceDataset


def create_dataloader(
    dataset_dict: Dict,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
    model_type: str = 'stateful',
    sequence_length: int = 1,
    **kwargs
) -> DataLoader:
    """
    Create DataLoader for training or evaluation.

    Args:
        dataset_dict: Dataset dictionary from SignalGenerator
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle data (should be False for L=1 stateful!)
        num_workers: Number of worker processes for data loading
        model_type: 'stateful' (L=1) or 'sequence' (L>1)
        sequence_length: Sequence length for L>1 model
        **kwargs: Additional arguments passed to DataLoader

    Returns:
        dataloader: PyTorch DataLoader

    Note:
        For L=1 stateful training, shuffle MUST be False to maintain
        temporal continuity within frequency sequences.
    """
    if model_type == 'stateful':
        # L=1: Use FrequencyExtractionDataset
        # CRITICAL: Do NOT shuffle to maintain temporal order
        dataset = FrequencyExtractionDataset(dataset_dict)

        if shuffle:
            print("WARNING: shuffle=True not recommended for stateful L=1 model!")
            print("         Temporal continuity may be disrupted.")

    elif model_type == 'sequence':
        # L>1: Use SequenceDataset
        dataset = SequenceDataset(
            dataset_dict,
            sequence_length=sequence_length,
            stride=1
        )
        # Shuffling is OK for sequence model

    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'stateful' or 'sequence'")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        **kwargs
    )

    return dataloader


def create_train_val_loaders(
    dataset_dict: Dict,
    val_split: float = 0.1,
    batch_size: int = 32,
    model_type: str = 'stateful',
    sequence_length: int = 1,
    num_workers: int = 0,
    seed: int = 42,
    interleave_frequencies: bool = True,
    chunk_size: int = 200
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation DataLoaders with stratified split.

    For L=1 stateful model, frequencies can be interleaved in chunks to help
    the model learn to use the one-hot selector effectively.

    Args:
        dataset_dict: Dataset dictionary from SignalGenerator
        val_split: Fraction of data to use for validation (0-1)
        batch_size: Batch size for DataLoaders
        model_type: 'stateful' (L=1) or 'sequence' (L>1)
        sequence_length: Sequence length for L>1 model
        num_workers: Number of worker processes
        seed: Random seed for split reproducibility
        interleave_frequencies: If True, interleave frequencies in chunks (helps multi-task learning)
        chunk_size: Samples per frequency before switching (only if interleave_frequencies=True)

    Returns:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
    """
    if model_type == 'stateful':
        dataset = FrequencyExtractionDataset(dataset_dict)
        num_frequencies = dataset.num_frequencies
        num_samples_per_freq = dataset.num_samples

        # Calculate samples per frequency for validation
        num_val_per_freq = int(num_samples_per_freq * val_split)
        num_train_per_freq = num_samples_per_freq - num_val_per_freq

        # Create indices for train and val
        train_indices = []
        val_indices = []

        rng = np.random.RandomState(seed)

        if interleave_frequencies:
            # INTERLEAVED: Alternate between frequencies in chunks
            # This helps the model learn to use the one-hot selector C
            num_chunks = (num_train_per_freq + chunk_size - 1) // chunk_size

            for chunk_idx in range(num_chunks):
                for freq_idx in range(num_frequencies):
                    freq_offset = freq_idx * num_samples_per_freq
                    chunk_start = chunk_idx * chunk_size
                    chunk_end = min((chunk_idx + 1) * chunk_size, num_train_per_freq)

                    if chunk_start < num_train_per_freq:
                        chunk_indices = list(range(
                            freq_offset + chunk_start,
                            freq_offset + chunk_end
                        ))
                        train_indices.extend(chunk_indices)

            # Validation: use last samples from each frequency
            for freq_idx in range(num_frequencies):
                freq_offset = freq_idx * num_samples_per_freq
                val_idx = list(range(freq_offset + num_train_per_freq,
                                     freq_offset + num_samples_per_freq))
                val_indices.extend(val_idx)
        else:
            # BLOCK: Original approach - all samples of freq-0, then freq-1, etc.
            for freq_idx in range(num_frequencies):
                freq_offset = freq_idx * num_samples_per_freq

                # Use last val_split fraction for validation to maintain temporal order
                train_idx = list(range(freq_offset, freq_offset + num_train_per_freq))
                val_idx = list(range(freq_offset + num_train_per_freq,
                                     freq_offset + num_samples_per_freq))

                train_indices.extend(train_idx)
                val_indices.extend(val_idx)

        # Create subset datasets
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)

        # Create DataLoaders (no shuffling for stateful)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,  # CRITICAL: Maintain order
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )

    elif model_type == 'sequence':
        dataset = SequenceDataset(
            dataset_dict,
            sequence_length=sequence_length,
            stride=1
        )

        # Random split for sequence model
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size

        generator = torch.Generator().manual_seed(seed)
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, val_size],
            generator=generator
        )

        # Shuffling OK for sequence model
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return train_loader, val_loader


def create_test_loader(
    dataset_dict: Dict,
    batch_size: int = 32,
    model_type: str = 'stateful',
    sequence_length: int = 1,
    num_workers: int = 0
) -> DataLoader:
    """
    Create test DataLoader.

    Args:
        dataset_dict: Test dataset dictionary
        batch_size: Batch size
        model_type: 'stateful' or 'sequence'
        sequence_length: For L>1 model
        num_workers: Number of workers

    Returns:
        test_loader: Test DataLoader
    """
    return create_dataloader(
        dataset_dict,
        batch_size=batch_size,
        shuffle=False,  # Never shuffle test data
        num_workers=num_workers,
        model_type=model_type,
        sequence_length=sequence_length
    )


if __name__ == '__main__':
    """Test DataLoader functionality."""

    print("Testing DataLoader utilities...")

    from signal_generator import generate_dataset

    # Generate test data
    print("\n1. Generating test data...")
    data = generate_dataset(seed=1)

    # Test stateful dataloader
    print("\n2. Creating stateful DataLoader (L=1)...")
    loader_stateful = create_dataloader(
        data,
        batch_size=16,
        shuffle=False,
        model_type='stateful'
    )
    print(f"   Number of batches: {len(loader_stateful)}")

    # Get first batch
    batch = next(iter(loader_stateful))
    print(f"   Batch input shape: {batch['input'].shape}")
    print(f"   Batch target shape: {batch['target'].shape}")
    print(f"   First 3 freq_idx: {batch['freq_idx'][:3]}")
    print(f"   First 3 sample_idx: {batch['sample_idx'][:3]}")

    # Test sequence dataloader
    print("\n3. Creating sequence DataLoader (L=10)...")
    loader_sequence = create_dataloader(
        data,
        batch_size=16,
        shuffle=True,
        model_type='sequence',
        sequence_length=10
    )
    print(f"   Number of batches: {len(loader_sequence)}")

    # Get first batch
    seq_batch = next(iter(loader_sequence))
    print(f"   Batch input shape: {seq_batch['input'].shape}")
    print(f"   Batch target shape: {seq_batch['target'].shape}")

    # Test train/val split
    print("\n4. Creating train/val split (L=1)...")
    train_loader, val_loader = create_train_val_loaders(
        data,
        val_split=0.1,
        batch_size=32,
        model_type='stateful'
    )
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")

    print("\n✓ All DataLoader tests passed!")
