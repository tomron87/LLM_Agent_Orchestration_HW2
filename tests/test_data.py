"""
Tests for Data Generation Module
=================================

Tests for signal_generator.py, dataset.py, and data_loader.py
"""

import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import SignalGenerator, generate_dataset, FrequencyExtractionDataset, SequenceDataset
from src.data import create_dataloader, create_train_val_loaders


class TestSignalGenerator:
    """Tests for SignalGenerator class."""

    def test_initialization(self, signal_generator):
        """Test SignalGenerator initialization."""
        assert signal_generator.frequencies == [1.0, 3.0, 5.0, 7.0]
        assert signal_generator.fs == 1000
        assert signal_generator.duration == 10.0
        assert signal_generator.num_samples == 10000

    def test_time_array_generation(self, signal_generator):
        """Test time array generation."""
        t = signal_generator.generate_time_array()
        assert len(t) == 10000
        assert t[0] == 0.0
        assert np.isclose(t[-1], 9.999)

    def test_noisy_component_generation(self, signal_generator):
        """Test noisy sinusoid generation."""
        t = signal_generator.generate_time_array()
        noisy_signal = signal_generator.generate_noisy_component(0, t)

        assert len(noisy_signal) == 10000
        assert noisy_signal.dtype == np.float64
        # Check amplitude range (should be roughly between -1.2 and 1.2)
        assert -2 < noisy_signal.min() < 2
        assert -2 < noisy_signal.max() < 2

    def test_mixed_signal_generation(self, signal_generator):
        """Test mixed signal generation."""
        S, t = signal_generator.generate_mixed_signal()

        assert len(S) == 10000
        assert len(t) == 10000
        assert S.shape == t.shape
        # Mixed signal should be average of 4 components
        assert -2 < S.min() < 2
        assert -2 < S.max() < 2

    def test_ground_truth_generation(self, signal_generator):
        """Test ground truth target generation."""
        t = signal_generator.generate_time_array()
        target = signal_generator.generate_ground_truth(1, t)  # Frequency index 1 (3 Hz)

        assert len(target) == 10000
        # Pure sinusoid should be between -1 and 1
        assert -1.1 < target.min() < -0.9
        assert 0.9 < target.max() < 1.1

    def test_complete_dataset_generation(self, sample_dataset):
        """Test complete dataset generation."""
        assert 'S' in sample_dataset
        assert 't' in sample_dataset
        assert 'targets' in sample_dataset
        assert 'frequencies' in sample_dataset

        assert sample_dataset['S'].shape == (10000,)
        assert sample_dataset['t'].shape == (10000,)
        assert sample_dataset['targets'].shape == (4, 10000)

    def test_seed_reproducibility(self):
        """Test that same seed produces same results."""
        gen1 = SignalGenerator(seed=123)
        gen2 = SignalGenerator(seed=123)

        S1, _ = gen1.generate_mixed_signal()
        S2, _ = gen2.generate_mixed_signal()

        np.testing.assert_array_almost_equal(S1, S2)

    def test_different_seeds(self):
        """Test that different seeds produce different results."""
        gen1 = SignalGenerator(seed=123)
        gen2 = SignalGenerator(seed=456)

        S1, _ = gen1.generate_mixed_signal()
        S2, _ = gen2.generate_mixed_signal()

        assert not np.array_equal(S1, S2)


class TestFrequencyExtractionDataset:
    """Tests for FrequencyExtractionDataset class."""

    def test_dataset_length(self, sample_dataset):
        """Test dataset returns correct length."""
        dataset = FrequencyExtractionDataset(sample_dataset)
        assert len(dataset) == 40000  # 10000 samples × 4 frequencies

    def test_dataset_getitem(self, sample_dataset):
        """Test dataset __getitem__ method."""
        dataset = FrequencyExtractionDataset(sample_dataset)
        sample = dataset[0]

        assert 'input' in sample
        assert 'target' in sample
        assert 'freq_idx' in sample
        assert 'sample_idx' in sample
        assert 't' in sample

        # Check shapes
        assert sample['input'].shape == (5,)  # S[t] + 4-dim one-hot
        assert sample['target'].shape == (1,)

    def test_one_hot_encoding(self, sample_dataset):
        """Test one-hot encoding is correct."""
        dataset = FrequencyExtractionDataset(sample_dataset)

        # First 10000 samples should have C=[1,0,0,0]
        sample_0 = dataset[0]
        assert sample_0['input'][1:5].tolist() == [1.0, 0.0, 0.0, 0.0]

        # Next 10000 samples should have C=[0,1,0,0]
        sample_10000 = dataset[10000]
        assert sample_10000['input'][1:5].tolist() == [0.0, 1.0, 0.0, 0.0]

    def test_frequency_slice(self, sample_dataset):
        """Test get_frequency_slice method."""
        dataset = FrequencyExtractionDataset(sample_dataset)
        S, targets, t = dataset.get_frequency_slice(0)

        assert len(S) == 10000
        assert len(targets) == 10000
        assert len(t) == 10000


class TestSequenceDataset:
    """Tests for SequenceDataset class."""

    def test_sequence_dataset_length(self, sample_dataset):
        """Test sequence dataset length calculation."""
        dataset = SequenceDataset(sample_dataset, sequence_length=10, stride=1)
        # (10000 - 10 + 1) * 4 frequencies = 9991 * 4 = 39964
        assert len(dataset) == 39964

    def test_sequence_dataset_getitem(self, sample_dataset):
        """Test sequence dataset returns correct shapes."""
        dataset = SequenceDataset(sample_dataset, sequence_length=10)
        sample = dataset[0]

        assert 'input' in sample
        assert 'target' in sample
        assert sample['input'].shape == (10, 5)  # (L, input_size)
        assert sample['target'].shape == (10, 1)  # (L, 1)


class TestDataLoaders:
    """Tests for data loader utilities."""

    def test_create_dataloader_stateful(self, sample_dataset):
        """Test creating dataloader for stateful model."""
        loader = create_dataloader(
            sample_dataset,
            batch_size=32,
            shuffle=False,
            model_type='stateful'
        )

        assert isinstance(loader, DataLoader)
        assert loader.batch_size == 32

        # Test getting a batch
        batch = next(iter(loader))
        assert batch['input'].shape[0] <= 32  # Batch size
        assert batch['input'].shape[1] == 5   # Input dimension

    def test_create_dataloader_sequence(self, sample_dataset):
        """Test creating dataloader for sequence model."""
        loader = create_dataloader(
            sample_dataset,
            batch_size=16,
            shuffle=True,
            model_type='sequence',
            sequence_length=10
        )

        batch = next(iter(loader))
        assert batch['input'].shape[1] == 10  # Sequence length
        assert batch['input'].shape[2] == 5   # Input dimension

    def test_train_val_split(self, sample_dataset):
        """Test train/validation split."""
        train_loader, val_loader = create_train_val_loaders(
            sample_dataset,
            val_split=0.1,
            batch_size=32,
            model_type='stateful'
        )

        # Check that loaders were created
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)

        # Check sizes
        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        assert train_size + val_size == 40000
        assert val_size == pytest.approx(4000, rel=0.1)  # ~10% for validation
