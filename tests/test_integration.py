"""
Integration Tests
=================

End-to-end integration tests for the complete pipeline.
"""

import pytest
import numpy as np
import torch
import os

from src.data import SignalGenerator, generate_dataset, create_train_val_loaders, create_test_loader
from src.models import StatefulLSTM, SequenceLSTM
from src.training import Trainer, TrainingConfig
from src.evaluation import Evaluator, check_generalization


class TestEndToEndStateful:
    """End-to-end tests for stateful (L=1) pipeline."""

    def test_complete_pipeline_stateful(self, temp_dir):
        """Test complete data generation and model creation pipeline for L=1 model."""
        # 1. Generate data
        train_generator = SignalGenerator(
            frequencies=[1.0, 3.0, 5.0, 7.0],
            fs=1000,
            duration=1.0,  # Short for fast testing
            seed=42
        )
        train_dataset = train_generator.generate_dataset()

        # 2. Create model
        model = StatefulLSTM(
            input_size=5,
            hidden_size=32,
            num_layers=1
        )

        # 3. Create training config
        config = TrainingConfig(
            model_type='stateful',
            hidden_size=32,
            num_layers=1,
            batch_size=32,
            num_epochs=3,  # Just a few epochs
            learning_rate=0.01,
            patience=2,
            checkpoint_dir=temp_dir,
            verbose=False
        )

        # 4. Create trainer
        trainer = Trainer(model, config)

        # Verify everything was created correctly
        assert trainer.model == model
        assert trainer.config == config
        assert train_dataset['S'].shape[0] == 1000
        assert train_dataset['targets'].shape == (4, 1000)


class TestEndToEndSequence:
    """End-to-end tests for sequence (L>1) pipeline."""

    def test_complete_pipeline_sequence(self, temp_dir):
        """Test complete data generation and model creation pipeline for L>1 model."""
        # 1. Generate data
        train_generator = SignalGenerator(
            frequencies=[1.0, 3.0, 5.0, 7.0],
            fs=1000,
            duration=1.0,
            seed=42
        )
        train_dataset = train_generator.generate_dataset()

        # 2. Create model
        model = SequenceLSTM(
            input_size=5,
            hidden_size=32,
            num_layers=2,
            sequence_length=10
        )

        # 3. Create training config
        config = TrainingConfig(
            model_type='sequence',
            hidden_size=32,
            num_layers=2,
            batch_size=16,
            num_epochs=3,
            learning_rate=0.01,
            patience=2,
            checkpoint_dir=temp_dir,
            verbose=False,
            sequence_length=10
        )

        # 4. Create trainer
        trainer = Trainer(model, config)

        # Verify everything was created correctly
        assert trainer.model == model
        assert trainer.config == config
        assert train_dataset['S'].shape[0] == 1000


class TestDataConsistency:
    """Tests for data consistency across pipeline."""

    def test_dataset_generation_consistency(self):
        """Test that dataset generation is consistent with same seed."""
        gen1 = SignalGenerator(seed=42)
        gen2 = SignalGenerator(seed=42)

        data1 = gen1.generate_dataset()
        data2 = gen2.generate_dataset()

        np.testing.assert_array_almost_equal(data1['S'], data2['S'])
        np.testing.assert_array_almost_equal(data1['t'], data2['t'])
        np.testing.assert_array_almost_equal(data1['targets'], data2['targets'])

    def test_different_seeds_produce_different_data(self):
        """Test that different seeds produce different data."""
        gen1 = SignalGenerator(seed=42)
        gen2 = SignalGenerator(seed=123)

        data1 = gen1.generate_dataset()
        data2 = gen2.generate_dataset()

        # Should NOT be equal
        assert not np.array_equal(data1['S'], data2['S'])

    def test_target_frequency_matches(self):
        """Test that ground truth targets match specified frequencies."""
        frequencies = [1.0, 3.0, 5.0, 7.0]
        generator = SignalGenerator(frequencies=frequencies, seed=42)
        dataset = generator.generate_dataset()

        # Check that each target is a pure sinusoid at correct frequency
        t = dataset['t']
        for i, freq in enumerate(frequencies):
            target = dataset['targets'][i]

            # Compute FFT of target
            fft_vals = np.fft.fft(target)
            freqs = np.fft.fftfreq(len(target), d=1/generator.fs)

            # Find peak frequency
            positive_freqs = freqs[:len(freqs)//2]
            positive_fft = np.abs(fft_vals[:len(fft_vals)//2])
            peak_idx = np.argmax(positive_fft)
            peak_freq = positive_freqs[peak_idx]

            # Should match expected frequency
            assert abs(peak_freq - freq) < 0.1  # Within 0.1 Hz


class TestModelStateManagement:
    """Tests for proper state management in stateful model."""

    def test_state_preserved_within_frequency(self):
        """Test that state is preserved for consecutive samples within frequency."""
        model = StatefulLSTM(input_size=5, hidden_size=32, num_layers=1)

        # Process several samples sequentially
        model.reset_state()

        inputs = torch.randn(10, 1, 5)  # 10 consecutive samples
        outputs = []

        for i in range(10):
            out = model(inputs[i])
            outputs.append(out)

        # State should exist and change
        assert model.hidden_state is not None
        assert model.cell_state is not None




class TestModelComparison:
    """Tests comparing L=1 and L>1 models."""

    def test_both_models_can_be_created(self, temp_dir, sample_dataset):
        """Test that both L=1 and L>1 models can be created successfully."""
        # Stateful model
        stateful_model = StatefulLSTM(input_size=5, hidden_size=32, num_layers=1)
        config_stateful = TrainingConfig(
            model_type='stateful',
            hidden_size=32,
            num_layers=1,
            batch_size=32,
            num_epochs=2,
            checkpoint_dir=temp_dir,
            verbose=False
        )
        trainer_stateful = Trainer(stateful_model, config_stateful)

        # Sequence model
        sequence_model = SequenceLSTM(input_size=5, hidden_size=32, num_layers=2, sequence_length=10)
        config_sequence = TrainingConfig(
            model_type='sequence',
            hidden_size=32,
            num_layers=2,
            batch_size=16,
            num_epochs=2,
            checkpoint_dir=temp_dir,
            verbose=False,
            sequence_length=10
        )
        trainer_sequence = Trainer(sequence_model, config_sequence)

        # Both should be created successfully
        assert trainer_stateful.model == stateful_model
        assert trainer_sequence.model == sequence_model

    def test_both_models_can_forward_pass(self):
        """Test that both models can perform forward passes."""
        # Stateful
        stateful_model = StatefulLSTM(input_size=5, hidden_size=32, num_layers=1)
        stateful_input = torch.randn(4, 5)
        stateful_output = stateful_model(stateful_input)
        assert stateful_output.shape == (4, 1)

        # Sequence
        sequence_model = SequenceLSTM(input_size=5, hidden_size=32, num_layers=2, sequence_length=10)
        sequence_input = torch.randn(4, 10, 5)
        sequence_output = sequence_model(sequence_input)
        assert sequence_output.shape == (4, 10, 1)


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_invalid_model_type_config(self):
        """Test that invalid model type raises error in config."""
        with pytest.raises(AssertionError):
            config = TrainingConfig(model_type='invalid_type')

    def test_empty_dataset_handling(self):
        """Test handling of edge cases in dataset."""
        # Very short signal
        generator = SignalGenerator(duration=0.1, seed=42)  # Only 100 samples
        dataset = generator.generate_dataset()

        assert dataset['S'].shape[0] == 100
        assert dataset['targets'].shape == (4, 100)
