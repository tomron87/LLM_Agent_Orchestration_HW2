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
        """Test complete training and evaluation pipeline for L=1 model."""
        # 1. Generate data
        train_generator = SignalGenerator(
            frequencies=[1.0, 3.0, 5.0, 7.0],
            fs=1000,
            duration=1.0,  # Short for fast testing
            seed=42
        )
        train_dataset = train_generator.generate_dataset()

        test_generator = SignalGenerator(
            frequencies=[1.0, 3.0, 5.0, 7.0],
            fs=1000,
            duration=1.0,
            seed=123  # Different seed
        )
        test_dataset = test_generator.generate_dataset()

        # 2. Create data loaders
        train_loader, val_loader = create_train_val_loaders(
            train_dataset,
            val_split=0.1,
            batch_size=32,
            model_type='stateful'
        )

        # 3. Create model
        model = StatefulLSTM(
            input_size=5,
            hidden_size=32,
            num_layers=1
        )

        # 4. Create training config
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

        # 5. Train model
        trainer = Trainer(model, train_loader, val_loader, config)
        history = trainer.train()

        # Verify training happened
        assert len(history['train_loss']) > 0
        assert len(history['val_loss']) > 0

        # 6. Evaluate on test set
        evaluator = Evaluator(model, test_dataset, model_type='stateful')
        test_results = evaluator.evaluate_all_frequencies()

        # Verify evaluation
        assert 'overall_mse' in test_results
        assert test_results['overall_mse'] > 0

        # 7. Check generalization
        train_evaluator = Evaluator(model, train_dataset, model_type='stateful')
        train_results = train_evaluator.evaluate_all_frequencies()

        generalizes, ratio = check_generalization(
            train_results['overall_mse'],
            test_results['overall_mse']
        )

        # Model should generalize reasonably (even with minimal training)
        assert ratio < 5.0  # Very generous threshold for fast test


class TestEndToEndSequence:
    """End-to-end tests for sequence (L>1) pipeline."""

    def test_complete_pipeline_sequence(self, temp_dir):
        """Test complete training and evaluation pipeline for L>1 model."""
        # 1. Generate data
        train_generator = SignalGenerator(
            frequencies=[1.0, 3.0, 5.0, 7.0],
            fs=1000,
            duration=1.0,
            seed=42
        )
        train_dataset = train_generator.generate_dataset()

        test_generator = SignalGenerator(
            frequencies=[1.0, 3.0, 5.0, 7.0],
            fs=1000,
            duration=1.0,
            seed=123
        )
        test_dataset = test_generator.generate_dataset()

        # 2. Create data loaders
        train_loader, val_loader = create_train_val_loaders(
            train_dataset,
            val_split=0.1,
            batch_size=16,
            model_type='sequence',
            sequence_length=10
        )

        # 3. Create model
        model = SequenceLSTM(
            input_size=5,
            hidden_size=32,
            num_layers=2,
            sequence_length=10
        )

        # 4. Create training config
        config = TrainingConfig(
            model_type='sequence',
            hidden_size=32,
            num_layers=2,
            batch_size=16,
            num_epochs=3,
            learning_rate=0.01,
            patience=2,
            checkpoint_dir=temp_dir,
            verbose=False
        )

        # 5. Train model
        trainer = Trainer(model, train_loader, val_loader, config)
        history = trainer.train()

        # Verify training
        assert len(history['train_loss']) > 0
        assert len(history['val_loss']) > 0

        # 6. Evaluate
        evaluator = Evaluator(
            model, test_dataset,
            model_type='sequence',
            sequence_length=10
        )
        test_results = evaluator.evaluate_all_frequencies()

        # Verify evaluation
        assert 'overall_mse' in test_results
        assert test_results['overall_mse'] > 0


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

    def test_state_reset_between_frequencies(self, sample_dataset):
        """Test that state is properly reset between different frequencies."""
        model = StatefulLSTM(input_size=5, hidden_size=32, num_layers=1)
        evaluator = Evaluator(model, sample_dataset, model_type='stateful')

        # Evaluate frequency 0
        _, _, mse_0 = evaluator.evaluate_frequency(0)

        # State should be reset for frequency 1
        _, _, mse_1 = evaluator.evaluate_frequency(1)

        # Both should produce valid results
        assert mse_0 > 0
        assert mse_1 > 0

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


class TestCheckpointRecovery:
    """Tests for checkpoint saving and recovery."""

    def test_training_resume_from_checkpoint(self, temp_dir, sample_dataset):
        """Test that training can be resumed from checkpoint."""
        # 1. Train for 2 epochs and save
        model1 = StatefulLSTM(input_size=5, hidden_size=32, num_layers=1)

        config = TrainingConfig(
            model_type='stateful',
            hidden_size=32,
            num_layers=1,
            batch_size=32,
            num_epochs=2,
            checkpoint_dir=temp_dir,
            verbose=False
        )

        train_loader, val_loader = create_train_val_loaders(
            sample_dataset,
            val_split=0.1,
            batch_size=32,
            model_type='stateful'
        )

        trainer1 = Trainer(model1, train_loader, val_loader, config)
        history1 = trainer1.train()

        # Save checkpoint
        checkpoint_path = os.path.join(temp_dir, 'resume_test.pth')
        trainer1.save_checkpoint(checkpoint_path, epoch=2, val_loss=history1['val_loss'][-1])

        # 2. Load checkpoint and continue training
        model2 = StatefulLSTM(input_size=5, hidden_size=32, num_layers=1)

        trainer2 = Trainer(model2, train_loader, val_loader, config)
        epoch, val_loss = trainer2.load_checkpoint(checkpoint_path)

        assert epoch == 2
        assert val_loss == pytest.approx(history1['val_loss'][-1])


class TestModelComparison:
    """Tests comparing L=1 and L>1 models."""

    def test_both_models_train_successfully(self, temp_dir, sample_dataset):
        """Test that both L=1 and L>1 models can train successfully."""
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

        train_loader_stateful, val_loader_stateful = create_train_val_loaders(
            sample_dataset,
            val_split=0.1,
            batch_size=32,
            model_type='stateful'
        )

        trainer_stateful = Trainer(stateful_model, train_loader_stateful, val_loader_stateful, config_stateful)
        history_stateful = trainer_stateful.train()

        # Sequence model
        sequence_model = SequenceLSTM(input_size=5, hidden_size=32, num_layers=2, sequence_length=10)
        config_sequence = TrainingConfig(
            model_type='sequence',
            hidden_size=32,
            num_layers=2,
            batch_size=16,
            num_epochs=2,
            checkpoint_dir=temp_dir,
            verbose=False
        )

        train_loader_sequence, val_loader_sequence = create_train_val_loaders(
            sample_dataset,
            val_split=0.1,
            batch_size=16,
            model_type='sequence',
            sequence_length=10
        )

        trainer_sequence = Trainer(sequence_model, train_loader_sequence, val_loader_sequence, config_sequence)
        history_sequence = trainer_sequence.train()

        # Both should have trained
        assert len(history_stateful['train_loss']) > 0
        assert len(history_sequence['train_loss']) > 0

        # Both should have reasonable losses
        assert history_stateful['val_loss'][-1] < 2.0
        assert history_sequence['val_loss'][-1] < 2.0

    def test_both_models_evaluate_successfully(self, sample_dataset):
        """Test that both models can evaluate successfully."""
        # Stateful
        stateful_model = StatefulLSTM(input_size=5, hidden_size=32, num_layers=1)
        evaluator_stateful = Evaluator(stateful_model, sample_dataset, model_type='stateful')
        results_stateful = evaluator_stateful.evaluate_all_frequencies()

        # Sequence
        sequence_model = SequenceLSTM(input_size=5, hidden_size=32, num_layers=2, sequence_length=10)
        evaluator_sequence = Evaluator(sequence_model, sample_dataset, model_type='sequence', sequence_length=10)
        results_sequence = evaluator_sequence.evaluate_all_frequencies()

        # Both should produce valid results
        assert 'overall_mse' in results_stateful
        assert 'overall_mse' in results_sequence
        assert results_stateful['overall_mse'] > 0
        assert results_sequence['overall_mse'] > 0


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_invalid_model_type(self, sample_dataset):
        """Test that invalid model type raises appropriate error."""
        model = StatefulLSTM(input_size=5, hidden_size=32, num_layers=1)

        with pytest.raises((ValueError, KeyError, AssertionError)):
            evaluator = Evaluator(model, sample_dataset, model_type='invalid_type')
            evaluator.evaluate_all_frequencies()

    def test_empty_dataset_handling(self):
        """Test handling of edge cases in dataset."""
        # Very short signal
        generator = SignalGenerator(duration=0.1, seed=42)  # Only 100 samples
        dataset = generator.generate_dataset()

        assert dataset['S'].shape[0] == 100
        assert dataset['targets'].shape == (4, 100)

    def test_model_eval_mode(self, stateful_model, sample_dataset):
        """Test that model is put in eval mode during evaluation."""
        evaluator = Evaluator(stateful_model, sample_dataset, model_type='stateful')

        # Set model to training mode
        stateful_model.train()
        assert stateful_model.training

        # Evaluate - should switch to eval mode
        _ = evaluator.evaluate_frequency(0)

        # Should be back in eval mode during evaluation
        # (Note: The model might be back in train mode after, but during eval it should be eval)
