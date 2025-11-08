"""
Tests for Evaluation and Visualization
=======================================

Tests for metrics calculation and visualization utilities.
"""

import pytest
import numpy as np
import torch
import os
from pathlib import Path

from src.evaluation import (
    Evaluator,
    compute_mse,
    check_generalization,
    print_evaluation_summary,
    Visualizer
)


class TestMetrics:
    """Tests for metric computation functions."""

    def test_compute_mse_perfect_prediction(self):
        """Test MSE with perfect predictions."""
        predictions = np.array([1.0, 2.0, 3.0, 4.0])
        targets = np.array([1.0, 2.0, 3.0, 4.0])

        mse = compute_mse(predictions, targets)

        assert mse == pytest.approx(0.0, abs=1e-6)

    def test_compute_mse_known_error(self):
        """Test MSE with known error."""
        predictions = np.array([1.0, 2.0, 3.0, 4.0])
        targets = np.array([2.0, 3.0, 4.0, 5.0])  # Off by 1

        mse = compute_mse(predictions, targets)

        assert mse == pytest.approx(1.0)

    def test_compute_mse_different_magnitudes(self):
        """Test MSE with different error magnitudes."""
        predictions = np.array([0.0, 0.0, 0.0])
        targets = np.array([1.0, 2.0, 3.0])

        mse = compute_mse(predictions, targets)

        # MSE = (1^2 + 2^2 + 3^2) / 3 = 14/3
        assert mse == pytest.approx(14.0 / 3.0)

    def test_compute_mse_torch_tensors(self):
        """Test MSE with PyTorch tensors."""
        predictions = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.5, 2.5, 3.5])

        mse = compute_mse(predictions, targets)

        # MSE = (0.5^2 + 0.5^2 + 0.5^2) / 3 = 0.25
        assert mse == pytest.approx(0.25)

    def test_check_generalization_good(self):
        """Test generalization check with good generalization."""
        train_mse = 0.05
        test_mse = 0.06

        generalizes, ratio = check_generalization(train_mse, test_mse, threshold=0.2)

        assert generalizes is True
        assert ratio == pytest.approx(1.2)

    def test_check_generalization_poor(self):
        """Test generalization check with poor generalization."""
        train_mse = 0.05
        test_mse = 0.10  # 2x train MSE

        generalizes, ratio = check_generalization(train_mse, test_mse, threshold=0.2)

        assert generalizes is False
        assert ratio == pytest.approx(2.0)

    def test_check_generalization_perfect(self):
        """Test generalization check with perfect generalization."""
        train_mse = 0.05
        test_mse = 0.05

        generalizes, ratio = check_generalization(train_mse, test_mse, threshold=0.2)

        assert generalizes is True
        assert ratio == pytest.approx(1.0)


class TestEvaluator:
    """Tests for Evaluator class."""

    def test_evaluator_initialization_stateful(self, stateful_model, sample_dataset):
        """Test Evaluator initialization for stateful model."""
        evaluator = Evaluator(
            model=stateful_model,
            dataset=sample_dataset,
            model_type='stateful'
        )

        assert evaluator.model == stateful_model
        assert evaluator.model_type == 'stateful'
        assert evaluator.device in ['cuda', 'cpu']

    def test_evaluator_initialization_sequence(self, sequence_model, sample_dataset):
        """Test Evaluator initialization for sequence model."""
        evaluator = Evaluator(
            model=sequence_model,
            dataset=sample_dataset,
            model_type='sequence',
            sequence_length=10
        )

        assert evaluator.model == sequence_model
        assert evaluator.model_type == 'sequence'

    def test_evaluate_frequency_stateful(self, stateful_model, sample_dataset):
        """Test evaluating a single frequency with stateful model."""
        evaluator = Evaluator(
            model=stateful_model,
            dataset=sample_dataset,
            model_type='stateful'
        )

        # Evaluate frequency 0
        predictions, targets, mse = evaluator.evaluate_frequency(freq_idx=0)

        assert len(predictions) == 10000  # Full signal length
        assert len(targets) == 10000
        assert mse >= 0.0
        assert mse < 5.0  # Reasonable upper bound

    def test_evaluate_all_frequencies_stateful(self, stateful_model, sample_dataset):
        """Test evaluating all frequencies with stateful model."""
        evaluator = Evaluator(
            model=stateful_model,
            dataset=sample_dataset,
            model_type='stateful'
        )

        results = evaluator.evaluate_all_frequencies()

        assert len(results) == 4  # 4 frequencies
        assert 'overall_mse' in results

        # Check each frequency has results
        for freq_idx in range(4):
            assert freq_idx in results
            assert 'predictions' in results[freq_idx]
            assert 'targets' in results[freq_idx]
            assert 'mse' in results[freq_idx]

    def test_predictions_shape_consistency(self, stateful_model, sample_dataset):
        """Test that predictions have correct shape."""
        evaluator = Evaluator(
            model=stateful_model,
            dataset=sample_dataset,
            model_type='stateful'
        )

        predictions, targets, _ = evaluator.evaluate_frequency(freq_idx=1)

        assert predictions.shape == targets.shape
        assert isinstance(predictions, np.ndarray)
        assert isinstance(targets, np.ndarray)

    def test_mse_is_positive(self, stateful_model, sample_dataset):
        """Test that MSE is always non-negative."""
        evaluator = Evaluator(
            model=stateful_model,
            dataset=sample_dataset,
            model_type='stateful'
        )

        for freq_idx in range(4):
            _, _, mse = evaluator.evaluate_frequency(freq_idx)
            assert mse >= 0.0


class TestVisualization:
    """Tests for visualization functions."""

    def test_visualizer_initialization(self, temp_dir):
        """Test Visualizer initialization."""
        viz = Visualizer(save_dir=temp_dir)
        assert viz.save_dir == temp_dir
        assert os.path.exists(temp_dir)

    def test_plot_single_frequency_creates_file(self, temp_dir):
        """Test that plot_single_frequency creates an output file."""
        viz = Visualizer(save_dir=temp_dir)

        # Create synthetic data
        t = np.linspace(0, 10, 1000)
        targets = np.sin(2 * np.pi * 3 * t)
        predictions = targets + np.random.randn(len(t)) * 0.1
        mixed = targets + np.random.randn(len(t)) * 0.5

        save_name = 'test_single.png'
        viz.plot_single_frequency_comparison(
            t=t,
            target=targets,
            lstm_output=predictions,
            mixed_signal=mixed,
            freq_hz=3.0,
            save_name=save_name
        )

        output_path = os.path.join(temp_dir, save_name)
        assert os.path.exists(output_path)

    def test_plot_all_frequencies_creates_file(self, temp_dir):
        """Test that plot_all_frequencies creates an output file."""
        viz = Visualizer(save_dir=temp_dir)

        # Create synthetic results for 4 frequencies
        results = {}
        t = np.linspace(0, 10, 1000)

        for freq_idx, freq in enumerate([1.0, 3.0, 5.0, 7.0]):
            targets = np.sin(2 * np.pi * freq * t)
            predictions = targets + np.random.randn(len(t)) * 0.1

            results[freq_idx] = {
                'predictions': predictions,
                'targets': targets,
                'mse': 0.01,
                't': t
            }

        results['overall_mse'] = 0.01

        save_name = 'test_all.png'
        viz.plot_all_frequencies(
            results=results,
            frequencies=[1.0, 3.0, 5.0, 7.0],
            save_name=save_name
        )

        output_path = os.path.join(temp_dir, save_name)
        assert os.path.exists(output_path)

    def test_plot_training_curves_creates_file(self, temp_dir):
        """Test that plot_training_curves creates an output file."""
        viz = Visualizer(save_dir=temp_dir)

        history = {
            'train_loss': [0.5, 0.4, 0.3, 0.25, 0.2],
            'val_loss': [0.55, 0.45, 0.35, 0.28, 0.22]
        }

        save_name = 'test_curves.png'
        viz.plot_training_curves(history, save_name=save_name)

        output_path = os.path.join(temp_dir, save_name)
        assert os.path.exists(output_path)


class TestPrintingUtilities:
    """Tests for printing utilities."""

    def test_print_evaluation_summary(self, capsys):
        """Test that evaluation summary can be printed."""
        results = {
            0: {'mse': 0.05},
            1: {'mse': 0.04},
            2: {'mse': 0.06},
            3: {'mse': 0.03},
            'overall_mse': 0.045
        }

        print_evaluation_summary(results, frequencies=[1.0, 3.0, 5.0, 7.0])

        captured = capsys.readouterr()

        # Check that output contains key information
        assert 'Evaluation Summary' in captured.out or '1.0 Hz' in captured.out
        assert 'Overall MSE' in captured.out or '0.045' in captured.out

    def test_generalization_message(self, capsys):
        """Test generalization check message."""
        generalizes, ratio = check_generalization(0.05, 0.06)

        # The function should work without errors
        assert isinstance(generalizes, (bool, np.bool_))
        assert isinstance(ratio, (float, np.floating))


class TestIntegrationEvaluation:
    """Integration tests for evaluation pipeline."""

    def test_full_evaluation_pipeline_stateful(self, stateful_model, sample_dataset, temp_dir):
        """Test complete evaluation pipeline for stateful model."""
        # 1. Create evaluator
        evaluator = Evaluator(
            model=stateful_model,
            dataset=sample_dataset,
            model_type='stateful'
        )

        # 2. Evaluate all frequencies
        results = evaluator.evaluate_all_frequencies()

        # 3. Generate plots
        viz = Visualizer(save_dir=temp_dir)

        save_name_single = 'eval_single.png'
        viz.plot_single_frequency_comparison(
            t=results[0]['t'],
            target=results[0]['targets'],
            lstm_output=results[0]['predictions'],
            mixed_signal=sample_dataset['S'],
            freq_hz=1.0,
            save_name=save_name_single
        )

        save_name_all = 'eval_all.png'
        viz.plot_all_frequencies(
            results=results,
            frequencies=[1.0, 3.0, 5.0, 7.0],
            save_name=save_name_all
        )

        # 4. Verify outputs
        assert os.path.exists(os.path.join(temp_dir, save_name_single))
        assert os.path.exists(os.path.join(temp_dir, save_name_all))
        assert 'overall_mse' in results
        assert results['overall_mse'] >= 0.0

    def test_full_evaluation_pipeline_sequence(self, sequence_model, sample_dataset, temp_dir):
        """Test complete evaluation pipeline for sequence model."""
        evaluator = Evaluator(
            model=sequence_model,
            dataset=sample_dataset,
            model_type='sequence',
            sequence_length=10
        )

        results = evaluator.evaluate_all_frequencies()

        viz = Visualizer(save_dir=temp_dir)
        save_name = 'eval_seq.png'
        viz.plot_all_frequencies(
            results=results,
            frequencies=[1.0, 3.0, 5.0, 7.0],
            save_name=save_name
        )

        assert os.path.exists(os.path.join(temp_dir, save_name))
        assert 'overall_mse' in results
