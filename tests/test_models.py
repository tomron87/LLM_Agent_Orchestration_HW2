"""
Tests for Model Architectures
==============================

Tests for StatefulLSTM and SequenceLSTM models.
"""

import pytest
import torch
import torch.nn as nn

from src.models import StatefulLSTM, SequenceLSTM


class TestStatefulLSTM:
    """Tests for StatefulLSTM (L=1) model."""

    def test_initialization(self, stateful_model):
        """Test model initialization."""
        assert isinstance(stateful_model, StatefulLSTM)
        assert stateful_model.input_size == 5
        assert stateful_model.hidden_size == 32
        assert stateful_model.num_layers == 1

    def test_forward_pass_shape(self, stateful_model, sample_input):
        """Test forward pass produces correct output shape."""
        output = stateful_model(sample_input)

        # Output should be (batch_size, 1)
        assert output.shape == (4, 1)

    def test_state_initialization(self, stateful_model):
        """Test that states are initialized to None."""
        assert stateful_model.hidden_state is None
        assert stateful_model.cell_state is None

    def test_state_creation_on_forward(self, stateful_model, sample_input):
        """Test that states are created during first forward pass."""
        # Initially None
        assert stateful_model.hidden_state is None

        # After forward pass
        _ = stateful_model(sample_input)

        assert stateful_model.hidden_state is not None
        assert stateful_model.cell_state is not None
        assert stateful_model.hidden_state.shape == (1, 4, 32)  # (num_layers, batch, hidden)

    def test_state_persistence(self, stateful_model, sample_input):
        """Test that states persist between forward passes."""
        # First forward pass
        _ = stateful_model(sample_input)
        first_state = stateful_model.hidden_state.clone()

        # Second forward pass
        _ = stateful_model(sample_input)
        second_state = stateful_model.hidden_state

        # States should be different (updated)
        assert not torch.equal(first_state, second_state)

    def test_state_reset(self, stateful_model, sample_input):
        """Test state reset functionality."""
        # Forward pass to create state
        _ = stateful_model(sample_input)

        # Reset state
        _ = stateful_model(sample_input, reset_state=True)

        # After reset, state should still exist but be reinitialized
        assert stateful_model.hidden_state is not None

    def test_output_range(self, stateful_model):
        """Test that output is in reasonable range."""
        # Create input that should produce bounded output
        test_input = torch.randn(10, 5)
        output = stateful_model(test_input)

        # Output should be roughly in [-2, 2] range (sinusoid amplitude)
        assert output.min() > -5.0
        assert output.max() < 5.0

    def test_gradient_flow(self, stateful_model, sample_input):
        """Test that gradients flow through the model."""
        target = torch.randn(4, 1)
        output = stateful_model(sample_input)

        loss = nn.MSELoss()(output, target)
        loss.backward()

        # Check that LSTM has gradients
        for name, param in stateful_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_batch_independence(self, stateful_model):
        """Test that different batch samples are processed independently."""
        # Create two different batches
        batch1 = torch.randn(4, 5)
        batch2 = torch.randn(4, 5)

        # Reset state before each
        stateful_model.reset_state()
        output1 = stateful_model(batch1)

        stateful_model.reset_state()
        output2 = stateful_model(batch2)

        # Outputs should be different for different inputs
        assert not torch.equal(output1, output2)

    def test_deterministic_with_reset(self, stateful_model):
        """Test that same input produces same output after reset."""
        test_input = torch.randn(4, 5)

        stateful_model.reset_state()
        output1 = stateful_model(test_input)

        stateful_model.reset_state()
        output2 = stateful_model(test_input)

        torch.testing.assert_close(output1, output2)


class TestSequenceLSTM:
    """Tests for SequenceLSTM (L>1) model."""

    def test_initialization(self, sequence_model):
        """Test model initialization."""
        assert isinstance(sequence_model, SequenceLSTM)
        assert sequence_model.input_size == 5
        assert sequence_model.hidden_size == 32
        assert sequence_model.num_layers == 2
        assert sequence_model.sequence_length == 10

    def test_forward_pass_shape(self, sequence_model, sample_sequence_input):
        """Test forward pass produces correct output shape."""
        output = sequence_model(sample_sequence_input)

        # Output should be (batch_size, sequence_length, 1)
        assert output.shape == (4, 10, 1)

    def test_sequence_processing(self, sequence_model):
        """Test that all timesteps in sequence are processed."""
        # Create a sequence with distinct values
        batch_size = 2
        seq_len = 10
        input_seq = torch.randn(batch_size, seq_len, 5)

        output = sequence_model(input_seq)

        # Each timestep should have an output
        assert output.shape == (batch_size, seq_len, 1)

    def test_output_range(self, sequence_model):
        """Test that output is in reasonable range."""
        test_input = torch.randn(8, 10, 5)
        output = sequence_model(test_input)

        # Output should be roughly in [-2, 2] range
        assert output.min() > -5.0
        assert output.max() < 5.0

    def test_gradient_flow(self, sequence_model, sample_sequence_input):
        """Test that gradients flow through the model."""
        target = torch.randn(4, 10, 1)
        output = sequence_model(sample_sequence_input)

        loss = nn.MSELoss()(output, target)
        loss.backward()

        # Check that LSTM has gradients
        for name, param in sequence_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_different_batch_sizes(self, sequence_model):
        """Test model works with different batch sizes."""
        for batch_size in [1, 4, 16, 32]:
            test_input = torch.randn(batch_size, 10, 5)
            output = sequence_model(test_input)
            assert output.shape == (batch_size, 10, 1)

    def test_temporal_consistency(self, sequence_model):
        """Test that similar sequences produce similar outputs."""
        # Create two very similar sequences
        base_seq = torch.randn(1, 10, 5)
        similar_seq = base_seq + torch.randn(1, 10, 5) * 0.01  # Small noise

        output1 = sequence_model(base_seq)
        output2 = sequence_model(similar_seq)

        # Outputs should be close (not identical due to noise)
        diff = (output1 - output2).abs().mean()
        assert diff < 0.5  # Reasonable threshold

    def test_statelessness(self, sequence_model):
        """Test that sequence model doesn't maintain state between batches."""
        # Put model in eval mode to disable dropout for deterministic testing
        sequence_model.eval()

        torch.manual_seed(42)
        seq1 = torch.randn(4, 10, 5)
        seq2 = torch.randn(4, 10, 5)

        # Process first sequence
        output1_a = sequence_model(seq1)
        output2_a = sequence_model(seq2)

        # Process in reverse order
        output2_b = sequence_model(seq2)
        output1_b = sequence_model(seq1)

        # Same sequences should give same outputs regardless of order
        # Allow small tolerance due to floating point
        torch.testing.assert_close(output1_a, output1_b, rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(output2_a, output2_b, rtol=1e-4, atol=1e-6)


class TestModelComparison:
    """Comparative tests between StatefulLSTM and SequenceLSTM."""

    def test_parameter_counts(self, stateful_model, sequence_model):
        """Test that both models have reasonable parameter counts."""
        stateful_params = sum(p.numel() for p in stateful_model.parameters())
        sequence_params = sum(p.numel() for p in sequence_model.parameters())

        # Both models have same hidden size (32), so parameters should be similar
        # Sequence model has 2 layers vs stateful's 1, so it will have more
        assert sequence_params > stateful_params

        # Both should be in reasonable range
        assert 5_000 < stateful_params < 100_000
        assert 5_000 < sequence_params < 200_000

    def test_both_trainable(self, stateful_model, sequence_model):
        """Test that both models are in training mode by default."""
        assert stateful_model.training
        assert sequence_model.training

    def test_eval_mode(self, stateful_model, sequence_model):
        """Test that both models can switch to eval mode."""
        stateful_model.eval()
        sequence_model.eval()

        assert not stateful_model.training
        assert not sequence_model.training

    def test_device_compatibility(self, stateful_model, sequence_model, device):
        """Test that models can be moved to available device."""
        stateful_model.to(device)
        sequence_model.to(device)

        # Test forward pass on device
        test_input_stateful = torch.randn(4, 5).to(device)
        test_input_sequence = torch.randn(4, 10, 5).to(device)

        output_stateful = stateful_model(test_input_stateful)
        output_sequence = sequence_model(test_input_sequence)

        assert output_stateful.device.type == device
        assert output_sequence.device.type == device
