"""
Sequence LSTM Model (L>1)
==========================

LSTM model that processes sequences of length L using sliding window approach.

This is the alternative implementation that leverages LSTM's built-in temporal
processing capabilities by feeding it sequences of consecutive samples rather
than individual samples.

Advantages over L=1:
- Better utilizes LSTM's sequential processing
- More efficient batching
- Captures temporal context more naturally

Requires justification document explaining:
- Choice of sequence length L
- How it contributes to temporal advantage
- Output handling strategy
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class SequenceLSTM(nn.Module):
    """
    LSTM with L>1: Processes sequences of length L.

    Architecture:
        Input (batch, L, 5) -> LSTM -> Linear -> Output (batch, L, 1)

    where L is the sequence length (e.g., 10 or 50).

    The model processes sliding windows of L consecutive samples, allowing
    it to capture temporal patterns across multiple timesteps simultaneously.

    Attributes:
        input_size (int): Input dimension per timestep (5)
        hidden_size (int): LSTM hidden dimension
        num_layers (int): Number of LSTM layers
        sequence_length (int): Expected sequence length L
        dropout (float): Dropout probability between layers
        lstm (nn.LSTM): LSTM layers
        fc (nn.Linear): Output linear layer
    """

    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 64,
        num_layers: int = 2,
        sequence_length: int = 10,
        dropout: float = 0.2
    ):
        """
        Initialize SequenceLSTM model.

        Args:
            input_size: Input dimension per timestep (default 5)
            hidden_size: LSTM hidden dimension
            num_layers: Number of stacked LSTM layers
            sequence_length: Expected sequence length L
            dropout: Dropout probability (applied if num_layers > 1)
        """
        super(SequenceLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.dropout = dropout

        # LSTM layers with dropout
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Output layer applied to all timesteps
        self.fc = nn.Linear(hidden_size, 1)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch, L, 5)
            hidden: Optional initial hidden state tuple (h_0, c_0)
                   If None, initialized to zeros internally by PyTorch

        Returns:
            output: Predictions for all L timesteps, shape (batch, L, 1)

        Example:
            >>> model = SequenceLSTM(hidden_size=64, sequence_length=10)
            >>> x = torch.randn(32, 10, 5)  # Batch of 32 sequences
            >>> output = model(x)
            >>> print(output.shape)  # torch.Size([32, 10, 1])
        """
        batch_size, seq_len, _ = x.shape

        # LSTM forward pass
        # If hidden is None, LSTM initializes it to zeros internally
        # lstm_out shape: (batch, L, hidden_size)
        # (h_n, c_n) shapes: (num_layers, batch, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x, hidden)

        # Apply output layer to all timesteps
        # Reshape to (batch * L, hidden_size) for efficiency
        lstm_out_flat = lstm_out.reshape(-1, self.hidden_size)

        # Apply linear layer: (batch * L, hidden_size) -> (batch * L, 1)
        output_flat = self.fc(lstm_out_flat)

        # Reshape back to (batch, L, 1)
        output = output_flat.reshape(batch_size, seq_len, 1)

        return output

    def forward_last_only(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass returning only the last timestep prediction.

        Useful when only the final prediction is needed (e.g., for evaluation).

        Args:
            x: Input tensor of shape (batch, L, 5)
            hidden: Optional initial hidden state

        Returns:
            output: Prediction for last timestep only, shape (batch, 1)
        """
        # Get predictions for all timesteps
        full_output = self.forward(x, hidden)

        # Return only last timestep
        return full_output[:, -1, :]

    def num_parameters(self) -> int:
        """
        Count total number of trainable parameters.

        Returns:
            count: Total parameter count
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        """String representation of the model."""
        return (
            f"SequenceLSTM(\n"
            f"  input_size={self.input_size},\n"
            f"  hidden_size={self.hidden_size},\n"
            f"  num_layers={self.num_layers},\n"
            f"  sequence_length={self.sequence_length},\n"
            f"  dropout={self.dropout},\n"
            f"  parameters={self.num_parameters():,}\n"
            f")"
        )


class SequenceLSTMWithAttention(nn.Module):
    """
    Enhanced Sequence LSTM with attention mechanism.

    This is an optional advanced implementation that adds attention
    to help the model focus on relevant timesteps within the sequence.

    Architecture:
        Input -> LSTM -> Attention -> Linear -> Output
    """

    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 64,
        num_layers: int = 2,
        sequence_length: int = 10,
        dropout: float = 0.2
    ):
        """Initialize SequenceLSTM with attention."""
        super(SequenceLSTMWithAttention, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        # Output layer
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention.

        Args:
            x: Input tensor (batch, L, 5)

        Returns:
            output: Predictions (batch, L, 1)
        """
        batch_size, seq_len, _ = x.shape

        # LSTM forward
        lstm_out, _ = self.lstm(x)  # (batch, L, hidden_size)

        # Compute attention weights
        # attention_scores: (batch, L, 1)
        attention_scores = self.attention(lstm_out)

        # Softmax over sequence dimension
        attention_weights = torch.softmax(attention_scores, dim=1)

        # Apply attention: weighted sum over sequence
        # Expand weights: (batch, L, hidden_size)
        weighted_output = lstm_out * attention_weights

        # Apply output layer to all timesteps
        output = self.fc(weighted_output)  # (batch, L, 1)

        return output

    def num_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    """Test SequenceLSTM model."""

    print("Testing SequenceLSTM...")

    # Create model
    print("\n1. Creating model...")
    model = SequenceLSTM(
        input_size=5,
        hidden_size=64,
        num_layers=2,
        sequence_length=10,
        dropout=0.2
    )
    print(model)

    # Test forward pass
    print("\n2. Testing forward pass...")
    x = torch.randn(32, 10, 5)  # Batch of 32 sequences, length 10
    output = model(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output sample values (first sequence, first 3 timesteps):")
    print(f"   {output[0, :3, 0]}")

    # Test forward_last_only
    print("\n3. Testing forward_last_only...")
    output_last = model.forward_last_only(x)
    print(f"   Output shape: {output_last.shape}")
    print(f"   Values match: {torch.allclose(output_last, output[:, -1, :])}")

    # Test with different sequence length
    print("\n4. Testing with different sequence length...")
    x_long = torch.randn(16, 50, 5)  # Longer sequences
    output_long = model(x_long)
    print(f"   Input shape: {x_long.shape}")
    print(f"   Output shape: {output_long.shape}")

    # Test attention model
    print("\n5. Testing SequenceLSTMWithAttention...")
    model_att = SequenceLSTMWithAttention(
        input_size=5,
        hidden_size=64,
        num_layers=2,
        sequence_length=10
    )
    print(f"   Model parameters: {model_att.num_parameters():,}")

    output_att = model_att(x)
    print(f"   Output shape: {output_att.shape}")

    # Compare parameter counts
    print("\n6. Comparing models...")
    print(f"   Standard SequenceLSTM: {model.num_parameters():,} parameters")
    print(f"   With Attention: {model_att.num_parameters():,} parameters")

    print("\n✓ All SequenceLSTM tests passed!")
