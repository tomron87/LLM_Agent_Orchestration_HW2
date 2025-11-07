"""
Stateful LSTM Model (L=1)
==========================

LSTM model that processes one sample at a time with explicit state management.

CRITICAL FEATURE:
The internal state (hidden_state h_t and cell_state c_t) is preserved between
consecutive time samples within the same frequency sequence. This allows the
LSTM to learn temporal dependencies despite per-sample noise variations.

State Management Protocol:
- State is initialized to zeros at the start of each frequency sequence
- State is preserved (NOT reset) between consecutive time samples
- State must be reset when switching to a different frequency

This implementation demonstrates the pedagogical value of explicit state
management in understanding LSTM's temporal memory capabilities.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class StatefulLSTM(nn.Module):
    """
    LSTM with L=1: Processes one sample at a time with manual state management.

    Architecture:
        Input (5) -> LSTM (hidden_size) -> Linear (1) -> Output

    Input:
        [S[t], C1, C2, C3, C4] where:
        - S[t]: Mixed noisy signal value
        - C: 4-dimensional one-hot frequency selector

    Output:
        Scalar value representing the extracted pure frequency component

    Attributes:
        input_size (int): Input dimension (5)
        hidden_size (int): LSTM hidden dimension
        num_layers (int): Number of LSTM layers
        lstm (nn.LSTM): LSTM layer
        fc (nn.Linear): Output linear layer
        hidden_state (torch.Tensor): Current hidden state h_t
        cell_state (torch.Tensor): Current cell state c_t
    """

    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0
    ):
        """
        Initialize StatefulLSTM model.

        Args:
            input_size: Input dimension (default 5: S[t] + 4-dim one-hot)
            hidden_size: LSTM hidden dimension
            num_layers: Number of stacked LSTM layers
            dropout: Dropout probability (only applied if num_layers > 1)
        """
        super(StatefulLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # Input shape: (batch, seq_len, input_size)
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Output layer: hidden_size -> 1
        self.fc = nn.Linear(hidden_size, 1)

        # State storage (initialized to None)
        self.hidden_state: Optional[torch.Tensor] = None
        self.cell_state: Optional[torch.Tensor] = None

        # Flag to track if state was reset (for validation)
        self.state_was_reset = False

    def init_hidden(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None
    ) -> None:
        """
        Initialize hidden and cell states to zeros.

        Called at the start of each frequency sequence.

        Args:
            batch_size: Batch size
            device: Device to create tensors on
        """
        if device is None:
            device = next(self.parameters()).device

        self.hidden_state = torch.zeros(
            self.num_layers, batch_size, self.hidden_size,
            device=device
        )
        self.cell_state = torch.zeros(
            self.num_layers, batch_size, self.hidden_size,
            device=device
        )

        self.state_was_reset = True

    def reset_state(self) -> None:
        """
        Reset state to None.

        Called when switching between different frequency sequences.
        """
        self.hidden_state = None
        self.cell_state = None
        self.state_was_reset = True

    def forward(
        self,
        x: torch.Tensor,
        reset_state: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape:
               - (batch, 5) if sequence dimension not present, or
               - (batch, 1, 5) if sequence dimension included
            reset_state: If True, reset internal state to zeros before processing

        Returns:
            output: Predicted value of shape (batch, 1)

        State Management:
            - If reset_state=True or state is None: Initialize state to zeros
            - Otherwise: Use existing state from previous forward pass
            - State is updated after each forward pass and detached from graph

        Example:
            >>> model = StatefulLSTM(hidden_size=64)
            >>> x = torch.randn(32, 5)  # Batch of 32 samples
            >>>
            >>> # First sample in sequence - reset state
            >>> output = model(x, reset_state=True)
            >>>
            >>> # Subsequent samples - preserve state
            >>> output = model(x, reset_state=False)
        """
        batch_size = x.size(0)

        # Add sequence dimension if not present
        if x.dim() == 2:  # (batch, 5)
            x = x.unsqueeze(1)  # (batch, 1, 5)

        # Initialize or reset state if needed
        if reset_state or self.hidden_state is None:
            self.init_hidden(batch_size, x.device)
        else:
            # Verify batch size matches
            if self.hidden_state.size(1) != batch_size:
                self.init_hidden(batch_size, x.device)

        self.state_was_reset = False

        # LSTM forward pass
        # lstm_out shape: (batch, 1, hidden_size)
        # (h_new, c_new) shapes: (num_layers, batch, hidden_size)
        lstm_out, (h_new, c_new) = self.lstm(
            x,
            (self.hidden_state, self.cell_state)
        )

        # Update internal state (detach to prevent backprop through time)
        # CRITICAL: This preserves state for next forward pass!
        self.hidden_state = h_new.detach()
        self.cell_state = c_new.detach()

        # Apply output layer to last (only) timestep
        # lstm_out[:, -1, :] extracts last timestep: (batch, hidden_size)
        output = self.fc(lstm_out[:, -1, :])  # (batch, 1)

        return output

    def get_state(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Get current internal state.

        Returns:
            hidden_state: Current hidden state (num_layers, batch, hidden_size)
            cell_state: Current cell state (num_layers, batch, hidden_size)
        """
        return self.hidden_state, self.cell_state

    def set_state(
        self,
        hidden_state: torch.Tensor,
        cell_state: torch.Tensor
    ) -> None:
        """
        Set internal state explicitly.

        Args:
            hidden_state: Hidden state tensor
            cell_state: Cell state tensor
        """
        self.hidden_state = hidden_state
        self.cell_state = cell_state

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
            f"StatefulLSTM(\n"
            f"  input_size={self.input_size},\n"
            f"  hidden_size={self.hidden_size},\n"
            f"  num_layers={self.num_layers},\n"
            f"  parameters={self.num_parameters():,}\n"
            f")"
        )


if __name__ == '__main__':
    """Test StatefulLSTM model."""

    print("Testing StatefulLSTM...")

    # Create model
    print("\n1. Creating model...")
    model = StatefulLSTM(input_size=5, hidden_size=64, num_layers=1)
    print(model)

    # Test forward pass with state reset
    print("\n2. Testing forward pass with state reset...")
    x = torch.randn(32, 5)  # Batch of 32 samples
    output = model(x, reset_state=True)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output sample values: {output[:3, 0]}")

    # Check state was created
    h, c = model.get_state()
    print(f"   Hidden state shape: {h.shape}")
    print(f"   Cell state shape: {c.shape}")

    # Test forward pass preserving state
    print("\n3. Testing forward pass preserving state...")
    h_before = model.hidden_state.clone()
    output2 = model(x, reset_state=False)
    h_after = model.hidden_state

    print(f"   State changed: {not torch.equal(h_before, h_after)}")

    # Test state reset
    print("\n4. Testing state reset...")
    model.reset_state()
    h, c = model.get_state()
    print(f"   State after reset: {h is None and c is None}")

    # Test with sequence dimension already present
    print("\n5. Testing with sequence dimension...")
    x_seq = torch.randn(16, 1, 5)  # (batch, seq_len=1, input_size)
    output3 = model(x_seq, reset_state=True)
    print(f"   Input shape: {x_seq.shape}")
    print(f"   Output shape: {output3.shape}")

    # Test multiple sequential forward passes (simulating training)
    print("\n6. Simulating sequential processing...")
    model.reset_state()

    for t in range(5):
        x_t = torch.randn(1, 5)  # Single sample
        output_t = model(x_t, reset_state=(t == 0))
        print(f"   Step {t}: output = {output_t.item():.4f}")

    print("\n✓ All StatefulLSTM tests passed!")
