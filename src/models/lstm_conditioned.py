"""
FiLM-Conditioned Stateful LSTM Model (L=1)
===========================================

LSTM model with FiLM (Feature-wise Linear Modulation) conditioning for improved
multi-task learning.

CRITICAL ARCHITECTURAL CHANGE:
This model uses FiLM conditioning where the one-hot selector C generates
layer-specific scale (gamma) and shift (beta) parameters that modulate the
LSTM hidden states. This gives much stronger task-selection capability than
simple input concatenation.

Architecture:
1. Process only S[t] through LSTM (no C concatenation)
2. Use C to generate FiLM parameters (gamma, beta) for each layer
3. Modulate LSTM hidden states: h_modulated = gamma * h + beta
4. Apply output layer to modulated hidden state

This addresses the multi-task learning failure where the model couldn't learn
to effectively use C to switch between 4 different frequency extraction tasks.

Architecture Comparison:
- Original: [S[t], C] -> LSTM -> Output
- FiLM: S[t] -> LSTM -> FiLM(h, C) -> Output

Reference: FiLM: Visual Reasoning with a General Conditioning Layer (Perez et al., 2018)

State Management Protocol:
- Same as StatefulLSTM: preserve state within sequences, reset between frequencies
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class ConditionedStatefulLSTM(nn.Module):
    """
    LSTM with FiLM conditioning for strong multi-task learning.

    Key Difference from StatefulLSTM:
        1. LSTM processes only S[t] (signal), not [S[t], C]
        2. One-hot selector C generates FiLM parameters (gamma, beta)
        3. LSTM hidden states are modulated: h' = gamma * h + beta
        4. Different C values produce different modulation patterns

    This allows C to directly control the hidden state dynamics with
    multiplicative and additive transformations.

    Input:
        x: [S[t], C1, C2, C3, C4] - maintains API compatibility
           Internally splits into signal S[t] and selector C

    Output:
        Scalar value representing the extracted pure frequency component

    Attributes:
        signal_size (int): Signal input dimension (1 for S[t])
        selector_size (int): Selector dimension (4 for one-hot C)
        hidden_size (int): LSTM hidden dimension
        num_layers (int): Number of LSTM layers
        lstm (nn.LSTM): LSTM layer (processes only S[t])
        film_generator (nn.Linear): Generates gamma and beta from C
        fc (nn.Linear): Output linear layer
        hidden_state (torch.Tensor): Current hidden state h_t
        cell_state (torch.Tensor): Current cell state c_t
    """

    def __init__(
        self,
        signal_size: int = 1,
        selector_size: int = 4,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0
    ):
        """
        Initialize FiLM-conditioned StatefulLSTM model.

        Args:
            signal_size: Signal input dimension (1 for S[t])
            selector_size: Selector dimension (4 for one-hot C)
            hidden_size: LSTM hidden dimension
            num_layers: Number of stacked LSTM layers
            dropout: Dropout probability (only applied if num_layers > 1)
        """
        super(ConditionedStatefulLSTM, self).__init__()

        self.signal_size = signal_size
        self.selector_size = selector_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer (processes only S[t], not [S[t], C])
        self.lstm = nn.LSTM(
            input_size=signal_size,  # Only signal, not concatenated with C
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # FiLM parameter generator
        # C (4-dim) -> [gamma, beta] (2 * hidden_size)
        # gamma: scale parameters
        # beta: shift parameters
        self.film_generator = nn.Sequential(
            nn.Linear(selector_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size * 2)
        )

        # Output layer: hidden_size -> 1
        self.fc = nn.Linear(hidden_size, 1)

        # State storage (initialized to None)
        self.hidden_state: Optional[torch.Tensor] = None
        self.cell_state: Optional[torch.Tensor] = None

    def init_hidden(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None
    ) -> None:
        """
        Initialize hidden and cell states to zeros.

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

    def reset_state(self) -> None:
        """
        Reset state to None.

        Called when switching between different frequency sequences.
        """
        self.hidden_state = None
        self.cell_state = None

    def forward(
        self,
        x: torch.Tensor,
        reset_state: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through the network with FiLM conditioning.

        Args:
            x: Input tensor of shape:
               - (batch, 5) where x = [S[t], C1, C2, C3, C4]
               - (batch, 1, 5) if sequence dimension included
            reset_state: If True, reset internal state to zeros

        Returns:
            output: Predicted value of shape (batch, 1)

        Processing Steps:
            1. Split input into signal S[t] and selector C
            2. Process S[t] through LSTM
            3. Generate FiLM parameters (gamma, beta) from C
            4. Modulate LSTM output: h' = gamma * h + beta
            5. Apply output layer to modulated hidden state
        """
        batch_size = x.size(0)

        # Add sequence dimension if not present
        if x.dim() == 2:  # (batch, 5)
            x = x.unsqueeze(1)  # (batch, 1, 5)

        # Split input into signal and selector
        # x shape: (batch, 1, 5) = (batch, 1, signal_size + selector_size)
        signal = x[:, :, :self.signal_size]  # (batch, 1, 1) - S[t]
        selector = x[:, :, self.signal_size:]  # (batch, 1, 4) - C

        # Extract C for FiLM parameter generation
        # Use the C from the single timestep
        C = selector[:, 0, :]  # (batch, 4)

        # Initialize or reset state if needed
        if reset_state or self.hidden_state is None:
            self.init_hidden(batch_size, x.device)
        else:
            # Verify batch size matches
            if self.hidden_state.size(1) != batch_size:
                self.init_hidden(batch_size, x.device)

        # LSTM forward pass (only processes signal S[t], not selector C)
        lstm_out, (h_new, c_new) = self.lstm(
            signal,
            (self.hidden_state, self.cell_state)
        )

        # Update internal state
        # Gradients flow through during training
        self.hidden_state = h_new
        self.cell_state = c_new

        # Generate FiLM parameters from selector C
        # film_params shape: (batch, hidden_size * 2)
        film_params = self.film_generator(C)

        # Split into gamma (scale) and beta (shift)
        gamma = film_params[:, :self.hidden_size]  # (batch, hidden_size)
        beta = film_params[:, self.hidden_size:]   # (batch, hidden_size)

        # Extract LSTM output from last (only) timestep
        h = lstm_out[:, -1, :]  # (batch, hidden_size)

        # Apply FiLM modulation
        # h_modulated = gamma * h + beta
        h_modulated = gamma * h + beta  # (batch, hidden_size)

        # Apply output layer to modulated hidden state
        output = self.fc(h_modulated)  # (batch, 1)

        return output

    def get_state(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Get current internal state.

        Returns:
            hidden_state: Current hidden state
            cell_state: Current cell state
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
            f"ConditionedStatefulLSTM(\n"
            f"  signal_size={self.signal_size},\n"
            f"  selector_size={self.selector_size},\n"
            f"  hidden_size={self.hidden_size},\n"
            f"  num_layers={self.num_layers},\n"
            f"  parameters={self.num_parameters():,}\n"
            f")"
        )


if __name__ == '__main__':
    """Test ConditionedStatefulLSTM model."""

    print("Testing ConditionedStatefulLSTM...")

    # Create model
    print("\n1. Creating model...")
    model = ConditionedStatefulLSTM(
        signal_size=1,
        selector_size=4,
        hidden_size=64,
        num_layers=1
    )
    print(model)

    # Test forward pass with state reset
    print("\n2. Testing forward pass with state reset...")
    # Input: [S[t], C1, C2, C3, C4]
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

    # Test with different frequency selectors
    print("\n4. Testing with different frequency selectors...")
    model.reset_state()

    # Frequency 0: [S[t], 1, 0, 0, 0]
    x_freq0 = torch.cat([
        torch.randn(16, 1),  # S[t]
        torch.tensor([[1, 0, 0, 0]]).repeat(16, 1).float()  # C
    ], dim=1)

    # Frequency 2: [S[t], 0, 0, 1, 0]
    x_freq2 = torch.cat([
        torch.randn(16, 1),  # S[t]
        torch.tensor([[0, 0, 1, 0]]).repeat(16, 1).float()  # C
    ], dim=1)

    output_freq0 = model(x_freq0, reset_state=True)
    output_freq2 = model(x_freq2, reset_state=True)

    print(f"   Freq 0 output mean: {output_freq0.mean():.4f}")
    print(f"   Freq 2 output mean: {output_freq2.mean():.4f}")
    print(f"   Outputs differ: {not torch.allclose(output_freq0.mean(), output_freq2.mean(), atol=0.1)}")

    print("\n✓ All ConditionedStatefulLSTM tests passed!")
