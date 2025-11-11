"""
Visualization Module
====================

Functions for generating all required visualizations:
- Graph 1: Single frequency detailed comparison
- Graph 2: All four frequencies extraction results
- Training curves
- FFT frequency domain analysis
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
from pathlib import Path


class Visualizer:
    """
    Visualization class for generating all required plots.

    Generates the specific graphs required by the assignment:
    1. Single frequency (f2=3Hz) with Target, LSTM Output, and Mixed Signal
    2. Four subplots showing extraction for all frequencies
    3. Training curves
    4. FFT analysis (optional)
    """

    def __init__(self, save_dir: str = 'outputs/figures'):
        """
        Initialize visualizer.

        Args:
            save_dir: Directory to save figures
        """
        self.save_dir = save_dir
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        # Set matplotlib style
        plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')

    def plot_single_frequency_comparison(
        self,
        t: np.ndarray,
        target: np.ndarray,
        lstm_output: np.ndarray,
        mixed_signal: np.ndarray,
        freq_hz: float = 3.0,
        time_range: Optional[Tuple[float, float]] = None,
        save_name: str = 'graph1_single_frequency.png'
    ) -> None:
        """
        Graph 1: Detailed comparison for one frequency (f2 = 3 Hz).

        Shows:
        - Target (pure sinusoid) as solid green line
        - LSTM output as blue dots/line
        - Mixed noisy signal as gray background

        Args:
            t: Time array
            target: Ground truth pure sinusoid
            lstm_output: LSTM predictions
            mixed_signal: Original mixed noisy signal
            freq_hz: Frequency value in Hz
            time_range: Optional (start, end) time range to plot
            save_name: Filename to save plot
        """
        fig, ax = plt.subplots(figsize=(14, 6))

        # Apply time range if specified
        if time_range:
            start_idx = np.searchsorted(t, time_range[0])
            end_idx = np.searchsorted(t, time_range[1])
            t = t[start_idx:end_idx]
            target = target[start_idx:end_idx]
            lstm_output = lstm_output[start_idx:end_idx]
            mixed_signal = mixed_signal[start_idx:end_idx]

        # Plot mixed signal (background - faded)
        ax.plot(t, mixed_signal, color='gray', alpha=0.3,
                label='Mixed Signal S(t)', linewidth=0.8)

        # Plot target (pure sinusoid - solid line)
        ax.plot(t, target, 'g-', label='Target (Pure Sinusoid)',
                linewidth=2.5, alpha=0.9)

        # Plot LSTM output (dots or line)
        if len(t) > 1000:
            # Use dots for long sequences
            ax.plot(t, lstm_output, 'b.', label='LSTM Output',
                    markersize=2, alpha=0.7)
        else:
            # Use line for shorter sequences
            ax.plot(t, lstm_output, 'b-', label='LSTM Output',
                    linewidth=1.5, alpha=0.8)

        ax.set_xlabel('Time (seconds)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Amplitude', fontsize=13, fontweight='bold')
        ax.set_title(f'Frequency Extraction: {freq_hz} Hz',
                     fontsize=15, fontweight='bold')
        ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.4)

        plt.tight_layout()

        # Save figure
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   Saved: {save_path}")

        plt.show()
        plt.close()

    def plot_all_frequencies(
        self,
        t: np.ndarray,
        targets: np.ndarray,
        lstm_outputs: np.ndarray,
        frequencies: List[float] = [1.0, 3.0, 5.0, 7.0],
        time_range: Optional[Tuple[float, float]] = None,
        save_name: str = 'graph2_all_frequencies.png'
    ) -> None:
        """
        Graph 2: Four subplots showing extraction for all frequencies.

        Args:
            t: Time array
            targets: Ground truth array (4, num_samples)
            lstm_outputs: LSTM predictions (4, num_samples)
            frequencies: List of frequency values in Hz
            time_range: Optional time range to plot
            save_name: Filename to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.flatten()

        # Apply time range if specified
        if time_range:
            start_idx = np.searchsorted(t, time_range[0])
            end_idx = np.searchsorted(t, time_range[1])
            t = t[start_idx:end_idx]
            targets = targets[:, start_idx:end_idx]
            lstm_outputs = lstm_outputs[:, start_idx:end_idx]

        for i, (ax, freq) in enumerate(zip(axes, frequencies)):
            # Plot target
            ax.plot(t, targets[i], 'g-', label='Target',
                    linewidth=2.5, alpha=0.9)

            # Plot LSTM output
            if len(t) > 1000:
                ax.plot(t, lstm_outputs[i], 'b-', label='LSTM',
                        linewidth=1.5, alpha=0.7)
            else:
                ax.plot(t, lstm_outputs[i], 'b-', label='LSTM',
                        linewidth=1.8, alpha=0.8)

            ax.set_title(f'Frequency: {freq} Hz', fontsize=13, fontweight='bold')
            ax.set_xlabel('Time (s)', fontsize=11)
            ax.set_ylabel('Amplitude', fontsize=11)
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(True, alpha=0.4)

        plt.suptitle('Frequency Extraction for All Four Frequencies',
                     fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        # Save figure
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   Saved: {save_path}")

        plt.show()
        plt.close()

    def plot_training_curves(
        self,
        history: Dict,
        save_name: str = 'training_curves.png'
    ) -> None:
        """
        Plot training and validation loss curves.

        Args:
            history: Training history dictionary
            save_name: Filename to save plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        epochs = range(1, len(history['train_loss']) + 1)

        # Loss curves
        ax1.plot(epochs, history['train_loss'], 'b-',
                 label='Training Loss', linewidth=2)
        ax1.plot(epochs, history['val_loss'], 'r-',
                 label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('MSE Loss', fontsize=12, fontweight='bold')
        ax1.set_title('Training Progress', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.4)
        ax1.set_yscale('log')  # Log scale for better visualization

        # Learning rate
        ax2.plot(epochs, history['learning_rate'], 'g-', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
        ax2.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.4)
        ax2.set_yscale('log')

        plt.tight_layout()

        # Save figure
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   Saved: {save_path}")

        plt.show()
        plt.close()

    def plot_fft_analysis(
        self,
        signals: Dict[str, np.ndarray],
        fs: int = 1000,
        save_name: str = 'fft_analysis.png'
    ) -> None:
        """
        Plot FFT analysis comparing frequency content.

        Args:
            signals: Dictionary of signal_name -> signal_array
            fs: Sampling rate in Hz
            save_name: Filename to save plot
        """
        fig, axes = plt.subplots(len(signals), 1, figsize=(12, 4 * len(signals)))

        if len(signals) == 1:
            axes = [axes]

        expected_freqs = [1, 3, 5, 7]

        for ax, (name, signal) in zip(axes, signals.items()):
            # Compute FFT
            N = len(signal)
            fft_vals = np.fft.fft(signal)
            fft_freq = np.fft.fftfreq(N, 1 / fs)

            # Only positive frequencies
            pos_mask = fft_freq > 0
            fft_freq = fft_freq[pos_mask]
            fft_magnitude = np.abs(fft_vals[pos_mask])

            # Plot up to 20 Hz
            plot_mask = fft_freq <= 20
            ax.plot(fft_freq[plot_mask], fft_magnitude[plot_mask], linewidth=1.5)

            # Mark expected frequencies
            for f in expected_freqs:
                ax.axvline(x=f, color='r', linestyle='--', alpha=0.5, linewidth=1)

            ax.set_xlabel('Frequency (Hz)', fontsize=11)
            ax.set_ylabel('Magnitude', fontsize=11)
            ax.set_title(f'FFT Analysis: {name}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.4)

        plt.tight_layout()

        # Save figure
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   Saved: {save_path}")

        plt.show()
        plt.close()

    def plot_error_distribution(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        save_name: str = 'error_distribution.png'
    ) -> None:
        """
        Plot error distribution analysis.

        Args:
            predictions: Model predictions
            targets: Ground truth values
            save_name: Filename to save plot
        """
        errors = predictions - targets

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Histogram
        axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
        axes[0].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Error', fontsize=11)
        axes[0].set_ylabel('Frequency', fontsize=11)
        axes[0].set_title('Error Distribution', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.4)

        # Scatter plot
        axes[1].scatter(targets, predictions, alpha=0.3, s=1)
        axes[1].plot([-1, 1], [-1, 1], 'r--', linewidth=2)  # Perfect prediction line
        axes[1].set_xlabel('Target', fontsize=11)
        axes[1].set_ylabel('Prediction', fontsize=11)
        axes[1].set_title('Predictions vs. Targets', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.4)

        # Error over time
        axes[2].plot(errors[:1000], linewidth=0.5, alpha=0.7)
        axes[2].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[2].set_xlabel('Sample', fontsize=11)
        axes[2].set_ylabel('Error', fontsize=11)
        axes[2].set_title('Error Over Time (first 1000 samples)', fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.4)

        plt.tight_layout()

        # Save figure
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   Saved: {save_path}")

        plt.show()
        plt.close()


def plot_all_results(
    visualizer: Visualizer,
    test_data: Dict,
    predictions: np.ndarray,
    history: Dict,
    frequencies: List[float] = [1.0, 3.0, 5.0, 7.0]
) -> None:
    """
    Generate all required visualizations.

    Args:
        visualizer: Visualizer instance
        test_data: Test dataset dictionary
        predictions: All model predictions (40000,)
        history: Training history
        frequencies: List of frequency values
    """
    print("\nGenerating visualizations...")

    t = test_data['t']
    S = test_data['S']
    targets = test_data['targets']

    # Reshape predictions to (4, 10000)
    num_samples = len(t)
    expected_size = len(frequencies) * num_samples  # 4 * 10000 = 40000

    # For sequence models with overlapping windows, predictions might be larger
    # Truncate to expected size if needed
    if len(predictions) > expected_size:
        print(f"   Note: Truncating predictions from {len(predictions)} to {expected_size} elements")
        predictions = predictions[:expected_size]
    elif len(predictions) < expected_size:
        raise ValueError(f"Predictions size {len(predictions)} is smaller than expected {expected_size}")

    predictions = predictions.reshape(len(frequencies), num_samples)

    # Graph 1: Single frequency (f2 = 3 Hz, index 1)
    print("\n1. Creating Graph 1 (Single Frequency Comparison)...")
    visualizer.plot_single_frequency_comparison(
        t=t,
        target=targets[1],  # Frequency 2 (3 Hz)
        lstm_output=predictions[1],
        mixed_signal=S,
        freq_hz=frequencies[1],
        time_range=(0, 2)  # Show first 2 seconds for clarity
    )

    # Graph 2: All frequencies
    print("\n2. Creating Graph 2 (All Four Frequencies)...")
    visualizer.plot_all_frequencies(
        t=t,
        targets=targets,
        lstm_outputs=predictions,
        frequencies=frequencies,
        time_range=(0, 2)  # Show first 2 seconds
    )

    # Training curves
    print("\n3. Creating Training Curves...")
    visualizer.plot_training_curves(history)

    # FFT analysis
    print("\n4. Creating FFT Analysis...")
    fft_signals = {
        'Mixed Signal': S,
        'Target (f2=3Hz)': targets[1],
        'LSTM Output (f2=3Hz)': predictions[1]
    }
    visualizer.plot_fft_analysis(fft_signals)

    # Error distribution
    print("\n5. Creating Error Distribution...")
    visualizer.plot_error_distribution(
        predictions.flatten(),
        targets.flatten()
    )

    print("\n✓ All visualizations generated!")


if __name__ == '__main__':
    """Test visualization functions."""

    print("Testing Visualizer...")

    # Create dummy data
    t = np.linspace(0, 2, 2000)
    target = np.sin(2 * np.pi * 3 * t)
    lstm_output = target + np.random.normal(0, 0.1, len(t))
    mixed_signal = np.mean([np.sin(2*np.pi*f*t) for f in [1,3,5,7]], axis=0)

    # Test visualizer
    visualizer = Visualizer(save_dir='outputs/figures/test')

    print("\n1. Testing single frequency plot...")
    visualizer.plot_single_frequency_comparison(
        t, target, lstm_output, mixed_signal, freq_hz=3.0
    )

    print("\n2. Testing all frequencies plot...")
    targets = np.array([np.sin(2*np.pi*f*t) for f in [1,3,5,7]])
    outputs = targets + np.random.normal(0, 0.05, targets.shape)
    visualizer.plot_all_frequencies(t, targets, outputs)

    print("\n✓ Visualization tests passed!")
