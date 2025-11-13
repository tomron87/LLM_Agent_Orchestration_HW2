"""
Computational Cost Analysis
============================

Tools for analyzing computational costs including:
- Training time
- Memory usage (CPU and GPU)
- FLOPs (Floating Point Operations)
- Parameter count
- Dataset size and storage

For LLM/AI Agent courses, this provides transparency about resource usage.
"""

import torch
import time
import psutil
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import timedelta
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class CostMetrics:
    """Container for computational cost metrics."""

    # Time metrics
    total_training_time_seconds: float = 0.0
    time_per_epoch_seconds: float = 0.0
    time_per_sample_ms: float = 0.0

    # Memory metrics
    peak_cpu_memory_mb: float = 0.0
    peak_gpu_memory_mb: float = 0.0
    average_cpu_memory_mb: float = 0.0

    # Model metrics
    total_parameters: int = 0
    trainable_parameters: int = 0
    model_size_mb: float = 0.0

    # Data metrics
    training_samples: int = 0
    validation_samples: int = 0
    test_samples: int = 0
    total_samples_processed: int = 0

    # Computational metrics
    estimated_flops: float = 0.0  # Floating point operations
    gpu_utilization_percent: float = 0.0

    # Cost estimates (optional)
    estimated_energy_kwh: float = 0.0
    estimated_co2_kg: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"""
Computational Cost Analysis
===========================

Time Metrics:
  Total Training Time: {timedelta(seconds=int(self.total_training_time_seconds))}
  Time per Epoch:      {self.time_per_epoch_seconds:.2f}s
  Time per Sample:     {self.time_per_sample_ms:.2f}ms

Memory Metrics:
  Peak CPU Memory:     {self.peak_cpu_memory_mb:.1f} MB
  Peak GPU Memory:     {self.peak_gpu_memory_mb:.1f} MB
  Avg CPU Memory:      {self.average_cpu_memory_mb:.1f} MB

Model Metrics:
  Total Parameters:    {self.total_parameters:,}
  Trainable Params:    {self.trainable_parameters:,}
  Model Size:          {self.model_size_mb:.2f} MB

Data Metrics:
  Training Samples:    {self.training_samples:,}
  Validation Samples:  {self.validation_samples:,}
  Test Samples:        {self.test_samples:,}
  Total Processed:     {self.total_samples_processed:,}

Computational Metrics:
  Est. FLOPs:          {self.estimated_flops:.2e}
  GPU Utilization:     {self.gpu_utilization_percent:.1f}%

Environmental Impact:
  Est. Energy:         {self.estimated_energy_kwh:.4f} kWh
  Est. CO2:            {self.estimated_co2_kg:.4f} kg
"""


class CostAnalyzer:
    """
    Analyze and track computational costs during training.

    Example:
        >>> analyzer = CostAnalyzer(model)
        >>> analyzer.start_training()
        >>> # ... training loop ...
        >>> analyzer.end_training(num_epochs=30, num_samples=40000)
        >>> metrics = analyzer.get_metrics()
        >>> print(metrics)
        >>> analyzer.save_report('outputs/cost_analysis.txt')
    """

    def __init__(self, model: torch.nn.Module, device: str = 'cpu'):
        """
        Initialize cost analyzer.

        Args:
            model: PyTorch model to analyze
            device: Device being used ('cpu', 'cuda', 'mps')
        """
        self.model = model
        self.device = device
        self.metrics = CostMetrics()

        # Memory tracking
        self.cpu_memory_samples: List[float] = []
        self.gpu_memory_samples: List[float] = []

        # Timing
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

        # Analyze model
        self._analyze_model()

        logger.info(f"CostAnalyzer initialized for device: {device}")

    def _analyze_model(self):
        """Analyze model parameters and size."""
        try:
            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

            self.metrics.total_parameters = total_params
            self.metrics.trainable_parameters = trainable_params

            # Estimate model size (4 bytes per float32 parameter)
            self.metrics.model_size_mb = (total_params * 4) / (1024 ** 2)

            logger.info(
                f"Model analysis: {total_params:,} total params, "
                f"{trainable_params:,} trainable, "
                f"{self.metrics.model_size_mb:.2f} MB"
            )
        except Exception as e:
            logger.warning(f"Could not analyze model: {e}")

    def start_training(self):
        """Mark start of training."""
        self.start_time = time.time()
        self._sample_memory()
        logger.info("Training cost tracking started")

    def end_training(self, num_epochs: int, num_samples: int):
        """
        Mark end of training and compute final metrics.

        Args:
            num_epochs: Total number of epochs trained
            num_samples: Total number of samples processed
        """
        self.end_time = time.time()
        self._sample_memory()

        # Calculate time metrics
        total_time = self.end_time - self.start_time
        self.metrics.total_training_time_seconds = total_time
        self.metrics.time_per_epoch_seconds = total_time / max(num_epochs, 1)
        self.metrics.time_per_sample_ms = (total_time * 1000) / max(num_samples, 1)
        self.metrics.total_samples_processed = num_samples

        # Calculate memory metrics
        if self.cpu_memory_samples:
            self.metrics.peak_cpu_memory_mb = max(self.cpu_memory_samples)
            self.metrics.average_cpu_memory_mb = sum(self.cpu_memory_samples) / len(self.cpu_memory_samples)

        if self.gpu_memory_samples:
            self.metrics.peak_gpu_memory_mb = max(self.gpu_memory_samples)

        # Estimate environmental impact
        self._estimate_environmental_impact()

        logger.info(f"Training completed in {timedelta(seconds=int(total_time))}")

    def _sample_memory(self):
        """Sample current memory usage."""
        try:
            # CPU memory
            process = psutil.Process(os.getpid())
            cpu_memory_mb = process.memory_info().rss / (1024 ** 2)
            self.cpu_memory_samples.append(cpu_memory_mb)

            # GPU memory
            if self.device.startswith('cuda') and torch.cuda.is_available():
                gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                self.gpu_memory_samples.append(gpu_memory_mb)
            elif self.device == 'mps' and torch.backends.mps.is_available():
                # MPS doesn't provide memory stats easily
                pass

        except Exception as e:
            logger.debug(f"Memory sampling failed: {e}")

    def sample_during_training(self):
        """Call this periodically during training to track memory."""
        self._sample_memory()

    def set_dataset_sizes(self, train_size: int, val_size: int = 0, test_size: int = 0):
        """
        Set dataset sizes for analysis.

        Args:
            train_size: Number of training samples
            val_size: Number of validation samples
            test_size: Number of test samples
        """
        self.metrics.training_samples = train_size
        self.metrics.validation_samples = val_size
        self.metrics.test_samples = test_size

    def _estimate_flops(self) -> float:
        """
        Estimate floating point operations (FLOPs).

        Rough estimate for LSTM:
        FLOPs ≈ 4 * num_layers * hidden_size^2 * sequence_length * batch_size
        """
        try:
            # This is a rough approximation
            # More accurate profiling requires detailed layer-by-layer analysis
            hidden_size = getattr(self.model, 'hidden_size', 128)
            num_layers = getattr(self.model, 'num_layers', 1)
            sequence_length = getattr(self.model, 'sequence_length', 50)

            # Approximate FLOPs per forward pass
            flops_per_sample = 4 * num_layers * (hidden_size ** 2) * sequence_length

            # Total FLOPs
            total_flops = flops_per_sample * self.metrics.total_samples_processed

            return float(total_flops)
        except Exception as e:
            logger.warning(f"Could not estimate FLOPs: {e}")
            return 0.0

    def _estimate_environmental_impact(self):
        """
        Estimate energy consumption and CO2 emissions.

        Uses average power consumption estimates:
        - CPU training: ~50W
        - GPU training: ~250W (depending on GPU model)

        CO2 factor: ~0.5 kg CO2/kWh (average global grid)
        """
        try:
            hours = self.metrics.total_training_time_seconds / 3600

            # Power consumption estimate (Watts)
            if self.device.startswith('cuda'):
                power_watts = 250  # Typical GPU power
            elif self.device == 'mps':
                power_watts = 80  # Apple Silicon efficiency
            else:
                power_watts = 50  # CPU only

            # Energy (kWh)
            energy_kwh = (power_watts * hours) / 1000
            self.metrics.estimated_energy_kwh = energy_kwh

            # CO2 emissions (kg)
            # Using average global grid factor: ~0.5 kg CO2/kWh
            co2_factor = 0.5
            self.metrics.estimated_co2_kg = energy_kwh * co2_factor

            logger.info(
                f"Estimated impact: {energy_kwh:.4f} kWh, "
                f"{self.metrics.estimated_co2_kg:.4f} kg CO2"
            )
        except Exception as e:
            logger.warning(f"Could not estimate environmental impact: {e}")

    def get_metrics(self) -> CostMetrics:
        """
        Get cost metrics.

        Returns:
            CostMetrics object with all computed metrics
        """
        # Update FLOPs estimate
        if self.metrics.total_samples_processed > 0:
            self.metrics.estimated_flops = self._estimate_flops()

        return self.metrics

    def save_report(self, output_path: str):
        """
        Save cost analysis report to file.

        Args:
            output_path: Path to save report
        """
        try:
            metrics = self.get_metrics()
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                f.write(str(metrics))

            logger.info(f"Cost analysis report saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save cost report: {e}")
            raise

    def print_summary(self):
        """Print cost summary to console."""
        print(str(self.get_metrics()))


def compare_configurations(metrics_list: List[CostMetrics], labels: List[str]) -> str:
    """
    Compare cost metrics across multiple configurations.

    Args:
        metrics_list: List of CostMetrics objects
        labels: Labels for each configuration

    Returns:
        Formatted comparison string
    """
    if len(metrics_list) != len(labels):
        raise ValueError("Number of metrics must match number of labels")

    comparison = "Configuration Comparison\n"
    comparison += "=" * 80 + "\n\n"

    # Training time comparison
    comparison += "Training Time:\n"
    for label, metrics in zip(labels, metrics_list):
        comparison += f"  {label:20s}: {timedelta(seconds=int(metrics.total_training_time_seconds))}\n"

    # Memory comparison
    comparison += "\nPeak Memory Usage:\n"
    for label, metrics in zip(labels, metrics_list):
        comparison += f"  {label:20s}: CPU {metrics.peak_cpu_memory_mb:.1f} MB"
        if metrics.peak_gpu_memory_mb > 0:
            comparison += f", GPU {metrics.peak_gpu_memory_mb:.1f} MB"
        comparison += "\n"

    # Parameter comparison
    comparison += "\nModel Parameters:\n"
    for label, metrics in zip(labels, metrics_list):
        comparison += f"  {label:20s}: {metrics.trainable_parameters:,}\n"

    # Environmental impact
    comparison += "\nEnvironmental Impact:\n"
    for label, metrics in zip(labels, metrics_list):
        comparison += f"  {label:20s}: {metrics.estimated_energy_kwh:.4f} kWh, {metrics.estimated_co2_kg:.4f} kg CO2\n"

    return comparison


if __name__ == '__main__':
    # Demonstration
    print("=== Cost Analyzer Demo ===\n")

    # Create dummy model
    import torch.nn as nn

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(5, 128, 2)
            self.fc = nn.Linear(128, 1)
            self.hidden_size = 128
            self.num_layers = 2
            self.sequence_length = 50

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out)

    model = DummyModel()
    analyzer = CostAnalyzer(model, device='cpu')
    analyzer.set_dataset_sizes(train_size=40000, test_size=40000)

    # Simulate training
    analyzer.start_training()
    time.sleep(0.5)  # Simulate training time
    analyzer.end_training(num_epochs=30, num_samples=40000)

    # Print results
    analyzer.print_summary()

    # Save report
    analyzer.save_report('outputs/cost_analysis.txt')
    print("\n✅ Cost analysis completed")
