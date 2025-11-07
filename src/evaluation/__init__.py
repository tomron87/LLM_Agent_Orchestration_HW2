"""
Evaluation Module
=================

This module handles:
- Performance metrics (MSE, generalization)
- Model evaluation on train/test sets
- Visualization of results (required graphs)
- Training curves and analysis
"""

from .metrics import Evaluator, compute_mse, check_generalization, print_evaluation_summary
from .visualization import Visualizer, plot_all_results

__all__ = [
    'Evaluator',
    'compute_mse',
    'check_generalization',
    'print_evaluation_summary',
    'Visualizer',
    'plot_all_results'
]
