"""
LSTM Frequency Extraction System
=================================

A PyTorch implementation of an LSTM system for extracting individual frequency
components from noisy mixed signals.

Modules:
    - data: Signal generation and dataset management
    - models: LSTM architectures (L=1 and L>1)
    - training: Training pipeline and optimization
    - evaluation: Metrics and visualization

Authors: Igor Nazarenko, Tom Ron, Roie Gilad
Course: M.Sc. Data Science - LLMs and Multi-Agent Orchestration
Instructor: Dr. Segal Yoram
Date: November 2025
"""

__version__ = '1.0.0'
__authors__ = ['Igor Nazarenko', 'Tom Ron', 'Roie Gilad']
