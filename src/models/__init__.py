"""
LSTM Models Module
==================

This module contains two LSTM implementations:
1. StatefulLSTM (L=1): Processes one sample at a time with manual state management
2. SequenceLSTM (L>1): Processes sequences using sliding window approach

Both models perform conditional regression to extract individual frequency
components from a mixed noisy signal.
"""

from .lstm_stateful import StatefulLSTM
from .lstm_sequence import SequenceLSTM

__all__ = ['StatefulLSTM', 'SequenceLSTM']
