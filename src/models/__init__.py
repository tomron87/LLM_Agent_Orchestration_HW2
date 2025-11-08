"""
LSTM Models Module
==================

This module contains three LSTM implementations:
1. StatefulLSTM (L=1): Processes one sample at a time with manual state management
2. ConditionedStatefulLSTM (L=1): Uses FiLM conditioning for improved multi-task learning
3. SequenceLSTM (L>1): Processes sequences using sliding window approach

All models perform conditional regression to extract individual frequency
components from a mixed noisy signal.
"""

from .lstm_stateful import StatefulLSTM
from .lstm_conditioned import ConditionedStatefulLSTM
from .lstm_sequence import SequenceLSTM

__all__ = ['StatefulLSTM', 'ConditionedStatefulLSTM', 'SequenceLSTM']
