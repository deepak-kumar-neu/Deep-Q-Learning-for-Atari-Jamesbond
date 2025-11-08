"""
Deep Q-Network for Atari Games
Author: Deepak Kumar
Course: INFO7375 - Fall 2025
"""

__version__ = "1.0.0"
__author__ = "Deepak Kumar"

from .network import DQN, DuelingDQN
from .dqn_agent import DQNAgent
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from .preprocessing import make_atari_env, get_state_shape
from .utils import *

__all__ = [
    'DQN',
    'DuelingDQN',
    'DQNAgent',
    'ReplayBuffer',
    'PrioritizedReplayBuffer',
    'make_atari_env',
    'get_state_shape',
]
