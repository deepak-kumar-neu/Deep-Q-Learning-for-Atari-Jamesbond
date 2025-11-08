"""
Experience Replay Buffer for DQN
Author: Deepak Kumar
Course: INFO7375 - Fall 2025

This module implements an experience replay buffer to store and sample transitions.
Experience replay breaks the correlation between consecutive samples and improves learning stability.

Reference: Mnih et al. (2015) - "Experience replay" technique
"""

import numpy as np
from collections import deque
from typing import Tuple, List
import random


class ReplayBuffer:
    """
    Circular buffer for storing experience tuples (s, a, r, s', done).
    
    The replay buffer stores transitions and allows random sampling of minibatches
    for training. This breaks the temporal correlation in the data and leads to
    more stable learning.
    """
    
    def __init__(self, capacity: int, seed: int = None):
        """
        Initialize the replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            seed: Random seed for reproducibility
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """
        Add a new experience tuple to the buffer.
        
        Args:
            state: Current state observation
            action: Action taken
            reward: Reward received
            next_state: Next state observation
            done: Whether the episode terminated
        """
        # Store as tuple
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                 np.ndarray, np.ndarray]:
        """
        Sample a random minibatch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
            Each element is a numpy array
        """
        # Random sample without replacement
        batch = random.sample(self.buffer, batch_size)
        
        # Unzip the batch
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to numpy arrays
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states)
        dones = np.array(dones, dtype=np.float32)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """Return current size of the buffer."""
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        """
        Check if buffer has enough samples for training.
        
        Args:
            batch_size: Required number of samples
            
        Returns:
            True if buffer has at least batch_size samples
        """
        return len(self.buffer) >= batch_size
    
    def clear(self):
        """Clear all experiences from the buffer."""
        self.buffer.clear()


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer (Optional Enhancement)
    
    Samples experiences based on their TD error priority rather than uniformly.
    This allows the agent to learn more from important experiences.
    
    Reference: Schaul et al. (2015) "Prioritized Experience Replay"
    
    This is provided as an optional enhancement for students who want to
    explore beyond the basic implementation.
    """
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4, 
                 beta_increment: float = 0.001, seed: int = None):
        """
        Initialize prioritized replay buffer.
        
        Args:
            capacity: Maximum buffer size
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling exponent
            beta_increment: Increment beta by this amount each sample
            seed: Random seed
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.position = 0
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.buffer = []
        
        if seed is not None:
            np.random.seed(seed)
    
    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        """Add experience with maximum priority."""
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        experience = (state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                 np.ndarray, np.ndarray, np.ndarray, List[int]]:
        """
        Sample batch with prioritization.
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, weights, indices)
        """
        buffer_size = len(self.buffer)
        priorities = self.priorities[:buffer_size]
        
        # Calculate sampling probabilities
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices
        indices = np.random.choice(buffer_size, batch_size, p=probabilities, replace=False)
        
        # Calculate importance sampling weights
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Get experiences
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(next_states), np.array(dones, dtype=np.float32),
                np.array(weights, dtype=np.float32), indices)
    
    def update_priorities(self, indices: List[int], td_errors: np.ndarray):
        """
        Update priorities based on TD errors.
        
        Args:
            indices: Indices of experiences to update
            td_errors: TD errors for those experiences
        """
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = abs(error) + 1e-6  # Small constant to avoid zero priority
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer is ready for sampling."""
        return len(self.buffer) >= batch_size
