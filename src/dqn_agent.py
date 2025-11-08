"""
DQN Agent Implementation
Author: Deepak Kumar
Course: INFO7375 - Fall 2025

This module implements the Deep Q-Network agent with Double DQN enhancement.

References:
- Mnih et al. (2015) "Human-level control through deep reinforcement learning"
- van Hasselt et al. (2016) "Deep Reinforcement Learning with Double Q-learning"
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Optional
import random


class DQNAgent:
    """
    Deep Q-Network Agent with Double DQN.
    
    This agent uses:
    - Experience replay for stable learning
    - Target network for stable Q-value estimates
    - Double DQN to reduce overestimation bias
    - Multiple exploration strategies
    """
    
    def __init__(self, network, n_actions: int, device: str = "cpu",
                 gamma: float = 0.99, learning_rate: float = 0.00025,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.01,
                 epsilon_decay_steps: int = 100000,
                 epsilon_decay_type: str = "exponential",
                 exploration_strategy: str = "epsilon_greedy"):
        """
        Initialize DQN agent.
        
        Args:
            network: Neural network class (not instance)
            n_actions: Number of possible actions
            device: Device to run on ("cpu", "cuda", or "mps")
            gamma: Discount factor for future rewards
            learning_rate: Learning rate for optimizer (alpha in Bellman equation)
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay_steps: Steps to decay epsilon from start to end
            epsilon_decay_type: Type of decay ("linear" or "exponential")
            exploration_strategy: Exploration strategy to use
        """
        self.n_actions = n_actions
        self.device = device
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_decay_type = epsilon_decay_type
        self.exploration_strategy = exploration_strategy
        
        # Initialize networks
        self.online_network = network.to(device)
        self.target_network = network.to(device)
        
        # Copy weights from online to target network
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()  # Target network is always in eval mode
        
        # Optimizer
        self.optimizer = optim.Adam(self.online_network.parameters(), lr=learning_rate)
        
        # Loss function
        self.criterion = nn.SmoothL1Loss()  # Huber loss
        
        # Step counter for epsilon decay
        self.steps = 0
        
        # For Boltzmann exploration
        self.temperature = 1.0
    
    def select_action(self, state: np.ndarray, epsilon: Optional[float] = None) -> int:
        """
        Select action using the current exploration strategy.
        
        Args:
            state: Current state observation
            epsilon: Exploration rate (if None, computed from decay schedule)
            
        Returns:
            Selected action index
        """
        if epsilon is None:
            epsilon = self.get_epsilon()
        
        if self.exploration_strategy == "epsilon_greedy":
            return self._epsilon_greedy(state, epsilon)
        elif self.exploration_strategy == "boltzmann":
            return self._boltzmann_exploration(state)
        elif self.exploration_strategy == "ucb":
            return self._ucb_exploration(state)
        else:
            return self._epsilon_greedy(state, epsilon)
    
    def _epsilon_greedy(self, state: np.ndarray, epsilon: float) -> int:
        """
        Epsilon-greedy action selection.
        
        With probability epsilon, select random action.
        Otherwise, select greedy action.
        
        Args:
            state: Current state
            epsilon: Exploration probability
            
        Returns:
            Selected action
        """
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            return self.get_greedy_action(state)
    
    def _boltzmann_exploration(self, state: np.ndarray) -> int:
        """
        Boltzmann (softmax) exploration.
        
        Selects actions with probability proportional to exp(Q(s,a) / temperature).
        Higher temperature = more exploration.
        
        Args:
            state: Current state
            
        Returns:
            Selected action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.online_network(state_tensor).cpu().numpy()[0]
        
        # Apply softmax with temperature
        exp_q = np.exp((q_values - np.max(q_values)) / self.temperature)
        probabilities = exp_q / exp_q.sum()
        
        return np.random.choice(self.n_actions, p=probabilities)
    
    def _ucb_exploration(self, state: np.ndarray) -> int:
        """
        Upper Confidence Bound exploration (simplified version).
        
        Note: This is a simplified implementation. Full UCB would require
        tracking action counts per state.
        
        Args:
            state: Current state
            
        Returns:
            Selected action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.online_network(state_tensor).cpu().numpy()[0]
        
        # Add exploration bonus (simplified)
        c = 2.0  # Exploration constant
        bonus = c * np.sqrt(np.log(self.steps + 1) / (1 + self.steps / self.n_actions))
        ucb_values = q_values + bonus
        
        return np.argmax(ucb_values)
    
    def get_greedy_action(self, state: np.ndarray) -> int:
        """
        Get greedy action (action with highest Q-value).
        
        Args:
            state: Current state
            
        Returns:
            Greedy action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.online_network(state_tensor)
            return q_values.argmax(dim=1).item()
    
    def learn(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
              next_states: np.ndarray, dones: np.ndarray) -> float:
        """
        Update Q-network using a batch of experiences.
        
        Uses Double DQN update rule:
        Q_target = r + γ * Q_target(s', argmax_a' Q_online(s', a'))
        
        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of done flags
            
        Returns:
            Loss value
        """
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.online_network(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN: use online network to select action, target network to evaluate
        with torch.no_grad():
            # Get best actions from online network
            next_actions = self.online_network(next_states).argmax(dim=1, keepdim=True)
            # Evaluate those actions with target network
            next_q_values = self.target_network(next_states).gather(1, next_actions)
            # Compute target Q values
            target_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q_values
        
        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.online_network.parameters(), 10.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """
        Update target network by copying weights from online network.
        
        This should be called periodically (e.g., every 1000 steps).
        """
        self.target_network.load_state_dict(self.online_network.state_dict())
    
    def get_epsilon(self) -> float:
        """
        Get current epsilon value based on decay schedule.
        
        Returns:
            Current epsilon value
        """
        if self.epsilon_decay_type == "linear":
            # Linear decay
            epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * \
                     min(self.steps / self.epsilon_decay_steps, 1.0)
        else:
            # Exponential decay
            decay_rate = -np.log(self.epsilon_end / self.epsilon_start) / self.epsilon_decay_steps
            epsilon = self.epsilon_start * np.exp(-decay_rate * self.steps)
            epsilon = max(epsilon, self.epsilon_end)
        
        return epsilon
    
    def increment_step(self):
        """Increment step counter for epsilon decay."""
        self.steps += 1
    
    def save(self, filepath: str):
        """
        Save agent state to file.
        
        Args:
            filepath: Path to save checkpoint
        """
        checkpoint = {
            'online_network_state_dict': self.online_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps': self.steps,
            'gamma': self.gamma,
            'epsilon_start': self.epsilon_start,
            'epsilon_end': self.epsilon_end,
            'epsilon_decay_steps': self.epsilon_decay_steps,
        }
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Load agent state from file.
        
        Args:
            filepath: Path to load checkpoint from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.online_network.load_state_dict(checkpoint['online_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps = checkpoint['steps']
        print(f"Model loaded from {filepath}")
