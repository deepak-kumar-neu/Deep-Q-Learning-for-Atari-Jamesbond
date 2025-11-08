"""
Deep Q-Network Implementation
Author: Deepak Kumar
Course: INFO7375 - Fall 2025

This module implements the CNN architecture for Deep Q-Learning.
Based on the architecture from Mnih et al. (2015) "Human-level control through deep reinforcement learning"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DQN(nn.Module):
    """
    Deep Q-Network with convolutional layers for processing Atari frames.
    
    Architecture:
    - 3 Convolutional layers with ReLU activation
    - 2 Fully connected layers
    - Output layer produces Q-values for each action
    
    This is a lightweight version optimized for MacBook Pro training.
    """
    
    def __init__(self, input_shape: Tuple[int, int, int], n_actions: int):
        """
        Initialize the DQN network.
        
        Args:
            input_shape: Shape of input state (channels, height, width)
                        For Atari: (4, 84, 84) - 4 stacked grayscale frames
            n_actions: Number of possible actions in the environment
        """
        super(DQN, self).__init__()
        
        self.input_shape = input_shape
        self.n_actions = n_actions
        
        # Convolutional layers
        # First conv layer: processes 84x84x4 input
        self.conv1 = nn.Conv2d(
            in_channels=input_shape[0],
            out_channels=32,
            kernel_size=8,
            stride=4
        )
        
        # Second conv layer
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=4,
            stride=2
        )
        
        # Third conv layer
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1
        )
        
        # Calculate size of flattened features
        conv_output_size = self._get_conv_output_size(input_shape)
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.fc2 = nn.Linear(512, n_actions)
        
        # Initialize weights using He initialization
        self._initialize_weights()
    
    def _get_conv_output_size(self, shape: Tuple[int, int, int]) -> int:
        """
        Calculate the size of the flattened convolutional output.
        
        Args:
            shape: Input shape (channels, height, width)
            
        Returns:
            Size of flattened output
        """
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)
            output = self.conv1(dummy_input)
            output = self.conv2(output)
            output = self.conv3(output)
            return int(torch.prod(torch.tensor(output.shape[1:])))
    
    def _initialize_weights(self):
        """
        Initialize network weights using He initialization for ReLU networks.
        This helps with gradient flow during training.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input state tensor of shape (batch_size, channels, height, width)
               Values should be normalized to [0, 1]
        
        Returns:
            Q-values for each action, shape (batch_size, n_actions)
        """
        # Ensure input is float and normalized
        if x.dtype != torch.float32:
            x = x.float() / 255.0
        
        # Convolutional layers with ReLU activation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten the output
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # No activation on output layer
        
        return x
    
    def get_action(self, state: torch.Tensor, epsilon: float = 0.0) -> int:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state observation
            epsilon: Exploration rate (0 = greedy, 1 = random)
        
        Returns:
            Selected action index
        """
        if torch.rand(1).item() < epsilon:
            # Explore: random action
            return torch.randint(0, self.n_actions, (1,)).item()
        else:
            # Exploit: greedy action
            with torch.no_grad():
                q_values = self.forward(state.unsqueeze(0))
                return q_values.argmax(dim=1).item()


class DuelingDQN(nn.Module):
    """
    Dueling DQN Architecture (Optional Enhancement)
    
    Separates state value and advantage functions for better learning.
    Reference: Wang et al. (2016) "Dueling Network Architectures for Deep RL"
    
    This is provided as an optional enhancement for students who want to
    explore beyond the basic DQN implementation.
    """
    
    def __init__(self, input_shape: Tuple[int, int, int], n_actions: int):
        """
        Initialize the Dueling DQN network.
        
        Args:
            input_shape: Shape of input state (channels, height, width)
            n_actions: Number of possible actions
        """
        super(DuelingDQN, self).__init__()
        
        self.input_shape = input_shape
        self.n_actions = n_actions
        
        # Shared convolutional layers
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        conv_output_size = self._get_conv_output_size(input_shape)
        
        # Value stream
        self.value_fc = nn.Linear(conv_output_size, 512)
        self.value = nn.Linear(512, 1)
        
        # Advantage stream
        self.advantage_fc = nn.Linear(conv_output_size, 512)
        self.advantage = nn.Linear(512, n_actions)
        
        self._initialize_weights()
    
    def _get_conv_output_size(self, shape: Tuple[int, int, int]) -> int:
        """Calculate size of flattened convolutional output."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)
            output = self.conv1(dummy_input)
            output = self.conv2(output)
            output = self.conv3(output)
            return int(torch.prod(torch.tensor(output.shape[1:])))
    
    def _initialize_weights(self):
        """Initialize weights using He initialization."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through dueling architecture.
        
        Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        """
        if x.dtype != torch.float32:
            x = x.float() / 255.0
        
        # Shared convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        
        # Value stream
        value = F.relu(self.value_fc(x))
        value = self.value(value)
        
        # Advantage stream
        advantage = F.relu(self.advantage_fc(x))
        advantage = self.advantage(advantage)
        
        # Combine streams
        # Q(s,a) = V(s) + (A(s,a) - mean_a(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values
