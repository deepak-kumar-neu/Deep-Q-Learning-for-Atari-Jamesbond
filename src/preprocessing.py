"""
Frame Preprocessing for Atari Environments
Author: Deepak Kumar
Course: INFO7375 - Fall 2025

This module provides preprocessing utilities for Atari game frames.
Standard preprocessing includes grayscaling, resizing, and frame stacking.

Based on preprocessing techniques from:
- Mnih et al. (2015) "Human-level control through deep reinforcement learning"
- Machado et al. (2018) "Revisiting the Arcade Learning Environment"
"""

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from collections import deque
import cv2
from typing import Tuple

# Register ALE environments with gymnasium
try:
    import ale_py
    gym.register_envs(ale_py)
except ImportError:
    print("Warning: ale_py not found. Atari environments may not be available.")
except Exception as e:
    print(f"Warning: Could not register ALE environments: {e}")


class AtariPreprocessing(gym.Wrapper):
    """
    Atari 2600 preprocessing wrapper.
    
    This wrapper:
    1. Converts frames to grayscale
    2. Resizes frames to 84x84
    3. Normalizes pixel values to [0, 1]
    4. Handles frame skipping (action repeat)
    """
    
    def __init__(self, env: gym.Env, frame_skip: int = 4, 
                 frame_width: int = 84, frame_height: int = 84,
                 grayscale: bool = True, normalize: bool = True):
        """
        Initialize preprocessing wrapper.
        
        Args:
            env: Gymnasium environment
            frame_skip: Number of frames to skip (action repeat)
            frame_width: Width to resize frames to
            frame_height: Height to resize frames to
            grayscale: Whether to convert to grayscale
            normalize: Whether to normalize pixel values to [0, 1]
        """
        super().__init__(env)
        
        self.frame_skip = frame_skip
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.grayscale = grayscale
        self.normalize = normalize
        
        # Update observation space
        num_channels = 1 if grayscale else 3
        self.observation_space = Box(
            low=0,
            high=255 if not normalize else 1.0,
            shape=(frame_height, frame_width, num_channels),
            dtype=np.uint8 if not normalize else np.float32
        )
    
    def reset(self, **kwargs):
        """Reset environment and return preprocessed observation."""
        obs, info = self.env.reset(**kwargs)
        return self._preprocess_frame(obs), info
    
    def step(self, action):
        """
        Take action with frame skipping and return preprocessed observation.
        
        For frame skipping, we take the max over the last two frames to handle
        flickering in some Atari games.
        """
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        
        # Frame buffer for max pooling
        frame_buffer = []
        
        # Repeat action for frame_skip frames
        for i in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            frame_buffer.append(obs)
            
            if terminated or truncated:
                break
        
        # Take max over last two frames to handle flickering
        if len(frame_buffer) >= 2:
            max_frame = np.maximum(frame_buffer[-2], frame_buffer[-1])
        else:
            max_frame = frame_buffer[-1]
        
        preprocessed_obs = self._preprocess_frame(max_frame)
        
        return preprocessed_obs, total_reward, terminated, truncated, info
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess a single frame.
        
        Args:
            frame: Raw frame from environment
            
        Returns:
            Preprocessed frame
        """
        # Convert to grayscale if needed
        if self.grayscale and len(frame.shape) == 3:
            # Use luminosity method: 0.299*R + 0.587*G + 0.114*B
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Resize frame
        frame = cv2.resize(frame, (self.frame_width, self.frame_height), 
                          interpolation=cv2.INTER_AREA)
        
        # Add channel dimension if grayscale
        if self.grayscale:
            frame = np.expand_dims(frame, axis=-1)
        
        # Normalize if requested
        if self.normalize:
            frame = frame.astype(np.float32) / 255.0
        
        return frame


class FrameStack(gym.Wrapper):
    """
    Stack k consecutive frames to give the agent temporal information.
    
    This helps the agent understand motion and velocity in the game.
    For example, with 4 stacked frames, the agent can see the last 4 positions
    of a moving object.
    """
    
    def __init__(self, env: gym.Env, k: int = 4):
        """
        Initialize frame stacking wrapper.
        
        Args:
            env: Gymnasium environment
            k: Number of frames to stack
        """
        super().__init__(env)
        
        self.k = k
        self.frames = deque([], maxlen=k)
        
        # Update observation space
        low = np.repeat(env.observation_space.low, k, axis=-1)
        high = np.repeat(env.observation_space.high, k, axis=-1)
        
        self.observation_space = Box(
            low=low,
            high=high,
            dtype=env.observation_space.dtype
        )
    
    def reset(self, **kwargs):
        """Reset environment and initialize frame stack."""
        obs, info = self.env.reset(**kwargs)
        
        # Fill frame stack with initial observation
        for _ in range(self.k):
            self.frames.append(obs)
        
        return self._get_observation(), info
    
    def step(self, action):
        """Take step and update frame stack."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_observation(), reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current stacked observation.
        
        Returns:
            Stacked frames concatenated along channel dimension
        """
        # Stack frames along the channel dimension
        return np.concatenate(list(self.frames), axis=-1)


class LazyFrames:
    """
    Lazy frame storage to save memory in replay buffer.
    
    Stores frames as references and only creates the array when needed.
    This significantly reduces memory usage when storing many transitions.
    """
    
    def __init__(self, frames: list):
        """
        Initialize lazy frames.
        
        Args:
            frames: List of frames to store
        """
        self._frames = list(frames)
        self._out = None
    
    def __array__(self, dtype=None):
        """Convert to numpy array when needed."""
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            if dtype is not None:
                self._out = self._out.astype(dtype)
        return self._out
    
    def __len__(self):
        """Return number of frames."""
        return len(self._frames)


def make_atari_env(env_name: str, frame_skip: int = 4, frame_stack: int = 4,
                   frame_width: int = 84, frame_height: int = 84,
                   seed: int = None, render_mode: str = None) -> gym.Env:
    """
    Create and preprocess Atari environment.
    
    Args:
        env_name: Name of Atari environment (e.g., "ALE/Jamesbond-v5")
        frame_skip: Number of frames to skip
        frame_stack: Number of frames to stack
        frame_width: Width to resize frames
        frame_height: Height to resize frames
        seed: Random seed
        render_mode: Render mode (None, "human", or "rgb_array")
    
    Returns:
        Preprocessed Gymnasium environment
    """
    # Create base environment
    env = gym.make(env_name, render_mode=render_mode)
    
    # Set seed if provided
    if seed is not None:
        env.reset(seed=seed)
    
    # Apply preprocessing
    env = AtariPreprocessing(
        env,
        frame_skip=frame_skip,
        frame_width=frame_width,
        frame_height=frame_height,
        grayscale=True,
        normalize=True
    )
    
    # Apply frame stacking
    env = FrameStack(env, k=frame_stack)
    
    return env


def get_state_shape(env: gym.Env) -> Tuple[int, int, int]:
    """
    Get the shape of the preprocessed state.
    
    Args:
        env: Preprocessed environment
        
    Returns:
        State shape as (channels, height, width) for PyTorch
    """
    obs_shape = env.observation_space.shape
    # Convert from (height, width, channels) to (channels, height, width)
    return (obs_shape[2], obs_shape[0], obs_shape[1])
