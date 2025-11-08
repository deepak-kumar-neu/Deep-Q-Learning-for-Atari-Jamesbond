"""
Utility Functions for DQN Training
Author: Deepak Kumar
Course: INFO7375 - Fall 2025

This module provides utility functions for plotting, logging, and video recording.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import os
import json
from datetime import datetime


# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_training_progress(rewards: List[float], episode_lengths: List[int],
                           epsilons: List[float], save_path: str = None):
    """
    Plot training progress over episodes.
    
    Args:
        rewards: List of episode rewards
        episode_lengths: List of episode lengths
        epsilons: List of epsilon values
        save_path: Path to save plot (if None, displays plot)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Calculate moving averages
    window = 50
    rewards_ma = moving_average(rewards, window)
    lengths_ma = moving_average(episode_lengths, window)
    
    # Plot rewards
    axes[0, 0].plot(rewards, alpha=0.3, color='blue', label='Episode Reward')
    axes[0, 0].plot(rewards_ma, color='red', linewidth=2, label=f'{window}-Episode MA')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Training Rewards Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot episode lengths
    axes[0, 1].plot(episode_lengths, alpha=0.3, color='green', label='Episode Length')
    axes[0, 1].plot(lengths_ma, color='orange', linewidth=2, label=f'{window}-Episode MA')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].set_title('Episode Length Over Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot epsilon decay
    axes[1, 0].plot(epsilons, color='purple', linewidth=2)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Epsilon')
    axes[1, 0].set_title('Exploration Rate (Epsilon) Decay')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot cumulative rewards
    cumulative_rewards = np.cumsum(rewards)
    axes[1, 1].plot(cumulative_rewards, color='teal', linewidth=2)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Cumulative Reward')
    axes[1, 1].set_title('Cumulative Rewards Over Training')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_comparison(results_dict: Dict[str, Dict[str, List]], 
                   metric: str = 'rewards', save_path: str = None):
    """
    Plot comparison of different experimental configurations.
    
    Args:
        results_dict: Dictionary mapping config names to results
                     e.g., {'gamma=0.95': {'rewards': [...], 'lengths': [...]}}
        metric: Metric to plot ('rewards' or 'lengths')
        save_path: Path to save plot
    """
    plt.figure(figsize=(14, 7))
    
    window = 50
    
    for config_name, results in results_dict.items():
        data = results[metric]
        ma = moving_average(data, window)
        plt.plot(ma, linewidth=2, label=config_name, alpha=0.8)
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel(f'Average {metric.capitalize()} ({window} episodes)', fontsize=12)
    plt.title(f'Comparison of {metric.capitalize()} Across Configurations', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def moving_average(data: List[float], window: int) -> np.ndarray:
    """
    Calculate moving average of data.
    
    Args:
        data: List of values
        window: Window size for moving average
        
    Returns:
        Moving average as numpy array
    """
    if len(data) < window:
        return np.array(data)
    
    cumsum = np.cumsum(np.insert(data, 0, 0))
    ma = (cumsum[window:] - cumsum[:-window]) / window
    
    # Pad the beginning
    padding = data[:window-1]
    return np.concatenate([padding, ma])


def save_metrics(rewards: List[float], episode_lengths: List[int],
                epsilons: List[float], losses: List[float],
                save_dir: str, config_name: str = "baseline"):
    """
    Save training metrics to CSV file.
    
    Args:
        rewards: List of episode rewards
        episode_lengths: List of episode lengths
        epsilons: List of epsilon values
        losses: List of training losses
        save_dir: Directory to save metrics
        config_name: Name of configuration
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create metrics dictionary
    metrics = {
        'episode': list(range(len(rewards))),
        'reward': rewards,
        'episode_length': episode_lengths,
        'epsilon': epsilons,
    }
    
    # Add losses (may have different length)
    if losses:
        metrics['loss'] = losses[:len(rewards)]
    
    # Save as CSV
    import pandas as pd
    df = pd.DataFrame(metrics)
    filepath = os.path.join(save_dir, f'{config_name}_metrics.csv')
    df.to_csv(filepath, index=False)
    print(f"Metrics saved to {filepath}")
    
    # Save summary statistics
    summary = {
        'config_name': config_name,
        'total_episodes': len(rewards),
        'mean_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'max_reward': float(np.max(rewards)),
        'min_reward': float(np.min(rewards)),
        'mean_episode_length': float(np.mean(episode_lengths)),
        'final_epsilon': float(epsilons[-1]) if epsilons else 0.0,
    }
    
    summary_path = os.path.join(save_dir, f'{config_name}_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"Summary saved to {summary_path}")


def print_episode_summary(episode: int, reward: float, length: int,
                         epsilon: float, loss: float = None):
    """
    Print formatted episode summary.
    
    Args:
        episode: Episode number
        reward: Total reward
        length: Episode length
        epsilon: Current epsilon
        loss: Training loss (optional)
    """
    loss_str = f", Loss: {loss:.4f}" if loss is not None else ""
    print(f"Episode {episode:5d} | Reward: {reward:8.2f} | "
          f"Length: {length:5d} | Epsilon: {epsilon:.4f}{loss_str}")


def print_training_summary(rewards: List[float], episode_lengths: List[int],
                          start_time: datetime, end_time: datetime):
    """
    Print summary statistics after training.
    
    Args:
        rewards: List of all episode rewards
        episode_lengths: List of all episode lengths
        start_time: Training start time
        end_time: Training end time
    """
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"Total Episodes:        {len(rewards)}")
    print(f"Training Duration:     {duration/3600:.2f} hours")
    print(f"\nReward Statistics:")
    print(f"  Mean:                {np.mean(rewards):.2f}")
    print(f"  Std:                 {np.std(rewards):.2f}")
    print(f"  Min:                 {np.min(rewards):.2f}")
    print(f"  Max:                 {np.max(rewards):.2f}")
    print(f"  Last 100 episodes:   {np.mean(rewards[-100:]):.2f}")
    print(f"\nEpisode Length Statistics:")
    print(f"  Mean:                {np.mean(episode_lengths):.2f}")
    print(f"  Std:                 {np.std(episode_lengths):.2f}")
    print(f"  Min:                 {np.min(episode_lengths):.2f}")
    print(f"  Max:                 {np.max(episode_lengths):.2f}")
    print("="*70 + "\n")


def create_video_recorder(env, save_dir: str, episode: int):
    """
    Create video recorder for environment.
    
    Args:
        env: Gymnasium environment
        save_dir: Directory to save videos
        episode: Episode number
        
    Returns:
        Video recorder wrapper
    """
    import gymnasium as gym
    
    os.makedirs(save_dir, exist_ok=True)
    video_path = os.path.join(save_dir, f"episode_{episode}")
    
    # Wrap environment with video recorder
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=save_dir,
        name_prefix=f"episode_{episode}",
        episode_trigger=lambda x: True
    )
    
    return env


def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
    """
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def set_random_seeds(seed: int):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    import random
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    
    print(f"Random seeds set to {seed}")


def get_device(prefer_mps: bool = True) -> str:
    """
    Get available device for PyTorch.
    
    Args:
        prefer_mps: Whether to prefer MPS (for Mac M1/M2) over CPU
        
    Returns:
        Device string ("cuda", "mps", or "cpu")
    """
    import torch
    
    if torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA (GPU)")
    elif prefer_mps and torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS (Apple Silicon)")
    else:
        device = "cpu"
        print("Using CPU")
    
    return device
