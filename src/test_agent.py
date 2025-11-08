"""
Testing Script for Trained DQN Agent
Author: Deepak Kumar
Course: INFO7375 - Fall 2025

This script tests a trained DQN agent and can record gameplay videos.
"""

import os
import sys
import argparse
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from network import DQN
from dqn_agent import DQNAgent
from preprocessing import make_atari_env, get_state_shape
from utils import get_device


def test_agent(model_path: str, num_episodes: int = 10, 
               render: bool = True, record_video: bool = False):
    """
    Test a trained agent.
    
    Args:
        model_path: Path to trained model checkpoint
        num_episodes: Number of episodes to test
        render: Whether to render the environment
        record_video: Whether to record gameplay videos
    """
    # Get device
    device = get_device(prefer_mps=True)
    
    # Create environment
    render_mode = "rgb_array" if (render or record_video) else None
    
    env = make_atari_env(
        env_name="ALE/Jamesbond-v5",
        frame_skip=4,
        frame_stack=4,
        frame_width=84,
        frame_height=84,
        seed=42,
        render_mode=render_mode
    )
    
    # Wrap with video recorder if needed
    if record_video:
        import gymnasium as gym
        env = gym.wrappers.RecordVideo(
            env,
            video_folder='videos/test',
            name_prefix='test',
            episode_trigger=lambda x: True
        )
    
    # Get state and action dimensions
    state_shape = get_state_shape(env)
    n_actions = env.action_space.n
    
    # Create network and agent
    network = DQN(state_shape, n_actions)
    agent = DQNAgent(
        network=network,
        n_actions=n_actions,
        device=device,
        gamma=0.99,
        learning_rate=0.00025
    )
    
    # Load trained model
    agent.load(model_path)
    agent.online_network.eval()
    
    print(f"\nTesting agent for {num_episodes} episodes...")
    print(f"Model: {model_path}\n")
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = np.transpose(state, (2, 0, 1))
        
        episode_reward = 0
        episode_length = 0
        
        done = False
        while not done:
            # Select greedy action (no exploration)
            action = agent.get_greedy_action(state)
            
            # Take action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            next_state = np.transpose(next_state, (2, 0, 1))
            
            state = next_state
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"Episode {episode+1:3d} | Reward: {episode_reward:8.2f} | Length: {episode_length:5d}")
    
    env.close()
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Episodes:        {num_episodes}")
    print(f"Mean Reward:     {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Max Reward:      {np.max(episode_rewards):.2f}")
    print(f"Min Reward:      {np.min(episode_rewards):.2f}")
    print(f"Mean Length:     {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print("="*60 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Test trained DQN agent')
    parser.add_argument(
        '--model',
        type=str,
        default='models/checkpoints/best_model.pth',
        help='Path to trained model'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=10,
        help='Number of test episodes'
    )
    parser.add_argument(
        '--no-render',
        action='store_true',
        help='Disable rendering'
    )
    parser.add_argument(
        '--record',
        action='store_true',
        help='Record gameplay videos'
    )
    
    args = parser.parse_args()
    
    test_agent(
        model_path=args.model,
        num_episodes=args.episodes,
        render=not args.no_render,
        record_video=args.record
    )


if __name__ == '__main__':
    main()
