#!/usr/bin/env python3
"""
Simple Video Recorder for Trained DQN Agent
Author: Deepak Kumar
Course: INFO7375 - Fall 2025

This script records gameplay videos from a trained model.
"""

import os
import sys
import argparse
import numpy as np
import gymnasium as gym

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from network import DQN
from dqn_agent import DQNAgent
from preprocessing import make_atari_env, get_state_shape
from utils import get_device


def record_gameplay(model_path: str, num_episodes: int = 5, 
                   output_dir: str = 'videos/gameplay'):
    """
    Record gameplay videos from a trained agent.
    
    Args:
        model_path: Path to trained model checkpoint
        num_episodes: Number of episodes to record
        output_dir: Directory to save videos
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get device
    device = get_device(prefer_mps=True)
    
    # Create environment with video recording
    env = make_atari_env(
        env_name="ALE/Jamesbond-v5",
        frame_skip=4,
        frame_stack=4,
        frame_width=84,
        frame_height=84,
        seed=42,
        render_mode='rgb_array'
    )
    
    # Wrap with video recorder
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=output_dir,
        name_prefix='jamesbond_gameplay',
        episode_trigger=lambda x: True  # Record all episodes
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
    print(f"Loading model from {model_path}...")
    agent.load(model_path)
    agent.online_network.eval()
    
    print(f"\nRecording {num_episodes} episodes to {output_dir}...")
    
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
        
        print(f"Episode {episode+1:2d} recorded | Reward: {episode_reward:8.2f} | Length: {episode_length:5d}")
    
    env.close()
    print(f"\n✅ Videos saved to {output_dir}/")
    print(f"   Look for files named 'jamesbond_gameplay-episode-*.mp4'")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Record gameplay videos')
    parser.add_argument(
        '--model',
        type=str,
        default='models/experiments/quick_test/best_model.pth',
        help='Path to trained model'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=5,
        help='Number of episodes to record'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='videos/gameplay',
        help='Output directory for videos'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"❌ Error: Model file not found: {args.model}")
        print("\nAvailable models:")
        for root, dirs, files in os.walk('models'):
            for file in files:
                if file.endswith('.pth'):
                    print(f"  {os.path.join(root, file)}")
        return
    
    record_gameplay(
        model_path=args.model,
        num_episodes=args.episodes,
        output_dir=args.output
    )


if __name__ == '__main__':
    main()
