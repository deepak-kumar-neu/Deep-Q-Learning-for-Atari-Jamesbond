"""
Training Script for DQN Agent
Author: Deepak Kumar
Course: INFO7375 - Fall 2025

This script trains a DQN agent on the Atari Jamesbond environment.
"""

import os
import sys
import argparse
from datetime import datetime
from tqdm import tqdm
import numpy as np
import torch
import gymnasium as gym

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from network import DQN
from dqn_agent import DQNAgent
from replay_buffer import ReplayBuffer
from preprocessing import make_atari_env, get_state_shape
from utils import (
    load_config, set_random_seeds, get_device,
    plot_training_progress, save_metrics, print_episode_summary,
    print_training_summary
)


class DQNTrainer:
    """
    Trainer class for DQN agent.
    
    Handles the complete training loop including:
    - Environment interaction
    - Experience collection
    - Network training
    - Logging and checkpointing
    - Video recording
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize trainer.
        
        Args:
            config_path: Path to configuration YAML file
        """
        # Load configuration
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), '..', 'config', 'hyperparameters.yaml'
            )
        
        self.config = load_config(config_path)
        
        # Set random seeds
        if 'seed' in self.config:
            set_random_seeds(self.config['seed'])
        
        # Get device
        self.device = get_device(prefer_mps=True)
        if 'device' in self.config and self.config['device'] == 'cpu':
            self.device = 'cpu'
        
        # Create environment
        self.env = make_atari_env(
            env_name=self.config['environment']['name'],
            frame_skip=self.config['preprocessing']['frame_skip'],
            frame_stack=self.config['preprocessing']['frame_stack'],
            frame_width=self.config['preprocessing']['frame_width'],
            frame_height=self.config['preprocessing']['frame_height'],
            seed=self.config.get('seed'),
            render_mode=self.config['environment'].get('render_mode')
        )
        
        # Get state and action dimensions
        state_shape = get_state_shape(self.env)
        n_actions = self.env.action_space.n
        
        print(f"\nEnvironment: {self.config['environment']['name']}")
        print(f"State shape: {state_shape}")
        print(f"Number of actions: {n_actions}")
        
        # Create network
        network = DQN(state_shape, n_actions)
        
        # Create agent
        self.agent = DQNAgent(
            network=network,
            n_actions=n_actions,
            device=self.device,
            gamma=self.config['network']['gamma'],
            learning_rate=self.config['network']['learning_rate'],
            epsilon_start=self.config['exploration']['epsilon_start'],
            epsilon_end=self.config['exploration']['epsilon_end'],
            epsilon_decay_steps=self.config['exploration']['epsilon_decay_steps'],
            epsilon_decay_type=self.config['exploration'].get('epsilon_decay_type', 'exponential'),
            exploration_strategy=self.config['exploration'].get('strategy', 'epsilon_greedy')
        )
        
        # Create replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=self.config['training']['memory_size'],
            seed=self.config.get('seed')
        )
        
        # Training parameters
        self.total_episodes = self.config['training']['total_episodes']
        self.max_steps = self.config['training']['max_steps_per_episode']
        self.batch_size = self.config['training']['batch_size']
        self.learning_starts = self.config['training']['learning_starts']
        self.train_frequency = self.config['training']['train_frequency']
        self.target_update_frequency = self.config['training']['target_update_frequency']
        
        # Logging and checkpointing
        self.log_frequency = self.config['logging']['log_frequency']
        self.save_frequency = self.config['checkpointing']['save_frequency']
        self.checkpoint_dir = self.config['checkpointing']['checkpoint_dir']
        
        # Create directories
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config['logging']['plot_dir'], exist_ok=True)
        
        # Metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.epsilons = []
        self.losses = []
        self.best_reward = -float('inf')
        
        # Step counter
        self.total_steps = 0
    
    def train(self):
        """Run the main training loop."""
        print("\n" + "="*70)
        print("STARTING TRAINING")
        print("="*70)
        
        start_time = datetime.now()
        
        try:
            for episode in range(1, self.total_episodes + 1):
                episode_reward, episode_length = self.train_episode()
                
                # Store metrics
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                self.epsilons.append(self.agent.get_epsilon())
                
                # Logging
                if episode % self.log_frequency == 0:
                    avg_loss = np.mean(self.losses[-100:]) if self.losses else 0.0
                    print_episode_summary(
                        episode, episode_reward, episode_length,
                        self.agent.get_epsilon(), avg_loss
                    )
                
                # Save checkpoint
                if episode % self.save_frequency == 0:
                    self.save_checkpoint(episode)
                
                # Save best model
                if episode_reward > self.best_reward:
                    self.best_reward = episode_reward
                    if self.config['checkpointing'].get('keep_best', True):
                        self.agent.save(
                            os.path.join(self.checkpoint_dir, 'best_model.pth')
                        )
                
                # Plot progress
                if episode % 100 == 0:
                    plot_path = os.path.join(
                        self.config['logging']['plot_dir'],
                        f'training_progress_ep{episode}.png'
                    )
                    plot_training_progress(
                        self.episode_rewards,
                        self.episode_lengths,
                        self.epsilons,
                        save_path=plot_path
                    )
        
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user!")
        
        end_time = datetime.now()
        
        # Print summary
        print_training_summary(
            self.episode_rewards,
            self.episode_lengths,
            start_time,
            end_time
        )
        
        # Save final results
        self.save_results()
        
        # Close environment
        self.env.close()
    
    def train_episode(self) -> tuple:
        """
        Train for one episode.
        
        Returns:
            Tuple of (episode_reward, episode_length)
        """
        state, _ = self.env.reset()
        # Convert state from (H, W, C) to (C, H, W) for PyTorch
        state = np.transpose(state, (2, 0, 1))
        
        episode_reward = 0
        episode_length = 0
        
        for step in range(self.max_steps):
            # Select action
            epsilon = self.agent.get_epsilon()
            action = self.agent.select_action(state, epsilon)
            
            # Take action in environment
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Convert next_state
            next_state = np.transpose(next_state, (2, 0, 1))
            
            # Store transition in replay buffer
            self.replay_buffer.push(state, action, reward, next_state, done)
            
            # Update state
            state = next_state
            episode_reward += reward
            episode_length += 1
            self.total_steps += 1
            
            # Train agent
            if (self.total_steps >= self.learning_starts and
                self.total_steps % self.train_frequency == 0 and
                self.replay_buffer.is_ready(self.batch_size)):
                
                # Sample batch
                batch = self.replay_buffer.sample(self.batch_size)
                states, actions, rewards, next_states, dones = batch
                
                # States are already in (C, H, W) format from storage
                # No need to transpose again
                
                # Train
                loss = self.agent.learn(states, actions, rewards, next_states, dones)
                self.losses.append(loss)
            
            # Update target network
            if self.total_steps % self.target_update_frequency == 0:
                self.agent.update_target_network()
            
            # Increment agent step counter
            self.agent.increment_step()
            
            if done:
                break
        
        return episode_reward, episode_length
    
    def save_checkpoint(self, episode: int):
        """
        Save checkpoint.
        
        Args:
            episode: Current episode number
        """
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f'checkpoint_ep{episode}.pth'
        )
        self.agent.save(checkpoint_path)
    
    def save_results(self):
        """Save final results and plots."""
        # Save metrics
        save_metrics(
            self.episode_rewards,
            self.episode_lengths,
            self.epsilons,
            self.losses,
            save_dir='results/metrics',
            config_name='baseline'
        )
        
        # Save final plot
        plot_path = os.path.join(
            self.config['logging']['plot_dir'],
            'final_training_progress.png'
        )
        plot_training_progress(
            self.episode_rewards,
            self.episode_lengths,
            self.epsilons,
            save_path=plot_path
        )
        
        # Save final model
        self.agent.save(
            os.path.join(self.checkpoint_dir, 'final_model.pth')
        )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train DQN agent on Atari games')
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration YAML file'
    )
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = DQNTrainer(config_path=args.config)
    
    # Start training
    trainer.train()


if __name__ == '__main__':
    main()
