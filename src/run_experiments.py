#!/usr/bin/env python3
"""
Experiment Runner for DQN Hyperparameter Tuning
Author: Deepak Kumar
Course: INFO7375 - Fall 2025

This script helps run multiple experiments with different configurations.
"""

import os
import sys
import yaml
import argparse
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from trainer import DQNTrainer


# Experiment configurations
EXPERIMENTS = {
    # Baseline
    'baseline': {
        'network': {'gamma': 0.99, 'learning_rate': 0.00025},
        'exploration': {'strategy': 'epsilon_greedy'},
        'training': {'total_episodes': 2000}
    },
    
    # Gamma experiments
    'gamma_0.95': {
        'network': {'gamma': 0.95, 'learning_rate': 0.00025},
        'exploration': {'strategy': 'epsilon_greedy'},
        'training': {'total_episodes': 2000}
    },
    'gamma_0.999': {
        'network': {'gamma': 0.999, 'learning_rate': 0.00025},
        'exploration': {'strategy': 'epsilon_greedy'},
        'training': {'total_episodes': 2000}
    },
    
    # Learning rate experiments
    'lr_0.0001': {
        'network': {'gamma': 0.99, 'learning_rate': 0.0001},
        'exploration': {'strategy': 'epsilon_greedy'},
        'training': {'total_episodes': 2000}
    },
    'lr_0.0005': {
        'network': {'gamma': 0.99, 'learning_rate': 0.0005},
        'exploration': {'strategy': 'epsilon_greedy'},
        'training': {'total_episodes': 2000}
    },
    
    # Exploration strategy experiments
    'boltzmann': {
        'network': {'gamma': 0.99, 'learning_rate': 0.00025},
        'exploration': {'strategy': 'boltzmann'},
        'training': {'total_episodes': 2000}
    },
    'ucb': {
        'network': {'gamma': 0.99, 'learning_rate': 0.00025},
        'exploration': {'strategy': 'ucb'},
        'training': {'total_episodes': 2000}
    },
    
    # Epsilon decay experiments
    'linear_decay': {
        'network': {'gamma': 0.99, 'learning_rate': 0.00025},
        'exploration': {
            'strategy': 'epsilon_greedy',
            'epsilon_decay_type': 'linear'
        },
        'training': {'total_episodes': 2000}
    },
    
    # Quick test (for debugging)
    'quick_test': {
        'network': {'gamma': 0.99, 'learning_rate': 0.00025},
        'exploration': {'strategy': 'epsilon_greedy'},
        'training': {'total_episodes': 100}
    },
}


def create_experiment_config(base_config_path: str, experiment_name: str, 
                            output_path: str):
    """
    Create a new config file for an experiment.
    
    Args:
        base_config_path: Path to base configuration
        experiment_name: Name of experiment from EXPERIMENTS dict
        output_path: Path to save new config
    """
    # Load base config
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get experiment modifications
    if experiment_name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment_name}")
    
    modifications = EXPERIMENTS[experiment_name]
    
    # Apply modifications
    for section, params in modifications.items():
        if section in config:
            config[section].update(params)
    
    # Save modified config
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Created config for experiment '{experiment_name}' at {output_path}")
    return output_path


def run_experiment(experiment_name: str, base_config_path: str):
    """
    Run a single experiment.
    
    Args:
        experiment_name: Name of experiment
        base_config_path: Path to base configuration
    """
    print("\n" + "="*70)
    print(f"STARTING EXPERIMENT: {experiment_name}")
    print("="*70)
    
    # Create experiment config
    config_dir = os.path.join('config', 'experiments')
    config_path = os.path.join(config_dir, f'{experiment_name}.yaml')
    create_experiment_config(base_config_path, experiment_name, config_path)
    
    # Create trainer
    trainer = DQNTrainer(config_path=config_path)
    
    # Update checkpoint and results directories
    trainer.checkpoint_dir = os.path.join('models', 'experiments', experiment_name)
    os.makedirs(trainer.checkpoint_dir, exist_ok=True)
    
    # Run training
    start_time = datetime.now()
    trainer.train()
    end_time = datetime.now()
    
    duration = (end_time - start_time).total_seconds() / 3600
    print(f"\nExperiment '{experiment_name}' completed in {duration:.2f} hours")
    
    # Save results with experiment name
    from utils import save_metrics
    save_metrics(
        trainer.episode_rewards,
        trainer.episode_lengths,
        trainer.epsilons,
        trainer.losses,
        save_dir='results/metrics',
        config_name=experiment_name
    )


def run_all_experiments(base_config_path: str, skip_baseline: bool = False):
    """
    Run all experiments sequentially.
    
    Args:
        base_config_path: Path to base configuration
        skip_baseline: Whether to skip baseline experiment
    """
    experiments_to_run = list(EXPERIMENTS.keys())
    
    if skip_baseline and 'baseline' in experiments_to_run:
        experiments_to_run.remove('baseline')
    
    print(f"\nWill run {len(experiments_to_run)} experiments:")
    for name in experiments_to_run:
        print(f"  - {name}")
    
    input("\nPress Enter to continue or Ctrl+C to cancel...")
    
    for i, exp_name in enumerate(experiments_to_run, 1):
        print(f"\n{'='*70}")
        print(f"EXPERIMENT {i}/{len(experiments_to_run)}")
        print(f"{'='*70}")
        
        try:
            run_experiment(exp_name, base_config_path)
        except KeyboardInterrupt:
            print("\n\nExperiment interrupted by user!")
            break
        except Exception as e:
            print(f"\n\nError in experiment '{exp_name}': {e}")
            print("Continuing to next experiment...")
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*70)


def list_experiments():
    """List all available experiments."""
    print("\nAvailable Experiments:")
    print("="*70)
    for name, config in EXPERIMENTS.items():
        print(f"\n{name}:")
        for section, params in config.items():
            for param, value in params.items():
                print(f"  {section}.{param}: {value}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run DQN experiments with different configurations'
    )
    parser.add_argument(
        '--experiment',
        type=str,
        help='Name of specific experiment to run'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all experiments'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available experiments'
    )
    parser.add_argument(
        '--skip-baseline',
        action='store_true',
        help='Skip baseline experiment when running all'
    )
    parser.add_argument(
        '--base-config',
        type=str,
        default='config/hyperparameters.yaml',
        help='Path to base configuration file'
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_experiments()
    elif args.all:
        run_all_experiments(args.base_config, args.skip_baseline)
    elif args.experiment:
        run_experiment(args.experiment, args.base_config)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python src/run_experiments.py --list")
        print("  python src/run_experiments.py --experiment baseline")
        print("  python src/run_experiments.py --experiment gamma_0.95")
        print("  python src/run_experiments.py --all")


if __name__ == '__main__':
    main()
