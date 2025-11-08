#!/usr/bin/env python3
"""
Compare Experiment Results
Generates comparison plots and analysis for all experiments
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Experiments to compare
EXPERIMENTS = [
    'baseline',
    'gamma_0.95',
    'gamma_0.999',
    'lr_0.0001',
    'lr_0.0005',
    'boltzmann',
    'linear_decay'
]

EXPERIMENT_LABELS = {
    'baseline': 'Baseline (γ=0.99, α=0.00025)',
    'gamma_0.95': 'Gamma=0.95 (Short-term)',
    'gamma_0.999': 'Gamma=0.999 (Long-term)',
    'lr_0.0001': 'LR=0.0001 (Slower)',
    'lr_0.0005': 'LR=0.0005 (Faster)',
    'boltzmann': 'Boltzmann Exploration',
    'linear_decay': 'Linear Epsilon Decay'
}

def load_experiment_data(experiment_name):
    """Load metrics for an experiment"""
    metrics_path = Path(f'results/metrics/{experiment_name}_metrics.csv')
    summary_path = Path(f'results/metrics/{experiment_name}_summary.json')
    
    if not metrics_path.exists():
        print(f"Warning: {metrics_path} not found")
        return None, None
    
    metrics = pd.read_csv(metrics_path)
    summary = None
    
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            summary = json.load(f)
    
    return metrics, summary

def plot_learning_curves():
    """Plot learning curves for all experiments"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Average Reward over Episodes
    ax1 = axes[0, 0]
    for exp in EXPERIMENTS:
        metrics, _ = load_experiment_data(exp)
        if metrics is not None:
            # Calculate rolling average
            window = 50
            rolling_reward = metrics['reward'].rolling(window=window, min_periods=1).mean()
            ax1.plot(metrics['episode'], rolling_reward, label=EXPERIMENT_LABELS.get(exp, exp), linewidth=2)
    
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Average Reward (50-ep window)', fontsize=12)
    ax1.set_title('Learning Curves: Reward Progression', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Episode Length over Episodes
    ax2 = axes[0, 1]
    for exp in EXPERIMENTS:
        metrics, _ = load_experiment_data(exp)
        if metrics is not None:
            window = 50
            rolling_length = metrics['episode_length'].rolling(window=window, min_periods=1).mean()
            ax2.plot(metrics['episode'], rolling_length, label=EXPERIMENT_LABELS.get(exp, exp), linewidth=2)
    
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Average Episode Length (50-ep window)', fontsize=12)
    ax2.set_title('Episode Length Progression', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Loss over Episodes
    ax3 = axes[1, 0]
    for exp in EXPERIMENTS:
        metrics, _ = load_experiment_data(exp)
        if metrics is not None and 'loss' in metrics.columns:
            # Filter out zero losses and calculate rolling average
            non_zero_loss = metrics[metrics['loss'] > 0].copy()
            if len(non_zero_loss) > 0:
                window = 50
                rolling_loss = non_zero_loss['loss'].rolling(window=window, min_periods=1).mean()
                ax3.plot(non_zero_loss['episode'], rolling_loss, label=EXPERIMENT_LABELS.get(exp, exp), linewidth=2)
    
    ax3.set_xlabel('Episode', fontsize=12)
    ax3.set_ylabel('Average Loss (50-ep window)', fontsize=12)
    ax3.set_title('Training Loss Progression', fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')  # Log scale for loss
    
    # Plot 4: Epsilon Decay
    ax4 = axes[1, 1]
    for exp in EXPERIMENTS:
        metrics, _ = load_experiment_data(exp)
        if metrics is not None and 'epsilon' in metrics.columns:
            ax4.plot(metrics['episode'], metrics['epsilon'], label=EXPERIMENT_LABELS.get(exp, exp), linewidth=2)
    
    ax4.set_xlabel('Episode', fontsize=12)
    ax4.set_ylabel('Epsilon', fontsize=12)
    ax4.set_title('Exploration Rate (Epsilon) Decay', fontsize=14, fontweight='bold')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path('results/plots/all_experiments_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison plot to {output_path}")
    plt.close()

def plot_final_performance():
    """Plot final performance comparison"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    experiments = []
    mean_rewards = []
    std_rewards = []
    mean_lengths = []
    final_losses = []
    
    for exp in EXPERIMENTS:
        metrics, summary = load_experiment_data(exp)
        if summary is not None:
            experiments.append(EXPERIMENT_LABELS.get(exp, exp))
            mean_rewards.append(summary.get('mean_reward', 0))
            std_rewards.append(summary.get('std_reward', 0))
            mean_lengths.append(summary.get('mean_episode_length', 0))
            
            # Get final 100 episodes loss
            if metrics is not None and 'loss' in metrics.columns:
                final_loss = metrics.tail(100)['loss'].mean()
                final_losses.append(final_loss)
            else:
                final_losses.append(0)
    
    # Plot 1: Mean Reward Comparison
    ax1 = axes[0]
    bars = ax1.bar(range(len(experiments)), mean_rewards, color=sns.color_palette("husl", len(experiments)))
    ax1.set_xticks(range(len(experiments)))
    ax1.set_xticklabels(experiments, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('Mean Reward', fontsize=12)
    ax1.set_title('Final Performance: Mean Reward', fontsize=14, fontweight='bold')
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, mean_rewards)):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.5, f'{val:.1f}', 
                ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Mean Episode Length Comparison
    ax2 = axes[1]
    bars = ax2.bar(range(len(experiments)), mean_lengths, color=sns.color_palette("husl", len(experiments)))
    ax2.set_xticks(range(len(experiments)))
    ax2.set_xticklabels(experiments, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Mean Episode Length', fontsize=12)
    ax2.set_title('Final Performance: Episode Length', fontsize=14, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3)
    
    for i, (bar, val) in enumerate(zip(bars, mean_lengths)):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 2, f'{val:.0f}', 
                ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Final Loss Comparison
    ax3 = axes[2]
    bars = ax3.bar(range(len(experiments)), final_losses, color=sns.color_palette("husl", len(experiments)))
    ax3.set_xticks(range(len(experiments)))
    ax3.set_xticklabels(experiments, rotation=45, ha='right', fontsize=9)
    ax3.set_ylabel('Mean Loss (Final 100 episodes)', fontsize=12)
    ax3.set_title('Final Performance: Training Loss', fontsize=14, fontweight='bold')
    ax3.grid(True, axis='y', alpha=0.3)
    ax3.set_yscale('log')
    
    for i, (bar, val) in enumerate(zip(bars, final_losses)):
        if val > 0:
            ax3.text(bar.get_x() + bar.get_width()/2, val * 1.1, f'{val:.0f}', 
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_path = Path('results/plots/final_performance_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved final performance plot to {output_path}")
    plt.close()

def generate_summary_table():
    """Generate a summary table of all experiments"""
    data = []
    
    for exp in EXPERIMENTS:
        metrics, summary = load_experiment_data(exp)
        if summary is not None:
            row = {
                'Experiment': EXPERIMENT_LABELS.get(exp, exp),
                'Total Episodes': summary.get('total_episodes', 'N/A'),
                'Mean Reward': f"{summary.get('mean_reward', 0):.2f}",
                'Std Reward': f"{summary.get('std_reward', 0):.2f}",
                'Max Reward': f"{summary.get('max_reward', 0):.2f}",
                'Last 100 Avg': f"{summary.get('last_100_avg_reward', 0):.2f}",
                'Mean Length': f"{summary.get('mean_episode_length', 0):.1f}",
                'Training Time (hrs)': f"{summary.get('training_duration_hours', 0):.2f}"
            }
            data.append(row)
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    output_path = Path('results/metrics/experiment_comparison_summary.csv')
    df.to_csv(output_path, index=False)
    print(f"✓ Saved summary table to {output_path}")
    
    # Print to console
    print("\n" + "="*100)
    print("EXPERIMENT COMPARISON SUMMARY")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100 + "\n")
    
    return df

def main():
    """Main function"""
    print("="*60)
    print("   Experiment Results Comparison & Analysis")
    print("="*60)
    print()
    
    # Create output directory if needed
    Path('results/plots').mkdir(parents=True, exist_ok=True)
    
    print("Loading experiment data...")
    available_experiments = []
    for exp in EXPERIMENTS:
        metrics, summary = load_experiment_data(exp)
        if metrics is not None:
            available_experiments.append(exp)
            print(f"  ✓ {exp}")
        else:
            print(f"  ✗ {exp} (not found)")
    
    if not available_experiments:
        print("\n❌ No experiment data found!")
        print("Please run experiments first using ./run_all_experiments.sh")
        return
    
    print(f"\nFound {len(available_experiments)} experiments\n")
    
    print("Generating visualizations...")
    plot_learning_curves()
    plot_final_performance()
    
    print("\nGenerating summary table...")
    generate_summary_table()
    
    print("\n" + "="*60)
    print("✓ Analysis complete!")
    print("="*60)
    print("\nGenerated files:")
    print("  - results/plots/all_experiments_comparison.png")
    print("  - results/plots/final_performance_comparison.png")
    print("  - results/metrics/experiment_comparison_summary.csv")
    print("\nNext steps:")
    print("  1. Review plots and summary")
    print("  2. Complete Jupyter notebook with findings")
    print("  3. Record gameplay videos")
    print("  4. Create demonstration video")
    print()

if __name__ == "__main__":
    main()
