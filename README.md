# Deep Q-Learning for Atari Jamesbond

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29+-red.svg)

*A comprehensive Deep Q-Learning implementation for mastering Atari games using modern reinforcement learning techniques*

[Features](#features) • [Installation](#installation) • [Quick Start](#quick-start) • [Results](#results) • [Documentation](#documentation)

</div>

---

## 🎯 Overview

This project implements a state-of-the-art Deep Q-Learning (DQN) agent capable of learning to play Atari Jamesbond directly from pixel observations. The implementation features Double DQN architecture, experience replay, and multiple exploration strategies, optimized for Apple Silicon (M1/M2) hardware.

### Key Highlights

- **Advanced Architecture**: Double DQN with dueling networks support
- **Multiple Exploration Strategies**: ε-greedy, Boltzmann (softmax), and UCB
- **Comprehensive Training System**: Automatic checkpointing, metrics tracking, and visualization
- **Experiment Framework**: Easy configuration-based hyperparameter experimentation
- **Production-Ready**: Modular design, extensive documentation, and professional codebase
- **Mac Optimized**: Native support for MPS (Metal Performance Shaders) acceleration

---

## 🚀 Features

### Core Implementation

- **Deep Q-Network (DQN)**:
  - Convolutional neural network for visual processing
  - Experience replay buffer for stable learning
  - Target network for reduced Q-value overestimation
  - Double DQN for improved action value estimation

- **Preprocessing Pipeline**:
  - Frame grayscaling and resizing (84×84)
  - Frame stacking (4 consecutive frames)
  - Normalization and efficient memory management

- **Training Infrastructure**:
  - Automatic checkpointing every 100 episodes
  - Real-time metrics logging (rewards, losses, epsilon)
  - TensorBoard integration
  - Live training visualization

### Experiment System

- **Flexible Configuration**: YAML-based experiment management
- **Hyperparameter Sweeps**: Easy testing of different parameters
- **Comparison Tools**: Automated result analysis and visualization
- **Reproducibility**: Seeded random number generation

### Analysis & Visualization

- **Training Curves**: Episode rewards, losses, and exploration rates
- **Performance Metrics**: Statistical summaries and comparisons
- **Gameplay Recording**: Video capture of agent behavior
- **Experiment Comparison**: Multi-experiment analysis tools

---

## 📦 Installation

### Prerequisites

- Python 3.12+
- macOS (for MPS support) or Linux/Windows (CPU/CUDA)
- 8GB+ RAM recommended
- ~5GB disk space for models and results

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd AtariGames
   ```

2. **Create virtual environment**
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Atari ROMs**
   ```bash
   python -m ale_py.roms.install
   ```

5. **Verify installation**
   ```bash
   python verify_setup.py
   ```

---

## 🎮 Quick Start

### Training a Baseline Agent

```bash
# Start training with default configuration
./run.sh baseline

# Monitor progress in real-time
tail -f logs/baseline_*.log

# Or use the monitoring script
./monitor.sh logs/baseline_*.log
```

### Running Experiments

```bash
# Single experiment
./run.sh gamma_0.95

# Batch experiments
./run_all_experiments.sh

# Check experiment status
./check_status.sh
```

### Recording Gameplay

```bash
# Record videos from trained model
python record_videos.py --model models/experiments/baseline/best_model.pth --num-episodes 5
```

### Analyzing Results

```bash
# Generate comparison plots
python src/compare_experiments.py

# Results will be saved to results/plots/
```

---

## 📊 Results

### Baseline Performance

Training on **2,000 episodes** (~35 minutes on M1 MacBook Pro):

| Metric | Value |
|--------|-------|
| Mean Reward | 14.88 ± 27.61 |
| Max Reward | 150.0 |
| Mean Episode Length | 158.9 steps |
| Total Episodes | 2,000 |
| Training Time | ~0.6 hours |

### Experiment Comparisons

| Experiment | Description | Mean Reward | Max Reward | Mean Length |
|------------|-------------|-------------|------------|-------------|
| **baseline** | γ=0.99, α=0.00025 | 14.88 ± 27.61 | 150.0 | 158.9 |
| **gamma_0.95** | Low discount factor | **60.17 ± 142.89** | **5500.0** ⭐ | 167.7 |
| **gamma_0.999** | High discount factor | 6.72 ± 19.71 | 150.0 | 157.5 |
| **lr_0.0001** | Low learning rate | 14.88 ± 27.61 | 150.0 | 158.9 |
| **lr_0.0005** | High learning rate | 11.88 ± 25.00 | 150.0 | 161.4 |

**Key Findings**:
- **gamma_0.95** achieved exceptional performance with **max reward of 5,500** (37× higher than baseline!)
- Lower discount factor (γ=0.95) encourages prioritizing immediate rewards, highly effective for this environment
- High discount factor (γ=0.999) showed reduced performance, suggesting short-term tactics more important than long-term planning
- Learning rate variations showed minimal impact compared to discount factor
- All experiments completed 2,000 episodes with consistent training time (~0.6 hours each)

*See [REPORT.md](REPORT.md) for detailed analysis and visualizations*

---

## 🏗️ Project Structure

```
Deep-Q-Learning-for-Atari-Jamesbond/
├── src/                          # Source code
│   ├── __init__.py               # Package initialization
│   ├── network.py                # CNN architecture
│   ├── dqn_agent.py              # DQN agent implementation
│   ├── replay_buffer.py          # Experience replay
│   ├── preprocessing.py          # Frame preprocessing
│   ├── trainer.py                # Training loop
│   ├── test_agent.py             # Agent evaluation
│   ├── utils.py                  # Utilities
│   ├── run_experiments.py        # Experiment runner
│   └── compare_experiments.py    # Analysis tools
│
├── config/                       # Configuration files
│   ├── hyperparameters.yaml      # Base configuration
│   └── experiments/              # Experiment-specific configs
│       ├── baseline.yaml
│       ├── boltzmann.yaml
│       ├── gamma_0.95.yaml
│       ├── gamma_0.999.yaml
│       ├── linear_decay.yaml
│       ├── lr_0.0001.yaml
│       ├── lr_0.0005.yaml
│       └── quick_test.yaml
│
├── models/                       # Model storage
│   └── best_checkpoints/         # Best performing models
│       ├── baseline_best.pth
│       ├── boltzmann_best.pth
│       ├── gamma_0.95_best.pth
│       ├── gamma_0.999_best.pth
│       ├── lr_0.0001_best.pth
│       └── lr_0.0005_best.pth
│
├── results/                      # Experiment results
│   ├── metrics/                  # Training metrics and summaries
│   └── plots/                    # Generated visualizations
│
├── videos/                       # Gameplay recordings
│   └── gameplay/                 # Agent gameplay videos
│
├── docs/                         # Documentation
│   ├── Final Report.pdf          # Complete project report
│   └── REPORT.md                 # Technical analysis
│
├── run.sh                        # Training launcher script
├── record_videos.py              # Video recording utility
├── verify_setup.py               # Setup verification
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
└── LICENSE                       # MIT License
```

---

## 🔧 Configuration

### Base Configuration

Edit `config/hyperparameters.yaml` to modify training parameters:

```yaml
training:
  total_episodes: 2000
  batch_size: 32
  learning_rate: 0.00025
  gamma: 0.99
  
exploration:
  strategy: "epsilon_greedy"
  epsilon_start: 1.0
  epsilon_end: 0.01
  epsilon_decay_steps: 100000
```

---

## 📈 Training Details

### Network Architecture

**Convolutional Layers**:
- Conv1: 4 → 32 filters (8×8 kernel, stride 4)
- Conv2: 32 → 64 filters (4×4 kernel, stride 2)
- Conv3: 64 → 64 filters (3×3 kernel, stride 1)

**Fully Connected Layers**:
- FC1: 3136 → 512 neurons
- FC2: 512 → 18 (action values)

**Total Parameters**: ~2 million

---

## 📚 Documentation

### Main Documents

- **[REPORT.md](REPORT.md)**: Comprehensive technical report with all experiments
- **[DOCUMENTATION.pdf](DOCUMENTATION.pdf)**: LaTeX-generated guide

### Interactive Analysis

Jupyter notebook available in `notebooks/JamesBond_DQN_Complete_Analysis.ipynb`

---

## 🛠️ Technical Stack

- **Deep Learning**: PyTorch 2.0+
- **RL Environment**: Gymnasium (OpenAI Gym)
- **Atari Emulator**: ALE-py
- **Visualization**: Matplotlib, Seaborn
- **Video**: MoviePy, OpenCV
- **Analysis**: NumPy, Pandas

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.


---

## 🙏 Acknowledgments

- DeepMind for the original DQN research
- OpenAI/Farama Foundation for Gymnasium
- PyTorch team for the excellent framework
- ALE developers for the Atari Learning Environment

---

## 📖 References

1. Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning." *Nature*, 518(7540), 529-533.
2. Van Hasselt, H., et al. (2016). "Deep reinforcement learning with double q-learning." *AAAI Conference on Artificial Intelligence*.
3. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction*. MIT press.

---

<div align="center">

**Built with ❤️ using PyTorch and Gymnasium**

[⬆ Back to Top](#deep-q-learning-for-atari-jamesbond)

</div>
