#!/bin/bash
# Training Launcher Script
# Author: Deepak Kumar
# Course: INFO7375 - Fall 2025

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}           DQN Training Launcher for Atari Jamesbond${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

# Get experiment name from argument or use default
EXPERIMENT="${1:-baseline}"

# Read episode count from config file
if [ -f "config/experiments/${EXPERIMENT}.yaml" ]; then
    CONFIG_FILE="config/experiments/${EXPERIMENT}.yaml"
elif [ -f "config/hyperparameters.yaml" ]; then
    CONFIG_FILE="config/hyperparameters.yaml"
else
    CONFIG_FILE="config/hyperparameters.yaml"
fi

# Extract episodes from YAML (look for 'episodes:' or 'num_episodes:')
EPISODES=$(grep -E '^\s*(episodes|num_episodes):' "$CONFIG_FILE" | head -1 | sed 's/.*:\s*//')
if [ -z "$EPISODES" ]; then
    EPISODES="unknown"
fi

echo -e "${GREEN}Starting experiment: ${EXPERIMENT}${NC}"
echo -e "${YELLOW}Configuration:${NC}"

# Show configuration based on experiment
case $EXPERIMENT in
  baseline)
    echo "  - Episodes: $EPISODES"
    echo "  - Gamma: 0.99"
    echo "  - Learning Rate: 0.00025"
    echo "  - Exploration: ε-greedy"
    ;;
  gamma_0.95)
    echo "  - Episodes: $EPISODES"
    echo "  - Gamma: 0.95 (short-term focus)"
    echo "  - Learning Rate: 0.00025"
    ;;
  gamma_0.999)
    echo "  - Episodes: $EPISODES"
    echo "  - Gamma: 0.999 (long-term focus)"
    echo "  - Learning Rate: 0.00025"
    ;;
  lr_0.0001)
    echo "  - Episodes: $EPISODES"
    echo "  - Learning Rate: 0.0001 (slower)"
    echo "  - Gamma: 0.99"
    ;;
  lr_0.0005)
    echo "  - Episodes: $EPISODES"
    echo "  - Learning Rate: 0.0005 (faster)"
    echo "  - Gamma: 0.99"
    ;;
  boltzmann)
    echo "  - Episodes: $EPISODES"
    echo "  - Exploration: Boltzmann (softmax)"
    ;;
  ucb)
    echo "  - Episodes: $EPISODES"
    echo "  - Exploration: UCB (optimistic)"
    ;;
  linear_decay)
    echo "  - Episodes: $EPISODES"
    echo "  - Epsilon Decay: Linear"
    ;;
esac

echo ""
echo -e "${YELLOW}Estimated training time: 8-12 hours on Mac M1/M2${NC}"
echo ""

# Ask for confirmation
read -p "Start training? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Training cancelled."
    exit 1
fi

# Run training in background with logging
LOG_FILE="logs/${EXPERIMENT}_$(date +%Y%m%d_%H%M%S).log"

echo -e "${GREEN}Starting training in background...${NC}"
echo -e "${YELLOW}Log file: ${LOG_FILE}${NC}"
echo ""

# Use venv python explicitly
VENV_PYTHON="venv/bin/python"
nohup $VENV_PYTHON src/run_experiments.py --experiment $EXPERIMENT > "$LOG_FILE" 2>&1 &
PID=$!

echo -e "${GREEN}✓ Training started!${NC}"
echo -e "  Process ID: ${PID}"
echo -e "  Log file: ${LOG_FILE}"
echo ""
echo -e "${YELLOW}Monitor progress with:${NC}"
echo -e "  tail -f ${LOG_FILE}"
echo ""
echo -e "${YELLOW}Check if still running:${NC}"
echo -e "  ps aux | grep $PID"
echo ""
echo -e "${YELLOW}Stop training:${NC}"
echo -e "  kill $PID"
echo ""
echo -e "${GREEN}Training will complete in ~8-12 hours.${NC}"
echo -e "${GREEN}Results will be saved to models/experiments/${EXPERIMENT}/${NC}"
echo ""
# Quick wrapper to run experiments with venv activated
# Usage: ./run.sh [experiment_name]
#   e.g., ./run.sh quick_test
#   e.g., ./run.sh baseline

# Activate venv
source venv/bin/activate

# Run the experiment
if [ -z "$1" ]; then
    echo "Usage: ./run.sh [experiment_name]"
    echo "Available experiments:"
    python src/run_experiments.py --list
else
    python src/run_experiments.py --experiment "$1"
fi
