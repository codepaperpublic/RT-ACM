# RTACM: 
Related Task-aware Curriculum Meta-learning

## Features

- Extend MAML (Model-Agnostic Meta-Learning) algorithm
- Related Task-aware: Autoencoder for task similarity computation
- Curriculum Strategy: difficulty predictor for task sampling
- Support for N-way K-shot learning
- Freezing layers and Fine-tuning capabilities for target tasks

## Requirements

- Python 3.7+
- PyTorch 1.7+
- NumPy

## Installation

1. Clone this repository:
  -git clone https://github.com/yourusername/RTACM.git
  -cd RTACM
2. Install the required packages:

## Usage
To run the RTACM:
You can modify the hyperparameters and model configurations in the `main.py` file.

## Project Structure

- `models.py`: Contains neural network model definitions (Autoencoder, DifficultyPredictor, BearingNet)
- `rtacm.py`: Implements the RTACM class with meta-learning algorithms
- `main.py`: Main script to run the RTACM system
  
