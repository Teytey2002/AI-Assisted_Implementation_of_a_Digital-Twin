# AI-Assisted Digital Twin Calibration

## Overview
This repository presents a complete pipeline for **calibrating Digital Twins** using both **physics-based optimization methods** and **data-driven approaches (machine learning)**.

The project is part of a Master’s thesis and focuses on:
- Parameter estimation of dynamic systems
- Bridging simulation and real-world observations
- Scaling calibration methods to higher-dimensional problems

---

## Digital Twin Calibration Pipeline

The workflow follows a structured scientific approach:

1. **Dataset Generation**
   - Synthetic experiments generated via EcoSimPro or Python scripts
   - Parameter space sampling (R1, R2, C, etc.)

2. **Forward Simulation**
   - Physics-based simulator (RC circuit model)
   - Generates system outputs from parameters

3. **Calibration Methods**
   - Least Squares
   - Bayesian MAP
   - Genetic Algorithm (GA)
   - Particle Swarm Optimization (PSO)

4. **Validation**
   - Leave-One-Experiment-Out Cross-Validation (LOO-CV)
   - Generalization to unseen systems

5. **Evaluation**
   - RMSE, NMSE, MSE metrics

6. **Data-Driven Models**
   - CNN / MLP inverse models
   - Probabilistic neural networks
   - Hybrid physics + ML selection

---

## Project Structure

```
.
├── Digital_Twin_Calibration/   # Core pipeline (calibration + ML)
├── EcoSimPro/                 # Dataset generation scripts
├── Compare_Simulation/        # Simulation comparison tools
├── workflows/                 # CI/CD (GitHub Actions)
├── LP_Dataset_Deep_Learning_* # Generated datasets
├── README.md                  # This file
```

### Core Module (dtcalib)

```
src/dtcalib/
├── data.py         # Dataset loading
├── simulation.py   # Physics-based models
├── calibration.py  # Calibration algorithms
├── validation.py   # Cross-validation
├── metrics.py      # Evaluation metrics
└── deep_learning/  # Neural models + training
```

---

## Installation

### 1. Clone the repository
```bash
git clone <YOUR_REPO_URL>
cd AI-Assisted_Implementation_of_a_Digital-Twin
```

### 2. Create environments

Main environment:
```bash
conda env create -f environment.yml
conda activate DT_AI
```

Deep Learning (GPU):
```bash
conda env create -f env_deep_learning.yml
conda activate torch_gpu
```

### 3. Install package
```bash
pip install -e Digital_Twin_Calibration
```

---

## Usage

### Run calibration pipeline
```bash
python Digital_Twin_Calibration/run_calibration_cv.py
```

### Train neural model
```bash
python train.py --dataset <path> --split <split.json> --model prob_cnn
```

### Run inference
```bash
python inference.py --checkpoint <model.pth> --root-dir <data>
```

---

## Key Features

- Modular and extensible architecture
- Multiple calibration strategies
- Physics-based + ML hybrid approach
- Probabilistic uncertainty estimation
- Robust validation (LOO-CV)
- Fully tested with pytest

---

## Scientific Contribution

This project explores the limitations of inverse problems in Digital Twin calibration:

- Demonstrates identifiability issues when increasing parameters
- Compares optimization vs neural approaches
- Introduces probabilistic calibration strategies
- Evaluates hybrid physics + ML selection methods

---

## Results

- Accurate calibration for low-dimensional problems
- Performance degradation in multi-parameter settings (ambiguity)
- Neural models capture uncertainty effectively
- Hybrid approaches improve robustness

---

## Testing

Run all tests:
```bash
pytest -q
```

---

## Author

Thesis project in Computer Engineering  
Focus: Digital Twins, Machine Learning, Optimization

---

## License

MIT License
