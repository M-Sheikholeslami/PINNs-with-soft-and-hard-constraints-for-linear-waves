# Physics-informed neural networks with hard and soft boundary conditions for linear free surface waves

This repository contains the code and experiments from the paper:

> **"Physics-Informed Neural Networks for Free-Surface Potential Ocean Flow with Periodic and Nonlinear Boundary Conditions"**  
> [Accepted Paper: PHFPOF25-AR-06910]

## Overview

This work explores the application of Physics-Informed Neural Networks (PINNs) for modeling water wave flows under various boundary condition treatments, including soft and hard impositions and periodicity.

## Repository Structure

├── VanillaPINN/                 # Baseline vanilla PINN without hard and periodic boundary handling
├── SoftPeriodic/                # PINN with soft periodic boundary conditions
├── HardPeriodic/                # PINN with hard periodic boundary conditions
├── ExponentialHardKBBC/         # PINN implementation with exponential hard kinematic bottom boundary condition
├── PolynomialHardKBBC/          # PINN with polynomial hard kinematic bottom boundary condition
├── unknownOmega/                # PINN modeling with omega as an unknown (learnable) parameter


## Getting Started

### Dependencies

Install Python requirements via:

```bash
pip install -r requirements.txt
```

Minimal dependencies include:

- `tensorflow >= 2.x`  
- `numpy`  
- `matplotlib`  
- `seaborn`  

### Running an Experiment

Each folder has its own training script.

## Output

Each run saves:

- Trained model (`.h5`) 
- Predictions (`.npz`)  

## Post-Processing

Visualizations and error metrics are available via the `postprocessing.py` scripts.
## Citation

If you use this codebase in your work, please cite our paper.

## Contact

For questions, please contact the corresponding author from the paper: Mohammad Sheikholeslami, mohshe@chalmers.se).


