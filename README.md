# SILO Simulation Reproduction

Reproduction project for:

SILO: Simulation-in-the-Loop Sim-to-Real Transfer
for Multi-Stage Cable Routing

## Target

This project reproduces the simulation-side components:

- Franka robot
- Articulated rigid-body cable approximation
- U-shaped harness
- Localized cable-routing RL task
- PPO training
- Intermediate state reset dataset
- Multi-harness simulation demonstration

## Environment

- Ubuntu 22.04
- Python 3.10
- PyTorch with CUDA
- ManiSkill3
- PhysX GPU backend

## Project Structure

- `configs/`: experiment configuration files
- `silo/envs/`: simulation environments
- `silo/controllers/`: robot controllers
- `silo/primitives/`: motion primitives
- `silo/rl/`: PPO implementation
- `scripts/`: executable scripts
- `tests/`: unit tests
- `datasets/`: generated datasets
- `checkpoints/`: model checkpoints
- `logs/`: experiment logs
- `videos/`: rendered videos

## Main Commands

```bash
python scripts/smoke_test_maniskill.py
python scripts/test_cable_dynamics.py
python scripts/test_reward.py
python scripts/generate_intermediate_states.py
python scripts/train_ppo.py
python scripts/evaluate_policy.py
