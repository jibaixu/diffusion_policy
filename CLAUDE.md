# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup
```bash
# Create conda environment (recommended: use mamba for faster installation)
mamba env create -f conda_environment.yaml
conda activate robodiff

# For macOS development (limited support)
mamba env create -f conda_environment_macos.yaml

# For real robot experiments
mamba env create -f conda_environment_real.yaml
```

### Training
```bash
# Single seed training
python train.py --config-name=image_pusht_diffusion_policy_cnn.yaml training.seed=42 training.device=cuda:0

# Multi-seed training with Ray
export CUDA_VISIBLE_DEVICES=0,1,2
ray start --head --num-gpus=3
python ray_train_multirun.py --config-name=image_pusht_diffusion_policy_cnn.yaml --seeds=42,43,44
```

### Evaluation
```bash
# Evaluate pre-trained checkpoint
python eval.py --checkpoint data/epoch=0550-test_mean_score=0.969.ckpt --output_dir data/pusht_eval_output --device cuda:0

# ManiSkill evaluation
python eval_maniskill.py

# Real robot evaluation  
python eval_real_robot.py -i data/outputs/blah/checkpoints/latest.ckpt -o data/eval_pusht_real --robot_ip 192.168.0.204
```

### Data Collection
```bash
# Real robot demonstration collection
python demo_real_robot.py -o data/demo_pusht_real --robot_ip 192.168.0.204
```

## High-Level Architecture

### Design Philosophy
The codebase follows a modular design where implementing N tasks and M methods requires O(N+M) code instead of O(N*M). Tasks and methods are independent, communicating through unified interfaces.

### Core Components

#### Task Side
- **Dataset**: Adapts datasets to unified interface, handles normalization
- **EnvRunner**: Executes policies and produces evaluation metrics 
- **Env**: Optional gym-compatible environment wrapper

#### Policy Side  
- **Policy**: Implements inference and training for a specific method
- **Workspace**: Manages training/evaluation lifecycle, checkpointing
- **Config**: YAML files define all parameters for reproducible experiments

### Key Interfaces

#### Low-Dimensional Tasks
- **Input**: `obs` tensor of shape `(B, To, Do)` where To=observation horizon
- **Output**: `action` tensor of shape `(B, Ta, Da)` where Ta=action horizon
- **Dataset**: Returns samples with `obs` and `action` tensors

#### Image-Based Tasks  
- **Input**: Dictionary with image tensors `(B, To, H, W, 3)` and other modalities
- **Output**: `action` tensor of shape `(B, Ta, Da)`
- **Normalization**: Handled by `LinearNormalizer` on GPU within policy

### Directory Structure
- `diffusion_policy/config/`: Hydra configuration files for tasks and methods
- `diffusion_policy/dataset/`: Task-specific dataset implementations
- `diffusion_policy/policy/`: Method-specific policy implementations  
- `diffusion_policy/workspace/`: Training/evaluation workspaces
- `diffusion_policy/model/`: Neural network components (diffusion, vision, etc.)
- `diffusion_policy/env_runner/`: Environment evaluation runners
- `diffusion_policy/real_world/`: Real robot integration components

### Important Notes
- Uses Hydra for configuration management with `@hydra.main` decorator
- Checkpointing happens at Workspace level with automatic state saving
- Real robot code uses lock-free `SharedMemoryRingBuffer` for multi-process camera capture
- ManiSkill integration supports multi-robot evaluation scenarios
- Vectorized environments use modified `AsyncVectorEnv` for parallel evaluation

### Experiment Configuration
- Main config files: `config/train_<method>_<modality>_workspace.yaml`
- Task configs: `config/task/<task_name>.yaml` 
- Override parameters via command line: `task=<task_name> training.seed=42`
- Experiment outputs saved to timestamped directories under `data/outputs/`