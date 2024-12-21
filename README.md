# dgm_2024_fall_diffvae_team8


# DDPM VAE Two-Stage Training

This repository provides a baseline implementation for training and inference using DDPM VAE with a two-stage process.

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Usage](#usage)
  - [Stage 1: Baseline Training](#stage-1-baseline-training)
  - [Stage 2 & 3: Two-Stage Training](#stage-2--3-two-stage-training)
  - [Inference](#inference)
- [Scripts Overview](#scripts-overview)

---

## Introduction

This project implements a two-stage training process for DDPM VAE, inspired by the original paper. The framework includes multiple classes in `main/models/diffusion/wrapper.py` for managing various stages of training and inference.

---

## Requirements

Ensure you have the following installed:
- Python == 3.7.16
- Pytorch == 1.11.0+cu102
- PyTorch_lighning == 1.4.9

## Usage

### Stage 1: Baseline Training
To train the baseline model, use the `DDPMWrapper` class in `main/models/diffusion/wrapper.py`. This implements the initial stage of training.

---

### Stage 2 & 3: Two-Stage Training
1. For **Stage 2 Training**, use the `DDPMWrapper_new2` class.
2. For **Stage 3 Training**, use the `DDPMWrapper_new3` class.

Execute the training scripts for stage 2 and stage 3 using:
```bash
bash scripts/train_ddpmvae.sh
```

### Inference
To perform inference, run:
```bash
bash scripts/test_ddpm.sh
```

## Scripts Overview
Training Script:
`bash scripts/train_ddpmvae.sh`: Executes training for stages 2 and 3.
Inference Script:
`bash scripts/test_ddpm.sh`: Runs the inference process.