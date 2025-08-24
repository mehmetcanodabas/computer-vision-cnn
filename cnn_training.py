# Project: Convolutional Neural Networks for Visual Recognition
# Chair of Digital Signal Processing and Circuit Technology
# Computer Vision Laboratory
# Note: Source code is not shared due to institutional restrictions.
#       This file only contains architecture-level explanations.

import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision.datasets as dset
import torchvision.transforms as T

# PART 1: DATASET LOADING
# Explanation:
# CIFAR-10 dataset (50k train, 10k test, 10 classes).
# Training set split into 49k train and 1k validation.

# PART 2: DEVICE SELECTION
# Explanation:
# Force CPU execution and set default tensor dtype (float32).
# To use GPU instead, replace "cpu" with "cuda".

# PART 3: UTILITY
# Explanation:
# Helper to reshape a 4D tensor [N, C, H, W] into a 2D tensor [N, C*H*W]
# before feeding it to fully-connected layers.

# PART 4: MODEL DEFINITIONS
# Explanation:
# Define three model classes:
#  - TwoLayerFC: simple MLP baseline.
#  - ThreeLayerConvNet: small CNN with two conv layers.
#  - DeepCIFAR10CNN: deeper VGG-lite style CNN with 3 conv blocks.
# All models output raw class scores (logits); weights use Kaiming init.

# PART 5: HYPERPARAMETERS
# Explanation:
# Centralize all tunable settings (batch size, epochs, LR, weight decay, etc.).
# Adjust these for experiments without touching training/eval code.

# PART 6: ACCURACY CHECK
# Explanation:
# Put model in eval mode and disable gradients to compute top-1 accuracy
# over a given dataloader (validation or test).

# PART 7: TRAINING LOOP
# Explanation:
# Standard supervised training with CrossEntropy:
# forward -> loss -> zero_grad -> backward -> step.
# Prints running loss and validates after each epoch.

# PART 8: SEEDING
# Explanation:
# Fix random seeds for Python, NumPy, and PyTorch to improve run-to-run
# reproducibility of training results.

# PART 9: MAIN ENTRY POINT
# Explanation:
# Orchestrates the full pipeline: seeding, device setup, dataloaders,
# model selection, optimizer creation, training, and final test evaluation.

# PART 10: SCRIPT GUARD
# Explanation:
# Standard Python entry point to allow running this file as a script.
# Choose which model to run by passing use_cnn: "deep", "shallow", or False.
