# Medical Image Segmentation Project

This repository provides scripts for training and evaluating 3D medical image segmentation models on NIfTI CT scans, focusing on lightweight and efficient pipelines using PyTorch and MONAI.

## Features

- **Train 3D UNet models** on patch-extracted datasets
- **Mixed-precision training** for efficiency
- **Multi-model inference** (UNet, ResUNet, DynUNet)
- **Automatic evaluation** using Dice, Precision, and Recall metrics
- **Save predicted masks and checkpoints** for downstream analysis

## Scripts Overview

### 1. `train_unet.py`
- Trains a lightweight 3D UNet on NIfTI medical images
- Extracts normalized patches from a small subset of patients
- Uses Dice loss and mixed-precision training
- Saves model checkpoints after each epoch

### 2. `pipeline.py`
- Performs inference on CT scans using multiple 3D segmentation models
- Saves predicted masks in the output directory
- Evaluates segmentation performance using Dice, Precision, and Recall metrics

## Folder Structure

