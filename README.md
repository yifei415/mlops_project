Train_unet.py : This script trains a lightweight 3D UNet on NIfTI medical images by extracting normalized patches from a small subset of patients, using Dice loss and mixed-precision training, and saves model checkpoints after each epoch.

pipeline.py : This script performs inference on CT scans using multiple 3D segmentation models, saves predicted masks, and evaluates their performance with Dice, precision, and recall metrics.
