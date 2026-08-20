# Diabetic Retinopathy Detection

Binary classification of diabetic retinopathy (DR-present vs No_DR) from retinal fundus images, using transfer learning with a pretrained ResNet18 in PyTorch. Includes Grad-CAM visualizations to interpret what the model focuses on when making predictions.

## Overview

Diabetic retinopathy is a leading cause of preventable blindness, and early detection from retinal fundus photographs can significantly improve patient outcomes. This project builds a binary classifier — distinguishing healthy retinas from retinas showing any signs of DR — using transfer learning on a pretrained ResNet18 backbone.

## Dataset

- **Source:** Diabetic Retinopathy 224x224 Gaussian Filtered dataset (Kaggle)
- **Classes:** 5-class severity labels, collapsed into binary for this task:

| Original Class | Images | Binary Label |
|---|---|---|
| No_DR | 1805 | 0 (No DR) |
| Mild | 370 | 1 (DR) |
| Moderate | 999 | 1 (DR) |
| Severe | 193 | 1 (DR) |
| Proliferate_DR | 295 | 1 (DR) |

- **Total:** 3662 images
- **Split:** 80/20 train/validation, stratified on the original 5-class label to ensure proportional representation of rare classes (e.g. Severe, Proliferate_DR) in both splits

## Approach

- **Model:** ResNet18 pretrained on ImageNet, with the final fully connected layer replaced for binary output
- **Training strategy:** convolutional backbone frozen; only the new classification head is trained (linear probing)
- **Loss / Optimizer:** CrossEntropyLoss, Adam (lr=0.001)
- **Preprocessing:** ImageNet normalization stats; training set augmented with random horizontal flip and ±10° rotation; validation set left unaugmented
- **Checkpointing:** best model (by validation accuracy) saved automatically during training

## Results

| Metric | No_DR | DR |
|---|---|---|
| Precision | 0.97 | 0.96 |
| Recall | 0.96 | 0.97 |
| F1-score | 0.96 | 0.96 |

**Overall accuracy: 96%** on the held-out validation set (733 images).

Of particular importance for a screening task: recall on the DR class was **97%**, meaning only 12 out of 372 actual DR cases were missed.

### Training curves
![Training curves](training_curves.png)

### Grad-CAM visualizations
Grad-CAM heatmaps were generated to visualize which regions of the retina the model attends to when predicting DR presence.

![Grad-CAM correct prediction](gradcam_correct.png)
![Grad-CAM misclassified example](gradcam_misclassified.png)

## Project structure

- explore_data.py — Full pipeline: data loading, training, evaluation, Grad-CAM
- requirements.txt
- training_curves.png
- gradcam_correct.png
- gradcam_misclassified.png
- README.md

## Setup

Install dependencies: pip install -r requirements.txt

Download the dataset from Kaggle and place it in a gaussian_filtered_images/ folder in the project root (or update DATA_DIR in the script to point to your local path).

## Running

Run the full pipeline: python explore_data.py

This runs everything end-to-end: data loading → training with checkpointing → evaluation → Grad-CAM generation.

## Future work

- Fine-tune deeper convolutional layers (e.g. layer4) with a low learning rate for potential further gains
- Extend to the full 5-class severity classification problem
- Build a simple inference function for single-image prediction