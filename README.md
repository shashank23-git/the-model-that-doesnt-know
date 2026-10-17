# The Model That Doesn't Know

## Overview

An uncertainty-aware image classification system that investigates whether a deep learning model can recognize when an input belongs to an unfamiliar class.

The project compares confidence-based and feature-based methods for out-of-distribution (OOD) detection.

## Objectives

- Classify images belonging to five known categories.
- Detect inputs from classes excluded during training.
- Investigate overconfident predictions.
- Compare multiple OOD detection techniques.
- Evaluate the trade-off between prediction accuracy and coverage.

## Dataset

CIFAR-10 is used for controlled experiments.

Known classes:
- Airplane
- Automobile
- Bird
- Cat
- Deer

OOD classes:
- Dog
- Frog
- Horse
- Ship
- Truck

The model is trained exclusively on the five known classes.

## Model

ResNet18 pretrained on ImageNet and fine-tuned for five-class classification.

Framework: PyTorch

## OOD Detection Methods

### 1. Maximum Softmax Probability (MSP)

Uses the maximum softmax probability as a confidence score.

### 2. Energy-Based Detection

Uses the energy score calculated from the model's logits.

### 3. Mahalanobis Distance

Measures how far extracted image features lie from the known-class feature distributions.

## Initial Experimental Results

| Method | OOD AUROC |
|---|---:|
| MSP | 0.7929 |
| Energy-Based Detection | 0.7885 |
| Mahalanobis Distance | 0.7066 |

These are preliminary results from the initial model and must not be interpreted as final retrained-model results.

## Key Observation

The classifier achieves high accuracy on known classes but can also assign high confidence to unfamiliar inputs.

This demonstrates why classification confidence alone is not sufficient to guarantee reliable predictions.

## Evaluation Metrics

- Classification accuracy
- OOD AUROC
- Known-class coverage
- OOD false acceptance rate
- Selective accuracy

## Limitations

- Evaluation uses CIFAR-10 classes held out from classifier training.
- Results may differ on external datasets and real-world images.
- High confidence does not guarantee that a prediction is correct.
- OOD detection performance depends on the model, scoring method, and threshold.

## Future Improvements

- Evaluate on external datasets.
- Investigate alternative uncertainty estimation methods.
- Improve threshold selection and calibration.
- Develop an interactive image-classification demo.

## Technologies

- Python
- PyTorch
- Torchvision
- NumPy
- Scikit-learn
- Matplotlib
