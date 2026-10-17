# The Model That Doesn't Know

## Overview

Deep learning models can make confident predictions even when they encounter images from classes they were never trained to recognize.

This project explores uncertainty estimation and Out-of-Distribution (OOD) detection using a ResNet18 image classifier trained on CIFAR-10.

The goal is to investigate whether a model can distinguish familiar images from unfamiliar ones instead of blindly making predictions.

## Problem Statement

Traditional image classifiers always return a prediction, even when an input belongs to an unknown class.

For example, a classifier trained to recognize airplanes, automobiles, birds, cats, and deer might confidently classify a dog as a cat.

This project investigates techniques that can help identify such unfamiliar inputs.

## Dataset

The project uses the CIFAR-10 dataset.

### Known Classes
- Airplane
- Automobile
- Bird
- Cat
- Deer

### Out-of-Distribution Classes
- Dog
- Frog
- Horse
- Ship
- Truck

The model is trained using the five known classes. The remaining five classes are used to evaluate its ability to identify unfamiliar images.

## Model Architecture

- Model: ResNet18
- Framework: PyTorch
- Pretrained weights: ImageNet
- Loss function: Cross-Entropy Loss
- Optimizer: Adam
- Learning rate: 0.0001
- Training epochs: 5

## OOD Detection Methods

### 1. Maximum Softmax Probability (MSP)

Uses the highest softmax probability as a confidence score. Lower confidence can indicate an unfamiliar input.

### 2. Energy-Based Detection

Uses energy scores derived from the model's logits to investigate whether an input differs from the training distribution.

### 3. Mahalanobis Distance

Measures how far an input's feature representation is from the known-class feature distributions.

## Evaluation Metrics

The project investigates:

- Classification accuracy
- OOD AUROC
- Known-class coverage
- OOD false acceptance rate
- Selective accuracy

## Experimental Results

### Classification Performance

| Metric | Result |
|---|---:|
| Known-class test accuracy | 94.94% |
| Selected confidence threshold | 0.50 |
| Known-class coverage | 99.54% |
| OOD false acceptance rate | 97.06% |
| Selective accuracy | 95.26% |

### OOD Detection Comparison

| Detection Method | OOD AUROC |
|---|---:|
| Maximum Softmax Probability (MSP) | 0.7698 |
| Energy-Based Detection | 0.7695 |
| Mahalanobis Distance | 0.7692 |

MSP achieved the highest AUROC among the three methods, although the differences were very small. All three methods showed moderate ability to distinguish known images from out-of-distribution (OOD) images.

**Evaluation note:** The Mahalanobis result is exploratory because its feature statistics were estimated using the known test set. A rigorous evaluation should estimate these statistics using training data and evaluate on held-out test data.

### Key Observation

The selected threshold of 0.50 achieved high known-class coverage but rejected very few OOD inputs. The OOD false acceptance rate was 97.06%, indicating that threshold-based rejection needs further improvement.

## Repository Contents

- `README.md` — Project documentation
- `your_notebook.ipynb` — Model training and evaluation notebook
- `results.json` — Recorded evaluation metrics

The trained model weights are not included because the file exceeds GitHub's regular upload limit.

## Limitations

- The model can assign high confidence to unfamiliar images.
- The selected confidence threshold rejects very few OOD inputs.
- MSP and Energy produced very similar AUROC values in this experiment.
- The Mahalanobis evaluation requires a stricter train/test separation.
- Results may vary with different datasets, training settings, and OOD classes.

## Future Improvements

- Experiment with temperature scaling.
- Tune the detection threshold using validation data.
- Recalculate and compare all OOD AUROC scores using the final model.
- Evaluate additional OOD detection methods.
- Test the model on more diverse and unfamiliar datasets.

## Technologies Used

Python, PyTorch, Torchvision, NumPy, Scikit-learn, Matplotlib, and Google Colab.

## Conclusion

This project explores an important limitation of deep learning: a model can be accurate on familiar data while remaining unreliable on unfamiliar inputs.

By comparing confidence-based and feature-based detection methods, the project investigates how uncertainty estimation can help build more reliable image classification systems.
