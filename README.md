# The Model That Doesn't Know

### Exploring Uncertainty Estimation and Out-of-Distribution Detection in Deep Learning

## Overview

Deep learning models can make confident predictions even when they encounter images from classes they were never trained to recognize.

For example, an image classifier trained to recognize airplanes, automobiles, birds, cats, and deer might confidently classify an unfamiliar dog image as a cat.

**The Model That Doesn't Know** explores this problem by investigating uncertainty estimation and Out-of-Distribution (OOD) detection using a ResNet18 image classifier trained on CIFAR-10.

The objective is to evaluate whether different scoring methods can help distinguish familiar images from unfamiliar inputs.

## Problem Statement

Traditional image classifiers generally assign an output class to every input, regardless of whether the input belongs to a known class.

This can lead to unreliable predictions when a model encounters unfamiliar data.

This project investigates three OOD detection techniques:

- Maximum Softmax Probability (MSP)
- Energy-Based Detection
- Mahalanobis Distance

The methods are evaluated to understand how effectively they distinguish in-distribution images from out-of-distribution images.

## Dataset

The project uses the CIFAR-10 dataset.

CIFAR-10 contains 60,000 color images across 10 classes.

### Known Classes

The model is trained to classify five classes:

- Airplane
- Automobile
- Bird
- Cat
- Deer

### Out-of-Distribution Classes

The remaining five classes are treated as unfamiliar inputs:

- Dog
- Frog
- Horse
- Ship
- Truck

This setup allows the project to evaluate both classification performance on known classes and OOD detection performance on held-out classes.

## Model Architecture

The project uses ResNet18, a convolutional neural network architecture commonly used for image classification.

### Configuration

| Component | Configuration |
|---|---|
| Model | ResNet18 |
| Framework | PyTorch |
| Initial pretrained weights | ImageNet |
| Dataset | CIFAR-10 |
| Known classes | 5 |
| OOD classes | 5 |
| Loss function | Cross-Entropy Loss |
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Training epochs | 5 |
| Image size | 224 × 224 |

The final classification layer is modified to produce predictions for the five known classes.

## OOD Detection Methods

### 1. Maximum Softmax Probability (MSP)

MSP uses the highest softmax probability produced by the classifier as a confidence score.

A lower maximum probability can indicate that the model is uncertain about an input.

In this project, inputs with confidence below a selected threshold can be flagged as potentially out-of-distribution.

### 2. Energy-Based Detection

Energy-Based Detection uses the model's logits to calculate an energy score.

The energy score provides an alternative way to measure how an input relates to the model's learned class distribution.

The implementation uses the negative log-sum-exp of the logits. With this convention, higher energy scores indicate potential OOD inputs.

### 3. Mahalanobis Distance

Mahalanobis Distance measures how far an input's feature representation lies from the distribution of known training examples.

The implementation extracts features from ResNet18's average-pooling layer and estimates the feature mean and covariance using known training features.

Larger distances indicate that an input may differ from the known-class feature distribution.

## Evaluation Metrics

The project uses the following metrics:

| Metric | Purpose |
|---|---|
| Classification Accuracy | Measures correct classification of known-class images |
| OOD AUROC | Measures how effectively a score separates known and unfamiliar inputs |
| Known-Class Coverage | Percentage of known images accepted at a confidence threshold |
| OOD False Acceptance Rate | Percentage of OOD images incorrectly accepted as known |
| OOD Rejection Rate | Percentage of OOD images rejected |
| Selective Accuracy | Classification accuracy among accepted known-class images |

An AUROC of 0.5 represents chance-level ranking, while a value closer to 1.0 indicates stronger separation between the two groups.

## Experimental Results

### 1. Known-Class Classification

| Metric | Result |
|---|---:|
| Known-class test accuracy | 94.94% |

The classifier achieved 94.94% accuracy on the known-class test set.

### 2. OOD Detection Comparison

The following AUROC values were obtained using the current evaluation pipeline.

| Detection Method | OOD AUROC |
|---|---:|
| Maximum Softmax Probability (MSP) | 0.7697 |
| Energy-Based Detection | 0.7695 |
| Mahalanobis Distance | 0.6829 |

MSP and Energy achieved similar AUROC values in this experiment. Mahalanobis Distance achieved a lower AUROC.

These results suggest that MSP and Energy provide moderate separation between known and unfamiliar images under the current experimental setup.

The small difference between MSP and Energy should not be interpreted as evidence that one method is consistently superior.

### 3. Confidence Threshold Evaluation

A confidence threshold was selected using validation data.

The selected threshold was then evaluated on the known-class and OOD test sets.

| Metric | Result |
|---|---:|
| Selected threshold | 0.99 |
| Known-class coverage | 82.62% |
| OOD false acceptance rate | 44.94% |
| OOD rejection rate | 55.06% |
| Selective accuracy | 99.15% |

### Interpretation

The results demonstrate the trade-off between accepting known images and rejecting unfamiliar inputs.

At a threshold of 0.99:

- 82.62% of known-class images were accepted.
- 44.94% of OOD images were incorrectly accepted.
- 55.06% of OOD images were rejected.
- Accepted known-class predictions achieved 99.15% selective accuracy.

The higher threshold improves the reliability of accepted known-class predictions but reduces the number of known images accepted.

However, the OOD false acceptance rate remains substantial. The model therefore cannot reliably reject all unfamiliar inputs.

## Project Workflow

The project follows these steps:

1. Load and preprocess the CIFAR-10 dataset.
2. Separate known classes from OOD classes.
3. Initialize a pretrained ResNet18 model.
4. Modify the classification layer for five known classes.
5. Train and evaluate the classifier.
6. Extract confidence scores and feature representations.
7. Calculate MSP, Energy, and Mahalanobis scores.
8. Evaluate OOD detection using AUROC.
9. Select a confidence threshold using validation data.
10. Evaluate known-class coverage and OOD rejection on held-out test data.

## Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Scikit-learn
- Matplotlib
- Google Colab
- GitHub

## Repository Contents

| File | Description |
|---|---|
| `README.md` | Project documentation |
| `The_Model_That_Doesnt.ipynb` | Model training, evaluation, and OOD detection implementation |
| `results.json` | Recorded evaluation metrics |



The trained model weights are not included because the file exceeds GitHub's regular upload limit. The notebook contains the implementation used to train and evaluate the model.

## Limitations

- The model can assign high confidence to unfamiliar images.
- The selected confidence threshold still allows many OOD images to be accepted.
- MSP and Energy produced very similar AUROC values in this experiment.
- OOD performance depends on the dataset, model, score distribution, and selected threshold.
- The Mahalanobis implementation relies on estimated feature statistics and covariance regularization.
- Additional experiments are needed to establish whether the observed differences between detection methods are statistically meaningful.

## Future Improvements

Potential improvements include:

- Investigating temperature scaling for confidence calibration.
- Evaluating additional threshold-selection strategies.
- Testing on more diverse OOD datasets.
- Exploring alternative OOD detection techniques.
- Investigating stronger feature representations.
- Comparing performance across different architectures.
- Evaluating detection performance across multiple random seeds.
- Adding a simple interface that allows users to upload images and view predictions and uncertainty scores.

## Conclusion

This project investigates a fundamental challenge in deep learning: a model can perform well on familiar data while remaining uncertain or overconfident when presented with unfamiliar inputs.

By comparing Maximum Softmax Probability, Energy-Based Detection, and Mahalanobis Distance, the project explores different approaches to identifying out-of-distribution images.

The experiments highlight the trade-off between classification coverage and OOD rejection and demonstrate why evaluating model confidence is important when building more reliable machine learning systems.
