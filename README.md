# ML Model CI/CD Pipeline

A simple CI/CD pipeline for a CNN model trained on MNIST dataset, with automated testing and deployment using GitHub Actions.

## Project Overview

This project demonstrates a basic CI/CD pipeline for machine learning models with:
- A simple CNN architecture for MNIST digit classification
- Automated model training
- Model validation and testing
- Automated deployment via GitHub Actions

## Model Architecture

The model is a Convolutional Neural Network (CNN) with:
- 2 convolutional layers
- 2 fully connected layers
- ReLU activation and max pooling
- Less than 25,000 parameters
- Designed for MNIST dataset (28x28 grayscale images)

## Requirements

- Python 3.8 or higher
- PyTorch (CPU version)
- torchvision
- pytest

## Local Setup

1. Clone the repository: 