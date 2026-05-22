🌱 PlantVillage Disease Classification using CNN

A Convolutional Neural Network (CNN) built in PyTorch to classify plant diseases from the PlantVillage dataset.

🌿 Overview

This project classifies images of plant leaves into healthy or diseased categories. It includes:

*Custom PyTorch dataset for loading images
*Data augmentation for better generalization
*CNN with multiple convolutional layers, pooling, batch normalization, and dropout
*Visualization of training and validation performance

🏗️ Model Architecture

CNN Layers:

3 Convolutional blocks with:
Conv2D → ReLU → BatchNorm → MaxPool → Dropout
Fully Connected Classifier:
Flatten → Linear(1024) → ReLU → BatchNorm → Dropout → Output Layer

Hyperparameters:

Epochs: 10 (demo)
Batch size: 32
Image size: 256×256
Loss: CrossEntropyLoss
Optimizer: Adam

📈 Training

The notebook trains the model on an 80/20 train/test split:

Data augmentation includes rotation, horizontal flip, random crop, and color jitter.
Tracks training and validation accuracy and loss over epochs.

Example output:

Epoch [10/10] Train Acc: 92.5%, Val Acc: 89.3%


📊 Results

The notebook generates plots for:

Training vs Validation Accuracy
Training vs Validation Loss

These help visualize model performance over epochs.


