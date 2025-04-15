# CIFAR-10 Image Classification Using CNN-KERAS


This repository contains code for image classification on the **CIFAR-10** dataset using a **Convolutional Neural Network (CNN)** model built with **TensorFlow** and **Keras**.

The model classifies images from the CIFAR-10 dataset into 10 classes, including categories such as **Dog**, **Cat**, **Bird**, etc.

## Table of Contents
- [Project Description](#project-description)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Usage](#usage)
- [Results](#results)

## Project Description

This project focuses on training a **CNN** to classify images from the **CIFAR-10** dataset. The dataset consists of 60,000 32x32 color images across 10 different classes. The goal is to train a CNN model to classify these images into one of the 10 categories.

## Installation

1. **Clone the repository:**
   - Clone the repository to your local machine.

2. **Install required dependencies:**
   - Ensure that you have Python 3.x installed on your system.

   The primary dependencies include TensorFlow (for building the model), Keras (for handling neural network layers), NumPy (for data manipulation), and Matplotlib (for visualizing the results).

## Dataset

This project uses the **CIFAR-10** dataset, which consists of 60,000 32x32 color images in 10 different classes. The dataset is split into:
- 50,000 training images
- 10,000 test images

Classes in CIFAR-10:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

The dataset can be easily loaded using Keras' built-in dataset loader.

## Model Architecture

The **CNN model** built in this project is composed of several layers:
1. **Convolutional Layers**: These layers apply filters to the input image to extract features.
2. **Pooling Layers**: These layers reduce the spatial dimensions, helping to decrease computation time and prevent overfitting.
3. **Fully Connected Layers**: These layers connect all neurons in the previous layer to each neuron in the next layer. They help in classification based on the learned features.
4. **Output Layer**: This layer classifies the images into one of the 10 CIFAR-10 categories.

The model is trained using the CIFAR-10 training dataset and evaluated on the test dataset.

## Usage

Once the repository is cloned and dependencies are installed, you can use the model to train and evaluate the CNN on the CIFAR-10 dataset. The process includes loading the dataset, building the CNN model, compiling the model, training it on the training data, and evaluating its performance on the test data.

After training, the model can be used to make predictions on new CIFAR-10 images.

## Results

The CNN model, after training, achieved a validation accuracy of approximately 70% on the CIFAR-10 test set. The performance can vary depending on the model architecture, training parameters, and the number of epochs used for training.



