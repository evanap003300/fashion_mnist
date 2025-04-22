# MNIST & Fashion-MNIST Neural Network Classifiers

This project contains two simple deep learning models built using **TensorFlow and Keras** for image classification tasks:

- **MNIST Handwritten Digit Classification**
- **Fashion-MNIST Clothing Item Classification**

Both models use a basic fully connected neural network architecture to achieve high accuracy on their respective datasets.

---

## Overview

### 🧠 MNIST Model:
A neural network trained to recognize handwritten digits (0–9) using the MNIST dataset.

- Input: 28x28 grayscale images.
- Architecture: 2 hidden layers (128 neurons each, ReLU) and an output layer (softmax).
- Optimizer: Adam.
- Loss Function: Sparse Categorical Crossentropy.
- Achieved ~95% accuracy on the test dataset.

---

### 👕 Fashion-MNIST Model:
A neural network trained to classify clothing items such as shirts, sneakers, and coats using the Fashion-MNIST dataset.

- Input: 28x28 grayscale images.
- Architecture: Identical to the MNIST model.
- Optimizer: Adam.
- Loss Function: Sparse Categorical Crossentropy.
- Achieved ~89-91% accuracy on the test dataset.

---

## Requirements

Before running the code, install the required libraries:

```bash
pip install tensorflow numpy matplotlib
