# Support Vector Machine (SVM) Project

This repository contains code and resources for implementing and experimenting with Support Vector Machines (SVM) using Python. The project is designed to explore the functionality and applications of SVMs in machine learning.

## Project Structure

The project directory contains the following files:

- **`Untitled.ipynb`**: A Jupyter Notebook containing the main implementation and experimentation with SVMs.
- **`.ipynb_checkpoints/`**: A folder containing checkpoint files for the Jupyter Notebook.
- **`README.md`**: This file, providing an overview of the project.

## Requirements

To run the code in this project, you need the following:

- Python 3.10.5 or later
- Jupyter Notebook
- Required Python libraries: `numpy`, `pandas`, `matplotlib`, `scikit-learn`

You can install the required libraries using the following command:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## Usage

1. Open the Jupyter Notebook `Untitled.ipynb` in your preferred environment.
2. Follow the steps in the notebook to load data, preprocess it, and train an SVM model.
3. Experiment with different hyperparameters and datasets to observe the performance of the SVM.

## Features

- Implementation of SVM for classification tasks.
- Visualization of decision boundaries.
- Hyperparameter tuning for optimal performance.

## Code Sections Explained

### 1. SVM Basics and Visualization

The notebook begins with a basic implementation of Support Vector Machines (SVM) for simple datasets. It includes code to visualize decision boundaries and understand how SVM separates different classes. This section helps users grasp the fundamentals of SVMs before moving to more complex data.

### 2. Fashion MNIST Classification

This section demonstrates how to use SVMs for image classification using the Fashion MNIST dataset. The steps include:

- **Data Loading:** Downloading and loading the Fashion MNIST dataset using PyTorch’s `torchvision` utilities.
- **Preprocessing:** Transforming images to grayscale, normalizing, and converting them to tensors.
- **Data Preparation:** Flattening image data and preparing it for SVM input.
- **Training and Evaluation:** Training an SVM classifier on the Fashion MNIST data and evaluating its performance using accuracy and classification reports.

### 3. Persian License Plate Recognition (LPR) Digits

The notebook also includes a section for recognizing Persian digits (such as 3, 7, and S) from license plate images:

- **Data Loading:** Reading images from folders (e.g., `persian_LPR/3`, `persian_LPR/7`, `persian_LPR/S`).
- **Preprocessing:** Converting images to grayscale and resizing them to a standard shape (16x16).
- **Labeling:** Assigning numeric labels to each digit class for supervised learning.
- **Preparation for SVM:** Collecting all processed images and labels for training and testing an SVM classifier on Persian digit recognition.

---

Each section is designed to show how SVMs can be applied to both standard datasets (like Fashion MNIST) and custom datasets (like Persian license plate digits), providing a comprehensive overview of SVM usage in Python.

## Acknowledgments

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Python Official Documentation](https://www.python.org/doc/)

Feel free to contribute to this project by submitting issues or pull requests!
