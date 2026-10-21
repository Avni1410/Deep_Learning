# Deep Learning Assignment 2: Multilayer Perceptron (MLP)

## Problem Statement
Design and implement a Multilayer Perceptron (MLP) for classification using the Wine dataset and evaluate its performance.

## Objectives
- Load and preprocess the Wine dataset.
- Perform train-test splitting and feature scaling.
- Build and train an MLP classifier.
- Evaluate the model using accuracy, confusion matrix, and classification report.

## Technologies Used
- Python
- Scikit-learn
- NumPy
- Jupyter Notebook

## Methodology
1. Load the Wine dataset.
2. Split the data into training and testing sets.
3. Apply StandardScaler for feature scaling.
4. Train an MLPClassifier with two hidden layers using ReLU activation and the Adam optimizer.
5. Evaluate the model using classification metrics.

## How to Run
1. Install the required libraries:
   ```bash
   pip install scikit-learn numpy jupyter
   ```
2. Open `DL_Assignment2.ipynb` in Jupyter Notebook.
3. Run all cells to train and evaluate the model.

## Conclusion
The assignment demonstrates how an MLP neural network can be used for multiclass classification and evaluated using standard performance metrics.
