# Deep Learning Assignment 3: Forward Propagation and Backpropagation

## Problem Statement
Implement forward propagation and backpropagation using TensorFlow/Keras on the Diabetes dataset and study the effect of learning rates and epochs on model performance.

## Objectives
- Build a neural network using TensorFlow/Keras.
- Experiment with different learning rates and numbers of epochs.
- Evaluate model performance using Mean Squared Error (MSE) and Mean Absolute Error (MAE).
- Visualize training and validation loss.

## Technologies Used
- Python
- TensorFlow/Keras
- Scikit-learn
- Pandas
- Matplotlib

## Methodology
1. Load the Diabetes dataset.
2. Split the dataset into training and testing sets.
3. Create a Sequential neural network with two hidden layers using ReLU activation and a linear output layer.
4. Experiment with different learning rates and epoch counts.
5. Plot MSE, training loss, validation loss, and MAE.
6. Evaluate the final model on the test dataset.

## How to Run
1. Install the required libraries:
   ```bash
   pip install tensorflow scikit-learn pandas matplotlib
   ```
2. Open `Assignment3_DL.ipynb` in Jupyter Notebook.
3. Run all cells sequentially.

## Conclusion
This assignment demonstrates neural network training using TensorFlow/Keras and analyzes how learning rates and training epochs affect regression performance.
