# Fashion MNIST Image Classification using TensorFlow/Keras

## Objective
To install and configure TensorFlow/Keras and perform image data preprocessing, normalization, visualization, model training, evaluation, and model saving using the Fashion MNIST dataset.

---

## Dataset

**Dataset:** Fashion MNIST

The Fashion MNIST dataset is a collection of grayscale images of fashion products. It contains 70,000 images of size **28 × 28** pixels divided into:

- Training Images: 60,000
- Testing Images: 10,000
- Number of Classes: 10

### Classes

| Label | Class |
|-------|----------------|
| 0 | T-shirt/Top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle Boot |

---

## Technologies Used

- Python
- TensorFlow
- Keras
- Matplotlib

---

## Project Workflow

1. Import the required libraries.
2. Load the Fashion MNIST dataset.
3. Normalize image pixel values.
4. Display dataset information.
5. Visualize sample images.
6. Build a Sequential Neural Network.
7. Compile the model.
8. Train the model.
9. Evaluate model performance.
10. Save the trained model.

---

## Model Architecture

- Flatten Layer (28 × 28)
- Dense Layer (128 neurons, ReLU activation)
- Output Layer (10 neurons, Softmax activation)

---

## Preprocessing

- Dataset loaded using Keras API.
- Pixel values normalized from **0–255** to **0–1**.
- Built-in training and testing datasets used for model training and evaluation.

---

## Training Configuration

- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Metric: Accuracy
- Epochs: 3

---

## Output

The model predicts one of the following fashion categories:

- T-shirt/Top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle Boot

The trained model is saved as:

```
model.keras
```

---

## Expected Result

- Successfully configure TensorFlow/Keras.
- Perform image preprocessing and normalization.
- Visualize sample images.
- Train a neural network on the Fashion MNIST dataset.
- Evaluate the model on the test dataset.
- Save the trained model for future use.

---

## Conclusion

This assignment demonstrates the complete workflow of image classification using TensorFlow and Keras. It covers data loading, preprocessing, visualization, neural network creation, model training, evaluation, and model saving using the Fashion MNIST dataset.
