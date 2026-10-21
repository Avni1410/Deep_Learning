# Transfer Learning for Image Classification

## Overview

This project implements and compares **AlexNet, VGG16, ResNet50, and EfficientNetB0** for image classification using the **CIFAR-10 dataset**.

The implementation is done using **TensorFlow/Keras in Jupyter Notebook**, with GPU acceleration used when available.

## Models

* **AlexNet** – AlexNet-style CNN trained on CIFAR-10
* **VGG16** – ImageNet pretrained model
* **ResNet50** – ImageNet pretrained model
* **EfficientNetB0** – ImageNet pretrained model

> **Note:** TensorFlow/Keras does not provide official ImageNet-pretrained AlexNet weights through `tf.keras.applications`. Therefore, an AlexNet-style architecture is implemented and trained on CIFAR-10.

## Dataset

**CIFAR-10**

* 50,000 training images
* 10,000 test images
* 10 classes
* Image size: 32 × 32 × 3

Classes:

`airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`

The dataset is downloaded manually and loaded locally into the Jupyter Notebook.

## Technologies

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

## GPU

The notebook automatically detects an available GPU using:

```python
tf.config.list_physical_devices("GPU")
```

The project is designed for an **NVIDIA RTX 4050 Laptop GPU**.

Jupyter Notebook does not have a separate GPU selection button. TensorFlow uses the GPU automatically when the current Jupyter environment supports GPU acceleration.

## Project Structure

```text
CIFAR10_Transfer_Learning/
│
├── data/
│   └── cifar-10-batches-py/
│
├── models/
├── results/
│
└── notebooks/
    ├── AlexNet_CIFAR10.ipynb
    ├── VGG16_CIFAR10.ipynb
    ├── ResNet50_CIFAR10.ipynb
    └── EfficientNetB0_CIFAR10.ipynb
```

## Evaluation

The models are compared using:

* Accuracy
* Precision
* Recall
* F1-score
* Training time
* Parameter count
* Confusion matrix

The final results are stored in a comparison table.

## Execution

To avoid GPU memory issues, models are trained **sequentially** rather than simultaneously.

Initial training can use a smaller dataset subset and batch size. The full CIFAR-10 dataset can be used after verifying that the implementation runs correctly.

## Conclusion

The project demonstrates transfer learning and performance comparison across different CNN architectures for CIFAR-10 image classification.
