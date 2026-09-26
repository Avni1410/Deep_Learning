# BERT Text Classification 

## Overview

This project performs **multi-class text classification** on e-commerce product descriptions using a **pretrained BERT model**. The goal is to predict the product category from its textual description.

## Dataset

**Kaggle:** E-Commerce Text Classification
https://www.kaggle.com/datasets/saurabhshahane/ecommerce-text-classification

The dataset contains:

* `category` – target product category
* `text` – product description

## Technologies

* Python
* TensorFlow
* Keras 3
* KerasHub
* BERT
* Pandas
* NumPy
* Scikit-learn
* Matplotlib & Seaborn

## Model

The project uses **BERT Base English Uncased**:

```text
Product Text
     ↓
BERT Tokenizer / Preprocessor
     ↓
Pretrained BERT
     ↓
Classification Head
     ↓
Softmax
     ↓
Product Category
```

Maximum sequence length: **128 tokens**

## Workflow

1. Load and inspect the dataset
2. Perform EDA and data cleaning
3. Encode category labels
4. Split data into training and testing sets
5. Load pretrained BERT using KerasHub
6. Fine-tune BERT for classification
7. Evaluate using accuracy, precision, recall, F1-score and confusion matrix

