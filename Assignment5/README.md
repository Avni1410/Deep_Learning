# RNN, LSTM and GRU for Sentiment Classification

## Aim

To implement and compare RNN, LSTM, and GRU models for classifying movie reviews as positive or negative and analyze their performance using appropriate evaluation metrics.

## Dataset

The **IMDb Movie Reviews dataset** contains 50,000 movie reviews labeled as either positive or negative. It contains two main columns: `review` and `sentiment`. The dataset is used for binary sentiment classification.

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colab / Jupyter Notebook

## Models Used

Three deep learning models were implemented:

1. Simple RNN
2. LSTM
3. GRU

All three models use the same basic architecture:

```text
Input
   ↓
Embedding Layer
   ↓
RNN / LSTM / GRU
   ↓
Dropout
   ↓
Dense Layer
   ↓
Output Layer
