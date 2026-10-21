# Deep Learning Assignment 4: LSTM-Based Weather Forecasting

## Problem Statement
Develop an LSTM-based deep learning model for time-series forecasting using weather data to predict temperature.

## Objectives
- Load and preprocess weather time-series data.
- Build an LSTM neural network for temperature prediction.
- Apply dropout regularization to reduce overfitting.
- Evaluate predictions using MAE and RMSE.
- Visualize actual and predicted temperatures.

## Technologies Used
- Python
- TensorFlow/Keras
- Pandas, NumPy
- Scikit-learn
- Matplotlib
- KaggleHub

## Dataset
Weather dataset downloaded from Kaggle using KaggleHub.

## Methodology
1. Load and clean the weather dataset.
2. Convert timestamps and select numerical weather features.
3. Split the data into training and testing sets in chronological order.
4. Normalize features using MinMaxScaler.
5. Create sequences using the previous 24 hours of weather data.
6. Train an LSTM model with dropout layers for 30 epochs.
7. Evaluate the model using MAE and RMSE.
8. Plot actual versus predicted temperatures and forecast the next temperature.

## How to Run
1. Install the required libraries:
   ```bash
   pip install tensorflow pandas numpy matplotlib scikit-learn kagglehub
   ```
2. Open `DL_Assignment4.ipynb` in Jupyter Notebook or Google Colab.
3. Run all cells sequentially. KaggleHub may require appropriate internet access.

## Conclusion
This assignment demonstrates the use of LSTM networks for time-series forecasting and evaluates their ability to predict temperature from historical weather data.
