# LSTM-Based Weather Forecasting

## Objective
Develop an LSTM-based deep learning model to forecast temperature using historical weather data.

## Dataset
The dataset contains hourly weather observations with features such as:
- Temperature
- Dew Point Temperature
- Relative Humidity
- Wind Speed
- Visibility
- Pressure

## Methodology
1. Load and preprocess the weather dataset.
2. Sort the data based on date and time.
3. Select relevant weather features.
4. Normalize the data using Min-Max Scaling.
5. Create sequences using the previous 24 hours.
6. Build and train an LSTM model.
7. Predict the next hour's temperature.
8. Evaluate the model using MAE and RMSE.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- TensorFlow/Keras
- LSTM

## Model Performance
- **MAE:** 0.97°C
- **RMSE:** 1.24°C

## Conclusion
The LSTM model successfully learned temporal patterns from the weather data and achieved reasonably accurate temperature predictions.
