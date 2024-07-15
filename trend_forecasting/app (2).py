import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import requests
from io import BytesIO
import os
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import matplotlib.pyplot as plt 
import io  # Import io for creating an in-memory buffer for the plot

# Download the VADER lexicon if not already present
nltk.download('vader_lexicon')

# Load the saved model
current_path = os.getcwd()
model_path = os.path.join(current_path, 'fashion_mnist_model.keras')
loaded_model = tf.keras.models.load_model(model_path)

# Define the prediction function
def predict_from_image_url(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content)).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img).reshape((1, 28, 28, 1)).astype('float32') / 255
    predictions = loaded_model.predict(img_array)
    class_index = np.argmax(predictions)
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    return class_names[class_index]

# Sentiment Analysis Function
sia = SentimentIntensityAnalyzer()
def analyze_sentiment(text):
    sentiment_scores = sia.polarity_scores(text)
    return sentiment_scores

# Trend Forecasting Data and Model
data = {
    'Date': pd.date_range(start='2023-01-01', periods=12, freq='M'),
    'Sales': [200, 220, 250, 210, 260, 270, 300, 320, 330, 340, 360, 380]
}
trend_df = pd.DataFrame(data)
trend_df.set_index('Date', inplace=True)

# Fit ARIMA model
model = ARIMA(trend_df['Sales'], order=(1, 1, 1))
model_fit = model.fit()
forecast = model_fit.forecast(steps=3)  # Get the forecast for the next 3 months

# Plot the results
def plot_forecast():
    plt.figure(figsize=(10, 6))
    plt.plot(trend_df.index, trend_df['Sales'], label='Historical Sales', color='blue')
    plt.plot(pd.date_range(start='2024-01-01', periods=3, freq='M'), forecast, label='Forecasted Sales', linestyle='--', color='red')
    plt.title('Fashion Sales Trends and Forecast')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid(True)
    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()  # Close the plt figure to avoid display issues
    return buf

# Streamlit app
def main():
    st.title('Fashion MNIST Classification and Trend Forecasting')
    
    # Fashion MNIST Prediction
    st.header("Fashion MNIST Prediction")
    image_url = st.text_input("Enter the image URL:")
    if image_url:
        predicted_class = predict_from_image_url(image_url)
        st.write(f"Predicted class: {predicted_class}")

    # Upload Image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('L')
        st.image(img, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        img = img.resize((28, 28))
        img_array = np.array(img).reshape((1, 28, 28, 1)).astype('float32') / 255
        predictions = loaded_model.predict(img_array)
        class_index = np.argmax(predictions)
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        st.write(f"Prediction: {class_names[class_index]}")
  
    # Display Forecast DataFrame
    st.header("Sales Trend Forecasting")
    st.write(trend_df)
    
    # Plot and display the forecast graph
    st.subheader("Sales Trend Forecast Plot")
    buf = plot_forecast()
    st.image(buf, caption="Sales Trend and Forecast", use_column_width=True)
    
    forecast_values = forecast.tolist()  # Convert forecast to list
    st.write("Forecasted Sales for the next 3 months:")
    for i, value in enumerate(forecast_values, 1):
        st.write(f"Month {i}: {value:.2f}")

    # Sentiment Analysis
    st.header("Sentiment Analysis")
    user_text = st.text_area("Enter text for sentiment analysis:")
    if user_text:
        sentiment = analyze_sentiment(user_text)
        st.write(f"Sentiment Scores: {sentiment}")

if __name__ == "__main__":
    main()
