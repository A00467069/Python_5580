import os
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers, models
from PIL import Image, ImageOps
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt 
from datetime import datetime, timedelta
import streamlit as st

def train_model():
    save_path = '../Models/mnist_model.keras'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    model = models.Sequential([
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.Flatten(),
                layers.Dense(64, activation='relu'),
                layers.Dense(10, activation='softmax')
            ])

    model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10, validation_split=0.2)
    model.save(save_path)

def remove_transparency(image):
    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        return background
    else:
        return image

def predict_digit(image, model):
    image = remove_transparency(image)
    image = image.convert('L')
    image = ImageOps.invert(image)
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    predictions = model.predict(image_array)
    predicted_digit = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)
    return predicted_digit, confidence

def coin_list():
    url = "https://api.coingecko.com/api/v3/coins/list"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data)
        return df
    else:
        return pd.DataFrame()

def historical_data(coin_id, timerange):
    api_key = "CG-Zqb3348miS6FQKP8dNpBWkSH"
    end_date = datetime.now()
    start_date = end_date - timedelta(days = timerange)

    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {
        'vs_currency': 'usd',
        'days': timerange,
        'interval': 'daily',
        'x_cg_demo_api_key': api_key  
    }
    
    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        prices = data['prices']
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('date', inplace=True, drop=True)
        df = df[df.index >= start_date]
        return df
    else:
        return pd.DataFrame()


def plot_historical_data(df):
    if not df.empty:
        # Plotting
        plt.figure(figsize=(10, 5))
        plt.plot(df.index, df['price'], label='Price')
        plt.title('Price over the last year')
        plt.xlabel('Date')
        plt.ylabel('USD')
        plt.legend()
        st.pyplot(plt.gcf())

        max_price = df['price'].max()
        min_price = df['price'].min()
        max_date = df['price'].idxmax()
        min_date = df['price'].idxmin()

        st.write(f"Maximum Price is ${max_price:.4f} on {max_date.strftime('%Y-%m-%d')}")
        st.write(f"Minimum Price is ${min_price:.4f} on {min_date.strftime('%Y-%m-%d')}")
    else:
        st.error("No data available to plot.")

def plot_historical_data_comp(df1, df2, coin_name1, coin_name2):
    if not df1.empty or not df2.empty:
        plt.figure(figsize=(10, 5))
        plt.plot(df1.index, df1['price'], label = coin_name1)
        plt.plot(df2.index, df2['price'], label = coin_name2)
        plt.title('Price Comparison over the Last Year')
        plt.xlabel('Date')
        plt.ylabel('USD')
        plt.legend()
        st.pyplot(plt.gcf())

        if df1.empty & df2.empty  :
            st.error(f"No data available to plot in {coin_name1} and {coin_name2}")
        elif df1.empty :
            st.error(f"No data available to plot in {coin_name1}")
        elif df2.empty: 
            st.error(f"No data available to plot in {coin_name2}")


    else:
        st.error("No data available to plot.")
