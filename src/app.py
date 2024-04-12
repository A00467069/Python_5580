import os
import streamlit as st
from stock_details import stock_details
from coin_comparison import coin_comparison
from image_classifier import image_classifier

from utils import train_model

model_path = '../Models/mnist_model.keras'
def ensure_model():
    st.write("Training Model")
    train_model()
    st.write("Model trained and saved successfully!")

if not os.path.exists(model_path):
    ensure_model()
    st.experimental_rerun()

st.sidebar.title('Navigation')
page = st.sidebar.radio('Go to', ['Stock Details', 'Coin Comparison', 'Image Classifier'])

if page == 'Stock Details':
    stock_details()
elif page == 'Coin Comparison':
    coin_comparison()
elif page == 'Image Classifier':
    image_classifier()
