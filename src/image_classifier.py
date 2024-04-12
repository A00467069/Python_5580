import streamlit as st
from PIL import Image
import numpy as np 
import tensorflow as tf
from utils import predict_digit

def image_classifier():
    model_path = '../Models/mnist_model.keras'
    model = tf.keras.models.load_model(model_path)

    st.title('Image Classifier')
    uploaded_file = st.file_uploader("Upload an image with number", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        submit_button = st.button(label="Classify Image")

        image = Image.open(uploaded_file)
        #st.image(image, caption='Image To Predict', use_column_width=True)
        
        if submit_button:
            st.write("Processing the image")
            predicted_digit, confidence = predict_digit(image, model)
            st.write(f'Predicted Value is: {predicted_digit}')
            st.write(f'Confidence Value is: {confidence:.2f}')

if __name__ == '__main__':
    image_classifier()




