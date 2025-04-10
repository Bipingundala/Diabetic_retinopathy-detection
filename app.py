import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import pandas as pd
from tensorflow.keras.models import load_model

# Load Trained Models (cached so they load only once)
@st.cache_resource
def load_models():
    model_cnn = load_model("cnn_model.h5")
    model_inception = load_model("inception_model.h5")
    model_vgg16 = load_model("vgg16_model.h5")
    return model_cnn, model_inception, model_vgg16

model_cnn, model_inception, model_vgg16 = load_models()

# Define class labels for diabetic retinopathy severity
class_labels = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

# Function to preprocess image for model input
def preprocess_image(image):
    # Convert BGR (OpenCV default) to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Resize image to 224x224 (adjust if needed for InceptionV3 input size)
    image = cv2.resize(image, (224, 224))
    # Normalize pixel values
    image = image / 255.0
    # Expand dims so that image has shape (1, 224, 224, 3)
    return np.expand_dims(image, axis=0)

# Function to perform weighted ensemble prediction
def ensemble_prediction(image):
    processed_image = preprocess_image(image)
    
    # Get predictions from each model
    pred_cnn = model_cnn.predict(processed_image)
    pred_inception = model_inception.predict(processed_image)
    pred_vgg16 = model_vgg16.predict(processed_image)
    
    # Define weights for each model (adjust these based on validation performance)
    weights = [0.3, 0.4, 0.3]
    
    

    
    # Weighted average of predictions
    ensemble_pred = (weights[0] * pred_cnn) + (weights[1] * pred_inception) + (weights[2] * pred_vgg16)
    # Flatten the prediction to a 1D array (shape: (5,))
    ensemble_pred = ensemble_pred.flatten()
    
    # Get the class with the highest probability and its confidence
    final_class_index = np.argmax(ensemble_pred)
    final_class = class_labels[final_class_index]
    final_confidence = ensemble_pred[final_class_index] * 100  # as a percentage
    return ensemble_pred, final_class, final_confidence

# Streamlit App UI
st.title("Diabetic Retinopathy Detection with Weighted Ensemble")
st.write("Upload a fundus image to see the confidence for all classes and the final prediction.")

# File uploader widget
uploaded_file = st.file_uploader("Choose a fundus image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Read the uploaded image file
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        # Get ensemble predictions and final result
        ensemble_pred, final_class, final_confidence = ensemble_prediction(image)
        
        # Prepare a DataFrame to display confidence for all classes
        confidences = (ensemble_pred * 100).round(2)  # Convert to percentage and round off
        df_confidences = pd.DataFrame({
            "Diabetic Retinopathy Class": class_labels,
            "Confidence (%)": confidences
        })
        
        st.subheader("Confidence Intervals for All Classes")
        st.dataframe(df_confidences)  # Display the DataFrame
        st.bar_chart(df_confidences.set_index("Diabetic Retinopathy Class"))
        
        st.subheader("Final Prediction")
        st.success(f"Prediction: **{final_class}**")
        st.info(f"Confidence: **{final_confidence:.2f}%**")
