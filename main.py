import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.utils import custom_object_scope
from tensorflow.keras import layers
from tensorflow.keras.layers import *
from time import sleep

# Load the .h5 model (for Model 1)
model_path_1 = "temp.h5"  # Update with the correct path to your .h5 file
model_1 = tf.keras.models.load_model(model_path_1)

with tf.device('/cpu:0'):
    model_1.compile()

# Dehazing function for Model 1 using the loaded .h5 model
def dehaze_model_1(image):
    # Preprocess image for model input
    img_array = np.array(image.resize((548,412))) / 255.0  # Assuming the model expects 256x256 images
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Apply the model to the image
    dehazed_img = model_1.predict(img_array)

    # Post-process the output image
    dehazed_img = np.squeeze(dehazed_img, axis=0)  # Remove batch dimension
    dehazed_img = (dehazed_img * 255).astype(np.uint8)  # Convert to uint8 for display

    return Image.fromarray(dehazed_img)

# Dictionary to hold model descriptions and functions
models = {
    "GMAN NET": (dehaze_model_1,'gman-h5.png', "Generic Model-Agnostic Neural Network for Dehazing, presents a CNN-based method that bypasses traditional atmospheric models, focusing instead on learning a direct mapping between hazy and clean images. This model employs an encoder-decoder architecture with residual learning, producing high-quality dehazed images without relying on parameter estimation(gmannet)."),
}

# Create the sidebar for page selection
st.sidebar.title("Satellite Image Dehazing")
st.sidebar.image('sidebar.png', use_column_width=True)
st.sidebar.title("Select Model")
model_selection = st.sidebar.radio("Choose a model:", list(models.keys()))

# Get the selected model's function and description
model_function, diagram_path, model_description = models[model_selection]

# Display model description
st.title(model_selection)
st.write(model_description)
st.image(diagram_path, caption=f'{model_selection} Architecture', use_column_width=True)

# Image upload
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Dehaze button
    if st.button("Dehaze"):
        # Dehaze the image using the selected model
        dehazed_image = model_function(image)
        
        # Display the dehazed image
        st.image(dehazed_image, caption='Dehazed Image', use_column_width=True)
