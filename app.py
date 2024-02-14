import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import tensorflow_hub as hub
# Load MobileNet V2 model from TensorFlow Hub
def create_model():
    mobilenet_model = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4'
    feature_extractor_layer = hub.KerasLayer(mobilenet_model, input_shape=(224, 224, 3), trainable=False)
    model = tf.keras.Sequential([
        feature_extractor_layer,
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    return model

# Function to preprocess image
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to make predictions
def predict(image, model):
    image = preprocess_image(image)
    prediction = model.predict(image)
    return prediction

# Main function
def main():
    st.title('Meow🐱 and Woof🐶 with MobileNet V2')
    st.write('This web application utilizes transfer learning for straightforward image classification')
    # Load the model
    model = create_model()

    # File uploader
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        try:
            # Display uploaded image
            image = np.array(Image.open(uploaded_file))
            st.image(image, caption='Uploaded Image', use_column_width=True)

            # Make prediction
            prediction = predict(image, model)
            if prediction[0][0] > prediction[0][1]:
                st.write('Predicted: Cat')
            else:
                st.write('Predicted: Dog')
        except Exception as e:
            st.error(f"Error occurred: {e}")

# Run the app
if __name__ == '__main__':
    main()
