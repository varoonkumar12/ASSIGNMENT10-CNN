import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

def make_prediction(img_path, model_path="saved_models/model.h5"):
    # Check if model exists
    if not os.path.exists(model_path):
        return None, "Model file not found. Please train the model first."
    
    # Load Model
    model = tf.keras.models.load_model(model_path)
    
    # Preprocess Image (Requirement D)
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    predictions = model.predict(img_array)
    
    # Get all class names from Dataset folder
    classes = sorted([d for d in os.listdir("Dataset") if os.path.isdir(os.path.join("Dataset", d))])
    
    result_label = classes[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    
    return result_label, confidence