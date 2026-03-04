import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from models.cnn_model import build_cnn_model

def start_training(data_path, epochs=10):
    # Data Preprocessing
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    
    train_gen = datagen.flow_from_directory(
        data_path, target_size=(150, 150), batch_size=32,
        class_mode='categorical', subset='training'
    )
    
    val_gen = datagen.flow_from_directory(
        data_path, target_size=(150, 150), batch_size=32,
        class_mode='categorical', subset='validation', shuffle=False
    )

    model = build_cnn_model(len(train_gen.class_indices))
    
    # Requirement B: Train and show progress
    history = model.fit(train_gen, validation_data=val_gen, epochs=epochs)
    
    # Save Model (Requirement B)
    if not os.path.exists("saved_models"): os.makedirs("saved_models")
    model.save("saved_models/model.h5")
    
    # Requirement C: Evaluation / Confusion Matrix
    preds = model.predict(val_gen)
    y_pred = np.argmax(preds, axis=1)
    cm = confusion_matrix(val_gen.classes, y_pred)
    
    return history, cm, train_gen.class_indices