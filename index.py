import matplotlib.pyplot as plt
import numpy as np
import PIL

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from pathlib import Path

#download and explore data set

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file("flower_photos.tgz", origin=dataset_url, cache_dir="/tmp", extract=True)
data_dir = Path(data_dir).with_suffix('')
image_count = len(list(data_dir.glob('*/*.jpg')))
print(f"Total images in the dataset: {image_count}")

print("\nSUCCESS - images downloaded and explored\n")

#load the data using Keras

batch_size = 32
img_height, img_width = 180, 180

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

print("\nSUCCESS - data loaded from disk\n")


# Creating the CNN model for image classification

num_classes = len(train_ds.class_names)

model = Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("\nSUCCESS - CNN model created\n")

# Training the model

print("\nPROGRESS... - Model training initiated\n")
epochs = 10
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
print("\nSUCCESS - Model training completed\n")

# Evaluate model

print("\nPROGRESS... - Model evaluating initiated\n")
test_loss, test_acc = model.evaluate(val_ds)
print(f"Validation accuracy: {test_acc:.4f}")
print("\nSUCCESS - Program terminated\n")