import pandas as pd
import numpy as np

# Read the dataset
dataset = pd.read_csv("Your dataset csv file path")

# Extract emotions and pixel values
emotions = dataset["emotion"].values
pixels = dataset["pixels"].values

# Convert pixel values to numpy arrays
images = np.zeros((pixels.shape[0], 48, 48, 1))
for i in range(pixels.shape[0]):
    img = np.fromstring(pixels[i], dtype=int, sep=" ")
    img = img.reshape((48, 48, 1))
    images[i] = img

# Normalize pixel values
images = images.astype("float32") / 255

# Print the shapes of images and emotions arrays
print(images.shape)
print(emotions.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(images, emotions, test_size=0.2, random_state=42)
from tensorflow import keras
from tensorflow.keras import layers

# Define the model architecture
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(7, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)
model.save('my_model.h5')
