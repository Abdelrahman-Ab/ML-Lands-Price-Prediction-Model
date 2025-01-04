import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Data directory
data_dir = r"C:\Users\M.Ghazali\Downloads\LandsAugmented\preview"
land_price_csv = r"C:\Users\M.Ghazali\Downloads\Cleaned_Lands.csv"

# Parameters
img_height, img_width = 32, 32  # LeNet input dimensions
batch_size = 32
epochs = 10

# Load land price data
land_prices = pd.read_csv(land_price_csv)
prices = land_prices.iloc[:, -1].values  # Extract the last column for land prices

# Normalize target values (land prices)
scaler = MinMaxScaler()
prices = scaler.fit_transform(prices.reshape(-1, 1)).flatten()

# Load and preprocess images
img_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg') or f.endswith('.jpeg')]
img_paths = [os.path.join(data_dir, f) for f in img_files]

images = []
for img_path in img_paths:
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    images.append(img_array)

images = np.array(images) / 255.0  # Normalize pixel values to [0, 1]

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(images, prices, test_size=0.2, random_state=42)

# Data augmentation
data_gen = ImageDataGenerator(
    rescale=1.0/255.0,
    horizontal_flip=True,
    zoom_range=0.2
)

train_generator = data_gen.flow(X_train, y_train, batch_size=batch_size)
val_generator = data_gen.flow(X_val, y_val, batch_size=batch_size)

# LeNet architecture
model = Sequential([
    Conv2D(6, kernel_size=(5, 5), activation='tanh', input_shape=(img_height, img_width, 3), padding='same'),
    AveragePooling2D(pool_size=(2, 2)),
    Conv2D(16, kernel_size=(5, 5), activation='tanh', padding='valid'),
    AveragePooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(120, activation='tanh'),
    Dense(84, activation='tanh'),
    Dense(1)  # Output layer for land price prediction
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse', metrics=['mae', 'accuracy'])

# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs
)
# Plot Training and Validation Accuracy
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']