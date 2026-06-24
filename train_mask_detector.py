import os
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import matplotlib.pyplot as plt

# Paths
mask_dir = "dataset/WithMask"
no_mask_dir = "dataset/WithoutMask"

data = []
labels = []

# Load images
for category, directory in enumerate([mask_dir, no_mask_dir]):  # 0=mask, 1=no mask
    for img in os.listdir(directory):
        img_path = os.path.join(directory, img)
        img_arr = cv2.imread(img_path)
        if img_arr is not None:
            img_arr = cv2.resize(img_arr, (128, 128))
            data.append(img_arr)
            labels.append(category)

# Convert to numpy
X = np.array(data) / 255.0
y = to_categorical(np.array(labels), num_classes=2)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save model
model.save("mask_detector.keras")

# Plot accuracy
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.show()
