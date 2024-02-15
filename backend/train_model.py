import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import layers, models
from keras.applications import VGG16

# Directory containing subdirectories for each class (0 to 18)
image_dir = 'images'

# List of emotion labels
emotion_labels = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprised']

# Load images and labels
images = []
labels = []

for label_id, label in enumerate(emotion_labels):
    label_dir = os.path.join(image_dir, str(label_id))
    for filename in os.listdir(label_dir):
        if filename.endswith('.jpg'):
            image_path = os.path.join(label_dir, filename)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            image = cv2.resize(image, (48, 48))  # Resize to 48x48
            images.append(image)
            labels.append(label_id)  # Assign label ID

# Convert grayscale images to RGB format
images_rgb = np.repeat(np.array(images)[..., np.newaxis], 3, -1)

# Convert lists to numpy arrays
labels = np.array(labels)

# Split the dataset
X_train, X_val, y_train, y_val = train_test_split(images_rgb, labels, test_size=0.2, random_state=42)

# Load pre-trained VGG16 model (without top layers)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(48, 48, 3))

# Freeze the convolutional base
base_model.trainable = False

# Add custom classification head
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(emotion_labels), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Evaluate the model
loss, accuracy = model.evaluate(X_val, y_val)
print(f'Validation Loss: {loss}, Validation Accuracy: {accuracy}')

# Save the model
model.save('emotion_detection_model_vgg.h5')
