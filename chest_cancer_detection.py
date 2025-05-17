import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# Base dataset path
data_dir = "/kaggle/input/chest-ctscan-images/Data"

# Dataset splits
splits = ['train', 'test', 'valid']

# Ensure the dataset structure exists
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Dataset path '{data_dir}' not found. Check the dataset location.")

# Parameters
img_size = 128  # Resize images for uniformity
X = []
y = []
categories = []

# Load data from train set
train_path = os.path.join(data_dir, 'train')
categories = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]

for category in categories:
    folder_path = os.path.join(train_path, category)
    label = categories.index(category)
    
    image_files = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img_resized = cv2.resize(img, (img_size, img_size))
            X.append(img_resized)
            y.append(label)

# Convert to NumPy arrays
X = np.array(X).reshape(-1, img_size, img_size, 1) / 255.0  # Normalize pixel values
y = to_categorical(np.array(y), num_classes=len(categories))

# Split train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Build an improved CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(categories), activation='softmax')
])

# Compile model with Adam optimizer
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_valid, y_valid), verbose=1)

# Evaluate on the test set
y_pred = model.predict(X_test)
print(classification_report(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), target_names=categories))
