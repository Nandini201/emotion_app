import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

# Path to dataset folder
DATASET_PATH = "/Users/nandinigoswami/Downloads/Dataset of Indian face images with various expressions"

# Image size (resize to this)
IMG_SIZE = 48

# Supported image file extensions
SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png')

# Load dataset
def load_data(dataset_path):
    labels = sorted([folder for folder in os.listdir(dataset_path) if not folder.startswith('.')])
    label_map = {label: idx for idx, label in enumerate(labels)}

    images = []
    image_labels = []

    for label in labels:
        folder = os.path.join(dataset_path, label)
        if not os.path.isdir(folder):
            continue
        for img_file in os.listdir(folder):
            # Skip hidden files and unsupported file formats
            if img_file.startswith('.') or not img_file.lower().endswith(SUPPORTED_FORMATS):
                print(f"Skipping file: {os.path.join(folder, img_file)}")
                continue

            img_path = os.path.join(folder, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
            if img is None:
                print(f"Skipping unreadable file: {img_path}")
                continue

            # Resize image
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            images.append(img)
            image_labels.append(label_map[label])

    images = np.array(images)
    images = images.reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0  # Normalize
    labels_cat = to_categorical(image_labels, num_classes=len(labels))
    return images, labels_cat, labels


# Build CNN model
def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == "__main__":
    print("Loading data...")
    X, y, class_labels = load_data(DATASET_PATH)
    print(f"Data loaded: {X.shape[0]} samples, {len(class_labels)} classes")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = build_model((IMG_SIZE, IMG_SIZE, 1), len(class_labels))
    print("Training model...")
    model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_val, y_val))

    # Save the model
    model.save("emotion_model.keras")
    print("Model saved as emotion_model.keras")

