import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 224


def load_data(base_path):
    X, y = [], []

    for label in ['NORMAL', 'PNEUMONIA']:
        path = os.path.join(base_path, label)
        lab = 0 if label == 'NORMAL' else 1

        for f in os.listdir(path):
            try:
                img = cv2.imread(os.path.join(path, f))

                if img is None:
                    continue

                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

                # Keep images as uint8 to reduce memory usage.
                # Normalization is performed during feature extraction.
                X.append(img)
                y.append(lab)

            except Exception:
                pass

    return np.array(X, dtype=np.uint8), np.array(y)


def visualize_augmented(X):
    gen = ImageDataGenerator(
        rotation_range=15,
        horizontal_flip=True
    )

    plt.figure(figsize=(10, 6))

    for i in range(3):
        # Convert only the image being displayed to float.
        img = X[i].astype(np.float32) / 255.0

        plt.subplot(2, 3, i + 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis("off")

        aug = gen.random_transform(img)

        plt.subplot(2, 3, i + 4)
        plt.imshow(cv2.cvtColor(aug, cv2.COLOR_BGR2RGB))
        plt.axis("off")

    plt.savefig("results/sample_images.png")
    plt.close()


def class_distribution(y):
    c = Counter(y)

    plt.bar(
        ['Normal', 'Pneumonia'],
        [c[0], c[1]]
    )

    plt.savefig("results/class_distribution.png")
    plt.close()
