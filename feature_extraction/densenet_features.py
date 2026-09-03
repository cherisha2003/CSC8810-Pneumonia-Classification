import numpy as np
from tensorflow.keras.applications import DenseNet121


def get_model():
    return DenseNet121(
        weights='imagenet',
        include_top=False,
        pooling='avg'
    )


def extract_features(model, X, batch_size=16):
    features = []

    total = len(X)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)

        # Only convert the current batch to float32.
        batch = X[start:end].astype(np.float32) / 255.0

        batch_features = model.predict(
            batch,
            batch_size=batch_size,
            verbose=0
        )

        features.append(batch_features)

        print(
            f"DenseNet feature extraction: "
            f"{end}/{total} images"
        )

    return np.concatenate(features, axis=0)
