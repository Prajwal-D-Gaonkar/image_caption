import os
import pickle
import numpy as np

from tqdm import tqdm

import tensorflow as tf

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input


# ==============================
# PATHS
# ==============================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

IMAGE_DIR = os.path.join(
    BASE_DIR,
    "data",
    "images"
)

FEATURE_FILE = os.path.join(
    BASE_DIR,
    "models",
    "features.pkl"
)


# ==============================
# LOAD CNN
# ==============================

print("Loading EfficientNetB0...")

cnn = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

cnn.trainable = False

print("CNN loaded successfully!")


# ==============================
# EXTRACT FEATURES
# ==============================

def extract_features():

    image_files = [
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    print(
        "Images found:",
        len(image_files)
    )

    features = {}

    for filename in tqdm(
        image_files,
        desc="Extracting features"
    ):

        image_path = os.path.join(
            IMAGE_DIR,
            filename
        )

        try:

            image = tf.keras.utils.load_img(
                image_path,
                target_size=(224, 224)
            )

            image = tf.keras.utils.img_to_array(
                image
            )

            image = np.expand_dims(
                image,
                axis=0
            )

            image = preprocess_input(
                image
            )

            feature = cnn.predict(
                image,
                verbose=0
            )

            features[filename] = feature[0]

        except Exception as e:

            print(
                f"Error processing {filename}: {e}"
            )

    os.makedirs(
        os.path.dirname(FEATURE_FILE),
        exist_ok=True
    )

    with open(
        FEATURE_FILE,
        "wb"
    ) as f:

        pickle.dump(
            features,
            f
        )

    print("\nFeatures saved!")
    print(FEATURE_FILE)

    print(
        "Total features:",
        len(features)
    )


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":

    extract_features()