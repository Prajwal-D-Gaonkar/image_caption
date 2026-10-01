import os
import pickle
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.applications import EfficientNetB0

from tensorflow.keras.applications.efficientnet import (
    preprocess_input
)

from tensorflow.keras.models import load_model


# ==============================
# PATHS
# ==============================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_FILE = os.path.join(
    BASE_DIR,
    "models",
    "caption_model.keras"
)

TOKENIZER_FILE = os.path.join(
    BASE_DIR,
    "models",
    "tokenizer.pkl"
)


# ==============================
# LOAD MODEL
# ==============================

print("Loading caption model...")

model = load_model(
    MODEL_FILE
)


# ==============================
# LOAD TOKENIZER
# ==============================

with open(
    TOKENIZER_FILE,
    "rb"
) as f:

    tokenizer = pickle.load(f)


index_word = {
    value: key
    for key, value
    in tokenizer.word_index.items()
}


# ==============================
# LOAD CNN
# ==============================

cnn = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

cnn.trainable = False


# ==============================
# MAX LENGTH
# ==============================

max_length = model.input[
    1
].shape[1]


# ==============================
# GENERATE CAPTION
# ==============================

def generate_caption(
    image_path
):

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

    caption = "startseq"


    for _ in range(max_length):

        sequence = tokenizer.texts_to_sequences(
            [caption]
        )[0]

        sequence = pad_sequences(
            [sequence],
            maxlen=max_length,
            padding="post"
        )

        prediction = model.predict(
            [
                feature,
                sequence
            ],
            verbose=0
        )[0]

        predicted_id = np.argmax(
            prediction
        )

        word = index_word.get(
            predicted_id,
            ""
        )

        if word == "endseq":
            break

        if word == "":
            break

        caption += " " + word


    caption = caption.replace(
        "startseq",
        ""
    ).strip()


    return caption


# ==============================
# TEST
# ==============================

if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:

        print(
            "Usage:"
        )

        print(
            "python src/caption.py path_to_image.jpg"
        )

        exit()


    image_path = sys.argv[1]

    result = generate_caption(
        image_path
    )

    print(
        "\nGenerated Caption:"
    )

    print(
        result
    )