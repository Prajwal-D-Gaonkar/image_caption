import os
import pickle
import random

import numpy as np
import pandas as pd

from tqdm import tqdm

import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Model

from tensorflow.keras.layers import (
    Input,
    Dense,
    LSTM,
    Embedding,
    Dropout,
    Add
)

from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping
)


# ==============================
# PATHS
# ==============================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

CAPTION_FILE = os.path.join(
    BASE_DIR,
    "data",
    "captions.txt"
)

FEATURE_FILE = os.path.join(
    BASE_DIR,
    "models",
    "features.pkl"
)

TOKENIZER_FILE = os.path.join(
    BASE_DIR,
    "models",
    "tokenizer.pkl"
)

MODEL_FILE = os.path.join(
    BASE_DIR,
    "models",
    "caption_model.keras"
)


# ==============================
# LOAD DATA
# ==============================

print("Loading captions...")

df = pd.read_csv(
    CAPTION_FILE
)

image_col = None
caption_col = None

for col in df.columns:

    if "image" in col.lower():
        image_col = col

    if "caption" in col.lower():
        caption_col = col

df = df[
    [image_col, caption_col]
]

df.columns = [
    "image",
    "caption"
]


# ==============================
# CLEAN CAPTIONS
# ==============================

import re


def clean_caption(caption):

    caption = caption.lower()

    caption = re.sub(
        r"[^a-z\s]",
        "",
        caption
    )

    caption = re.sub(
        r"\s+",
        " ",
        caption
    ).strip()

    return (
        "startseq "
        + caption
        + " endseq"
    )


df["caption"] = df[
    "caption"
].astype(str).apply(
    clean_caption
)


# ==============================
# LOAD TOKENIZER
# ==============================

with open(
    TOKENIZER_FILE,
    "rb"
) as f:

    tokenizer = pickle.load(f)


vocab_size = min(
    5000,
    len(tokenizer.word_index) + 1
)

print(
    "Vocabulary:",
    vocab_size
)


# ==============================
# MAX LENGTH
# ==============================

max_length = max(
    len(caption.split())
    for caption in df["caption"]
)

print(
    "Maximum caption length:",
    max_length
)


# ==============================
# LOAD FEATURES
# ==============================

print("Loading image features...")

with open(
    FEATURE_FILE,
    "rb"
) as f:

    features = pickle.load(f)

print(
    "Features loaded:",
    len(features)
)


# ==============================
# BUILD TRAINING DATA
# ==============================

X_image = []
X_sequence = []
y = []


print("\nCreating training sequences...")


for _, row in tqdm(
    df.iterrows(),
    total=len(df)
):

    image_name = row["image"]

    if image_name not in features:
        continue

    caption = row["caption"]

    sequence = tokenizer.texts_to_sequences(
        [caption]
    )[0]

    for i in range(
        1,
        len(sequence)
    ):

        input_sequence = sequence[:i]

        target = sequence[i]

        input_sequence = pad_sequences(
            [input_sequence],
            maxlen=max_length,
            padding="post"
        )[0]

        X_image.append(
            features[image_name]
        )

        X_sequence.append(
            input_sequence
        )

        y.append(
            target
        )


X_image = np.array(
    X_image,
    dtype=np.float32
)

X_sequence = np.array(
    X_sequence,
    dtype=np.int32
)

y = np.array(
    y,
    dtype=np.int32
)


print("\nTraining data created")

print(
    "Image features:",
    X_image.shape
)

print(
    "Sequences:",
    X_sequence.shape
)

print(
    "Targets:",
    y.shape
)


# ==============================
# LIMIT DATA FOR FAST TRAINING
# ==============================

MAX_SAMPLES = 150000

if len(X_image) > MAX_SAMPLES:

    print(
        f"\nLimiting dataset to {MAX_SAMPLES} samples..."
    )

    indices = np.random.choice(
        len(X_image),
        MAX_SAMPLES,
        replace=False
    )

    X_image = X_image[
        indices
    ]

    X_sequence = X_sequence[
        indices
    ]

    y = y[
        indices
    ]


# ==============================
# MODEL
# ==============================

print("\nBuilding model...")


image_input = Input(
    shape=(1280,),
    name="image_features"
)

image_dense = Dense(
    256,
    activation="relu"
)(image_input)

image_dense = Dropout(
    0.3
)(image_dense)


caption_input = Input(
    shape=(max_length,),
    name="caption_sequence"
)


embedding = Embedding(
    input_dim=vocab_size,
    output_dim=256,
    mask_zero=True
)(caption_input)


lstm = LSTM(
    256
)(embedding)


merged = Add()([
    image_dense,
    lstm
])


x = Dense(
    256,
    activation="relu"
)(merged)

x = Dropout(
    0.3
)(x)


output = Dense(
    vocab_size,
    activation="softmax"
)(x)


model = Model(
    inputs=[
        image_input,
        caption_input
    ],
    outputs=output
)


# ==============================
# COMPILE
# ==============================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


model.summary()


# ==============================
# CALLBACKS
# ==============================

checkpoint = ModelCheckpoint(
    MODEL_FILE,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)


early_stop = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)


# ==============================
# TRAIN
# ==============================

print("\nStarting training...")


history = model.fit(
    [
        X_image,
        X_sequence
    ],
    y,
    batch_size=256,
    epochs=5,
    validation_split=0.1,
    callbacks=[
        checkpoint,
        early_stop
    ],
    verbose=1
)


# ==============================
# SAVE MODEL
# ==============================

model.save(
    MODEL_FILE
)

print(
    "\nModel saved to:"
)

print(
    MODEL_FILE
)

print(
    "\nTraining completed!"
)