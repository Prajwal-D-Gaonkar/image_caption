import os
import re
import pickle
import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ==============================
# PATHS
# ==============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CAPTION_FILE = os.path.join(BASE_DIR, "data", "captions.txt")
TOKENIZER_FILE = os.path.join(BASE_DIR, "models", "tokenizer.pkl")


# ==============================
# CLEAN CAPTION
# ==============================

def clean_caption(caption):

    caption = caption.lower()

    # Remove punctuation
    caption = re.sub(r"[^a-z\s]", "", caption)

    # Remove extra spaces
    caption = re.sub(r"\s+", " ", caption).strip()

    # Add special tokens
    caption = "startseq " + caption + " endseq"

    return caption


# ==============================
# LOAD DATASET
# ==============================

def load_captions():

    df = pd.read_csv(CAPTION_FILE)

    print("Columns:", df.columns)

    # Automatically detect columns
    image_col = None
    caption_col = None

    for col in df.columns:

        if "image" in col.lower():
            image_col = col

        if "caption" in col.lower():
            caption_col = col

    if image_col is None or caption_col is None:

        raise ValueError(
            "Could not find image/caption columns in captions.txt"
        )

    df = df[[image_col, caption_col]]

    df.columns = ["image", "caption"]

    df["caption"] = df["caption"].astype(str)
    df["caption"] = df["caption"].apply(clean_caption)

    print("\nDataset loaded")
    print("Total captions:", len(df))
    print("\nExample:")
    print(df.head())

    return df


# ==============================
# TOKENIZER
# ==============================

def create_tokenizer(df):

    tokenizer = Tokenizer(
        num_words=5000,
        oov_token="<unk>"
    )

    tokenizer.fit_on_texts(df["caption"])

    os.makedirs(
        os.path.dirname(TOKENIZER_FILE),
        exist_ok=True
    )

    with open(TOKENIZER_FILE, "wb") as f:

        pickle.dump(tokenizer, f)

    print("\nTokenizer saved to:")
    print(TOKENIZER_FILE)

    print(
        "Vocabulary size:",
        len(tokenizer.word_index) + 1
    )

    return tokenizer


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":

    df = load_captions()

    tokenizer = create_tokenizer(df)

    print("\nPreprocessing completed successfully!")