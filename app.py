# RNN Practical (Many to One)
# SMS Spam Detection using Simple RNN

import os
import re
import pickle
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==========================
# Configuration
# ==========================

MODEL = "spam_model.keras"
TOKENIZER = "tokenizer.pkl"

MAX_WORDS = 5000
MAX_LEN = 50


# ==========================
# Clean Text
# ==========================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ==========================
# Train Model
# ==========================

def train_model():

    print("Training Dataset...")

    df = pd.read_csv("spam.csv", encoding="latin-1")

    # Keep only required columns
    df = df[["v1", "v2"]]
    df.columns = ["label", "text"]

    print(df.head())
    print(df["label"].value_counts())

    # Convert labels
    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    # Clean messages
    df["text"] = df["text"].apply(clean_text)

    # Tokenizer
    tokenizer = Tokenizer(
        num_words=MAX_WORDS,
        oov_token="<OOV>"
    )

    tokenizer.fit_on_texts(df["text"])

    sequences = tokenizer.texts_to_sequences(df["text"])

    x = pad_sequences(
        sequences,
        maxlen=MAX_LEN,
        padding="post"
    )

    y = df["label"]

    print("Shape of x:", x.shape)
    print("Shape of y:", y.shape)

    # Save tokenizer
    with open(TOKENIZER, "wb") as f:
        pickle.dump(tokenizer, f)

    # Train-Test Split
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42
    )

    # Build Model
    model = Sequential()

    model.add(
        Embedding(
            input_dim=MAX_WORDS,
            output_dim=32
        )
    )

    model.add(SimpleRNN(128))

    model.add(Dense(32, activation="relu"))

    model.add(Dense(1, activation="sigmoid"))

    model.summary()

    # Compile
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # Train
    model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=10,
        batch_size=32
    )

    # Save Model
    model.save(MODEL)

    # Evaluate
    loss, accuracy = model.evaluate(
        x_test,
        y_test,
        verbose=0
    )

    print("\nAccuracy:", accuracy)

    predictions = (
        model.predict(x_test) > 0.5
    ).astype(int)

    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))


# ==========================
# Prediction Function
# ==========================

def predict_msg(message):

    model = load_model(MODEL)

    with open(TOKENIZER, "rb") as f:
        tokenizer = pickle.load(f)

    message = clean_text(message)

    sequence = tokenizer.texts_to_sequences([message])

    sequence = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post"
    )

    probability = model.predict(
        sequence,
        verbose=0
    )[0][0]

    if probability > 0.5:
        return "Spam", probability
    else:
        return "Ham", probability


# ==========================
# Train Model (only once)
# ==========================

if not os.path.exists(MODEL):
    train_model()


# ==========================
# Streamlit UI
# ==========================

st.title("SMS Spam Detector")

st.write("### Many-to-One RNN Example")

message = st.text_area("Enter SMS Message")

if st.button("Predict"):

    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        prediction, probability = predict_msg(message)

        if prediction == "Spam":
            st.error(f"Prediction : {prediction}")
        else:
            st.success(f"Prediction : {prediction}")

        st.write(
            "Confidence:",round(probability * 100, 2), "%"
        )