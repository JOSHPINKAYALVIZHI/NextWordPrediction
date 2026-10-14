
import json
import pickle
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

st.set_page_config(
    page_title="NextWord AI",
    page_icon="⌨️",
    layout="wide"
)

st.title("⌨️ NextWord AI")
st.caption("LSTM-powered next-word prediction system")

# Check whether the model has been trained
required_files = [
    MODEL_DIR / "next_word_model.keras",
    MODEL_DIR / "tokenizer.pkl",
    MODEL_DIR / "config.json"
]

if not all(path.exists() for path in required_files):
    st.error("Trained model not found. Run `python train.py` first.")
    st.stop()


@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(
        MODEL_DIR / "next_word_model.keras"
    )

    with open(MODEL_DIR / "tokenizer.pkl", "rb") as file:
        tokenizer = pickle.load(file)

    with open(MODEL_DIR / "config.json", "r", encoding="utf-8") as file:
        config = json.load(file)

    return model, tokenizer, config


model, tokenizer, config = load_artifacts()


def predict_next_words(text, top_k=3):
    words = text.lower().strip().split()

    if not words:
        return []

    sequence = tokenizer.texts_to_sequences([" ".join(words)])[0]

    if not sequence:
        return []

    max_len = config["max_context_len"]
    sequence = sequence[-max_len:]

    padded = pad_sequences(
        [sequence],
        maxlen=max_len,
        padding="pre"
    )

    logits = model.predict(padded, verbose=0)[0]
    probabilities = tf.nn.softmax(logits).numpy()

    # Exclude padding and unknown-token IDs
    probabilities[0] = 0

    oov_id = tokenizer.word_index.get("<OOV>")
    if oov_id is not None:
        probabilities[oov_id] = 0

    top_ids = np.argsort(probabilities)[-top_k:][::-1]

    results = []
    for token_id in top_ids:
        word = tokenizer.index_word.get(int(token_id))

        if word and probabilities[token_id] > 0:
            results.append({
                "word": word,
                "probability": float(probabilities[token_id])
            })

    return results


# Main prediction section
st.subheader("💬 Try a prediction")

user_text = st.text_input(
    "Enter your sentence",
    value="i am going to",
    placeholder="Example: i want to learn"
)

if st.button("Predict next words", type="primary"):
    if not user_text.strip():
        st.warning("Please enter a sentence.")
    else:
        predictions = predict_next_words(user_text, top_k=3)

        if predictions:
            st.success("Prediction complete!")

            cols = st.columns(len(predictions))

            for col, prediction in zip(cols, predictions):
                with col:
                    st.metric(
                        label="Suggested word",
                        value=prediction["word"],
                        delta=f'{prediction["probability"] * 100:.2f}%'
                    )
                    st.progress(
                        min(prediction["probability"], 1.0)
                    )
        else:
            st.info(
                "No suitable predictions. Try words from the training dataset."
            )

st.caption(
    "Probabilities are model scores, not guarantees of correctness."
)

# Evaluation section
st.divider()
st.header("📊 Model Evaluation")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Test Accuracy", f'{config["test_accuracy"] * 100:.2f}%')
c2.metric("Test Loss", f'{config["test_loss"]:.4f}')
c3.metric("Top-1 Accuracy", f'{config["top1_accuracy"] * 100:.2f}%')
c4.metric("Top-3 Accuracy", f'{config["top3_accuracy"] * 100:.2f}%')
c5.metric("Top-5 Accuracy", f'{config["top5_accuracy"] * 100:.2f}%')

st.subheader("📚 Dataset Information")

d1, d2 = st.columns(2)
d1.metric("Dataset Sentences", config["dataset_sentences"])
d2.metric("Vocabulary Size", config["vocab_size"])

# Training history graphs
st.divider()
st.header("📈 Training History")

history_path = MODEL_DIR / "history.pkl"

if history_path.exists():
    try:
        with open(history_path, "rb") as file:
            history = pickle.load(file)

        history_data = history.history if hasattr(history, "history") else history

        if isinstance(history_data, dict):
            history_df = pd.DataFrame(history_data)
            history_df.index = range(1, len(history_df) + 1)
            history_df.index.name = "Epoch"

            left, right = st.columns(2)

            with left:
                st.subheader("Model Accuracy")
                accuracy_columns = [
                    col for col in ["accuracy", "val_accuracy"]
                    if col in history_df.columns
                ]

                if accuracy_columns:
                    st.line_chart(history_df[accuracy_columns])
                else:
                    st.info("Accuracy history is not available.")

            with right:
                st.subheader("Model Loss")
                loss_columns = [
                    col for col in ["loss", "val_loss"]
                    if col in history_df.columns
                ]

                if loss_columns:
                    st.line_chart(history_df[loss_columns])
                else:
                    st.info("Loss history is not available.")

            st.caption(
                "Training curves show performance across epochs. "
                "Validation results help assess generalization."
            )
        else:
            st.info("Training history format is not recognized.")

    except (OSError, pickle.PickleError, ValueError, TypeError) as error:
        st.warning(f"Could not load training history: {error}")
else:
    st.info("Training history not found. Run `python train.py` first.")

# About section
st.divider()
st.header("ℹ️ About the Project")

st.write(
    "NextWord AI uses an Embedding layer and an LSTM neural network "
    "to learn word sequences and suggest likely next words."
)

st.write(
    "**Technology:** Python, TensorFlow/Keras, NumPy, Pandas and Streamlit."
)

st.write(
    "**Privacy:** Prediction runs locally using the trained model; "
    "the app does not need to send the entered text to an external AI API."
)
