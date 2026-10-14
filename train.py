
import json
import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

np.random.seed(42)
tf.random.set_seed(42)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "sentences.txt"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# 1. Load sentences
with open(DATA_FILE, "r", encoding="utf-8") as file:
    sentences = [
        line.strip().lower()
        for line in file
        if line.strip()
    ]

if len(sentences) < 20:
    raise ValueError("Add more sentences before training.")

print(f"Total sentences: {len(sentences)}")

# 2. Split complete sentences before creating sequences
train_sentences, temp_sentences = train_test_split(
    sentences, test_size=0.20, random_state=42
)

val_sentences, test_sentences = train_test_split(
    temp_sentences, test_size=0.50, random_state=42
)

print(f"Training sentences: {len(train_sentences)}")
print(f"Validation sentences: {len(val_sentences)}")
print(f"Test sentences: {len(test_sentences)}")

# 3. Fit tokenizer ONLY on training sentences
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(train_sentences)

vocab_size = len(tokenizer.word_index) + 1
train_tokens = tokenizer.texts_to_sequences(train_sentences)
val_tokens = tokenizer.texts_to_sequences(val_sentences)
test_tokens = tokenizer.texts_to_sequences(test_sentences)

# 4. Create next-word sequences independently per sentence
def create_sequences(tokenized_sentences):
    sequences = []

    for tokens in tokenized_sentences:
        for i in range(2, len(tokens) + 1):
            sequences.append(tokens[:i])

    return sequences

train_sequences = create_sequences(train_tokens)
val_sequences = create_sequences(val_tokens)
test_sequences = create_sequences(test_tokens)

if not train_sequences or not val_sequences or not test_sequences:
    raise ValueError("Not enough usable word sequences. Expand the dataset.")

max_sequence_len = max(len(seq) for seq in train_sequences)
max_context_len = max_sequence_len - 1

# 5. Pad inputs and separate next-word targets
def prepare_xy(sequences):
    padded = pad_sequences(
        sequences,
        maxlen=max_sequence_len,
        padding="pre",
        truncating="pre"
    )
    return padded[:, :-1], padded[:, -1]

X_train, y_train = prepare_xy(train_sequences)
X_val, y_val = prepare_xy(val_sequences)
X_test, y_test = prepare_xy(test_sequences)

print(f"Vocabulary size: {vocab_size}")
print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")

# 6. Build LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(max_context_len,)),
    tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=64,
        mask_zero=True
    ),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(vocab_size)
])

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True
    ),
    metrics=["accuracy"]
)

model.summary()

# 7. Train
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=40,
    batch_size=16,
    callbacks=[early_stopping],
    verbose=1
)

# 8. Evaluate on held-out sentences
test_loss, test_accuracy = model.evaluate(
    X_test, y_test, verbose=0
)

logits = model.predict(X_test, verbose=0)
probabilities = tf.nn.softmax(logits, axis=1).numpy()

def top_k_accuracy(probabilities, targets, k):
    k = min(k, probabilities.shape[1])
    top_ids = np.argsort(probabilities, axis=1)[:, -k:]
    return float(np.mean([
        target in predictions
        for target, predictions in zip(targets, top_ids)
    ]))

top1 = top_k_accuracy(probabilities, y_test, 1)
top3 = top_k_accuracy(probabilities, y_test, 3)
top5 = top_k_accuracy(probabilities, y_test, 5)

print("\n========== MODEL EVALUATION ==========")
print(f"Test Loss:      {test_loss:.4f}")
print(f"Test Accuracy:  {test_accuracy * 100:.2f}%")
print(f"Top-1 Accuracy: {top1 * 100:.2f}%")
print(f"Top-3 Accuracy: {top3 * 100:.2f}%")
print(f"Top-5 Accuracy: {top5 * 100:.2f}%")

# 9. Save model and artifacts
model.save(MODEL_DIR / "next_word_model.keras")

with open(MODEL_DIR / "tokenizer.pkl", "wb") as file:
    pickle.dump(tokenizer, file)

config = {
    "vocab_size": vocab_size,
    "max_context_len": max_context_len,
    "test_loss": float(test_loss),
    "test_accuracy": float(test_accuracy),
    "top1_accuracy": top1,
    "top3_accuracy": top3,
    "top5_accuracy": top5,
    "dataset_sentences": len(sentences),
    "training_sentences": len(train_sentences),
    "validation_sentences": len(val_sentences),
    "test_sentences": len(test_sentences),
    "training_samples": len(X_train),
    "validation_samples": len(X_val),
    "test_samples": len(X_test)
}

with open(MODEL_DIR / "config.json", "w", encoding="utf-8") as file:
    json.dump(config, file, indent=4)

with open(MODEL_DIR / "history.pkl", "wb") as file:
    pickle.dump(history.history, file)

print(f"\nTraining completed. Files saved to: {MODEL_DIR}")
