import pandas as pd
import numpy as np
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer

import tensorflow as tf
from tensorflow.keras import layers, models

# =======================
# 1. Load Data
# =======================

SAMPLE_FILE_PATH = 'data/philosophy_data.csv'

print(f"Loading data from: {SAMPLE_FILE_PATH}")
if not os.path.exists(SAMPLE_FILE_PATH):
    print("Error: sample file not found.")
    exit()

df = pd.read_csv(SAMPLE_FILE_PATH)
df = df.dropna()
print(f"Loaded {len(df)} rows.")

# =======================
# 2. Prepare TF-IDF
# =======================

print("\nStarting TF-IDF Vectorization...")
my_stop_words = list(text.ENGLISH_STOP_WORDS) + ['pron']

tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words=my_stop_words
)

start_time = time.time()
X_tfidf = tfidf_vectorizer.fit_transform(df['lemmatized_str'])
print(f"TF-IDF done in {time.time() - start_time:.2f}s")

y = df['school']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# =======================
# 3. Encode Labels
# =======================

# Convert labels to integers
unique_classes = sorted(list(y.unique()))
class_to_idx = {c: i for i, c in enumerate(unique_classes)}
idx_to_class = {i: c for c, i in class_to_idx.items()}

# Apply transform
y_train_idx = np.array([class_to_idx[c] for c in y_train])
y_test_idx = np.array([class_to_idx[c] for c in y_test])

# =======================
# 4. Build NN
# =======================

# Convert sparse matrix to dense (Keras needs dense)
# Keep it only for training batches, not all at once
X_train_dense = X_train.toarray()
X_test_dense = X_test.toarray()

input_dim = X_train_dense.shape[1]
num_classes = len(unique_classes)

print(f"\nInput dim: {input_dim}, Classes: {num_classes}")

# Build model
model = models.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nNN model summary:")
model.summary()

# =======================
# 5. Train NN
# =======================

print("\nTraining NN...")
start_time = time.time()

history = model.fit(
    X_train_dense,
    y_train_idx,
    epochs=5,             # keep small for first run
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

print(f"Training done in {time.time() - start_time:.2f}s")

# =======================
# 6. Evaluate
# =======================

print("\nEvaluating NN...")
y_pred_probs = model.predict(X_test_dense)
y_pred = np.argmax(y_pred_probs, axis=1)

acc = accuracy_score(y_test_idx, y_pred)
print(f"Test Accuracy: {acc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test_idx, y_pred, target_names=unique_classes, zero_division=0))

print("\n--- TF-IDF NN Complete ---")