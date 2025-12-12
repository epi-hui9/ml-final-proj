import pandas as pd
import numpy as np
import spacy
import os
import time
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# --- Configuration ---
SAMPLE_FILE_PATH = 'data/philosophy_sample_50k.csv'
SPACY_MODEL_NAME = 'en_core_web_md'
BATCH_SIZE = 32
EPOCHS = 15  # LSTMs can take longer to converge, but 15 is a good start
LEARNING_RATE = 0.001
MAX_SEQ_LEN = 100  # We will cut sentences longer than this and pad shorter ones
EMBEDDING_DIM = 300  # Dimension of GloVe vectors in spaCy models

# --- 1. Setup Device (Apple Silicon / M2 Support) ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Success: Using Apple M-series MPS acceleration!")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅ Success: Using NVIDIA CUDA acceleration!")
else:
    device = torch.device("cpu")
    print("⚠️ Using CPU (No GPU found).")

# --- 2. Load Data ---
print(f"\nLoading data from {SAMPLE_FILE_PATH}...")
if not os.path.exists(SAMPLE_FILE_PATH):
    print(f"Error: {SAMPLE_FILE_PATH} not found.")
    exit()

df = pd.read_csv(SAMPLE_FILE_PATH)
df = df.dropna()
print(f"Loaded {len(df)} rows.")

# --- 3. Build Vocabulary & Embedding Matrix ---
print(f"\nLoading spaCy model '{SPACY_MODEL_NAME}' to build vocabulary...")
try:
    nlp = spacy.load(SPACY_MODEL_NAME)
except OSError:
    print(f"Error: Could not load '{SPACY_MODEL_NAME}'.")
    exit()

print("Building vocabulary from dataset...")
# Tokenize all sentences (split by space since it is lemmatized_str)
# We use a simple whitespace split because the data is already preprocessed.
all_text = df['lemmatized_str'].astype(str).tolist()
words = [sentence.split() for sentence in all_text]

# Count all words to build a vocab
word_counts = Counter([word for sentence in words for word in sentence])
# Sort by frequency (most common first)
sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)

# Create mapping: Word -> Integer ID
# We reserve 0 for padding (<PAD>) and 1 for unknown words (<UNK>)
vocab_to_int = {word: i + 2 for i, word in enumerate(sorted_vocab)}
vocab_to_int['<PAD>'] = 0
vocab_to_int['<UNK>'] = 1
vocab_size = len(vocab_to_int)

print(f"Vocabulary size: {vocab_size} unique words.")

# Create Embedding Matrix
# This is a lookup table: Row 0 is vectored for <PAD>, Row 1 for <UNK>, Row 2 for "the"...
print("Creating embedding matrix (copying weights from spaCy)...")
embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))


# Helper to get vector from spaCy
def get_vector(word):
    return nlp.vocab[word].vector


# Fill the matrix
for word, i in vocab_to_int.items():
    if word == '<PAD>':
        continue  # Leave as zeros
    elif word == '<UNK>':
        # Initialize a random vector for unknowns (or use mean)
        embedding_matrix[i] = np.random.normal(scale=0.6, size=(EMBEDDING_DIM,))
    else:
        # Check if spaCy has a vector for this word
        if nlp.vocab.has_vector(word):
            embedding_matrix[i] = get_vector(word)
        else:
            # If spaCy doesn't know it, treat as random
            embedding_matrix[i] = np.random.normal(scale=0.6, size=(EMBEDDING_DIM,))

# Convert to Tensor
embedding_tensor = torch.tensor(embedding_matrix, dtype=torch.float32)
print("Embedding matrix ready.")

# --- 4. Encode Sentences (Tokenize & Pad) ---
print("\nEncoding and padding sequences...")
# Convert words to integers
encoded_sentences = []
for sentence in words:
    encoded_sent = [vocab_to_int.get(word, vocab_to_int['<UNK>']) for word in sentence]
    encoded_sentences.append(encoded_sent)

# Pad sequences to MAX_SEQ_LEN
# If shorter, add 0s. If longer, cut off.
features = np.zeros((len(encoded_sentences), MAX_SEQ_LEN), dtype=int)
for i, sent in enumerate(encoded_sentences):
    length = len(sent)
    if length != 0:
        # Take the first MAX_SEQ_LEN words
        sent = sent[:MAX_SEQ_LEN]
        features[i, :len(sent)] = sent

X = features

# --- 5. Prepare Labels ---
y_text = df['school']
encoder = LabelEncoder()
y_int = encoder.fit_transform(y_text)
num_classes = len(encoder.classes_)

# --- 6. Split and Create Loaders ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y_int, test_size=0.2, random_state=42, stratify=y_int
)

# Create Tensor Datasets
# Note: Input X must be LongTensor (integers) because they are indices for lookup
train_data = TensorDataset(torch.tensor(X_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.long))
test_data = TensorDataset(torch.tensor(X_test, dtype=torch.long), torch.tensor(y_test, dtype=torch.long))

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)


# --- 7. Define LSTM Model ---
class PhilosophyLSTM(nn.Module):
    def __init__(self, vocab_size, output_dim, embedding_dim, hidden_dim, n_layers, weights):
        super(PhilosophyLSTM, self).__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # 1. Embedding Layer
        # We load our pre-trained spaCy weights here.
        # freeze=False allows the model to fine-tune the vectors to our specific dataset
        self.embedding = nn.Embedding.from_pretrained(weights, freeze=False, padding_idx=0)

        # 2. LSTM Layer
        # batch_first=True means input shape is (batch, seq_len, features)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            dropout=0.5, batch_first=True)

        # 3. Dropout
        self.dropout = nn.Dropout(0.3)

        # 4. Fully Connected Layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: [batch_size, seq_len]

        # Get embeddings
        # embeds shape: [batch_size, seq_len, embedding_dim]
        embeds = self.embedding(x)

        # Pass through LSTM
        # lstm_out shape: [batch_size, seq_len, hidden_dim]
        # hidden shape: [n_layers, batch_size, hidden_dim] (The final state)
        lstm_out, (hidden, cell) = self.lstm(embeds)

        # We want the output from the LAST time step
        # Since we padded with zeros, we ideally want the last non-zero step,
        # but grabbing the end of the sequence is a standard simplified approach.
        # To be safe, we use the "hidden" state returned by the LSTM, which represents the end.
        # hidden[-1] gives the state of the last layer
        last_hidden = hidden[-1]

        last_hidden = self.dropout(last_hidden)

        out = self.fc(last_hidden)
        return out


# Initialize Model
hidden_dim = 128
n_layers = 2  # Stack 2 LSTMs on top of each other

model = PhilosophyLSTM(vocab_size, num_classes, EMBEDDING_DIM, hidden_dim, n_layers, embedding_tensor)
model.to(device)

print("\nModel Structure:")
print(model)

# --- 8. Training ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("\nStarting training...")
start_train_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, labels)
        loss.backward()

        # Clip gradients to prevent "exploding gradient" problem in LSTMs
        nn.utils.clip_grad_norm_(model.parameters(), 5)

        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {running_loss / len(train_loader):.4f} - Acc: {correct / total:.4f}")

print(f"Training finished in {time.time() - start_train_time:.2f} seconds.")

# --- 9. Evaluation ---
print("\nEvaluating...")
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        output = model(inputs)
        _, predicted = torch.max(output.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\nClassification Report (LSTM):")
print(classification_report(all_labels, all_preds, target_names=encoder.classes_, zero_division=0))
