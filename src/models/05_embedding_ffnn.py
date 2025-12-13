import pandas as pd
import numpy as np
import spacy
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# --- Configuration ---
SAMPLE_FILE_PATH = '../../data/philosophy_full.csv'
# We use the 'medium' (md) model because it contains the actual vector numbers.
# The 'small' (sm) model does not have vectors.
SPACY_MODEL_NAME = 'en_core_web_md'
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

# --- 1. Setup Device (Apple Silicon) ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Success: Using Apple MPS acceleration!")
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

# --- 3. Feature Extraction (Word Embeddings) ---
# We use spaCy to turn words into numerical vectors.
print(f"\nLoading spaCy model '{SPACY_MODEL_NAME}'...")
try:
    nlp = spacy.load(SPACY_MODEL_NAME)
except OSError:
    print(f"Error: Could not load '{SPACY_MODEL_NAME}'.")
    print(f"Please run: python -m spacy download {SPACY_MODEL_NAME}")
    exit()

print("Generating sentence embeddings (converting text to dense vectors)...")
start_time = time.time()
with nlp.disable_pipes():
    # nlp.pipe processes text efficiently in batches.
    # doc.vector automatically calculates the mean of all word vectors in the sentence.
    # This turns a sentence of any length into a single list of 300 numbers.
    vectors = [doc.vector for doc in nlp.pipe(df['lemmatized_str'].astype(str), batch_size=1000)]

X = np.array(vectors)
print(f"Embedding complete in {time.time() - start_time:.2f} seconds.")

# --- 4. Prepare Labels ---
# Encode labels to integers: "plato" -> 0, "aristotle" -> 1, etc.
y_text = df['school']
encoder = LabelEncoder()
y_int = encoder.fit_transform(y_text)
num_classes = len(encoder.classes_)
print(f"Classes found: {num_classes}")

# --- 5. Prepare PyTorch Tensors ---
# We set the data types: float32 for inputs, long (integers) for labels.

# Split data (80% train, 20% test)
X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
    X, y_int, test_size=0.2, random_state=42, stratify=y_int
)

# Create Datasets
# TensorDataset wraps the inputs and labels together so the DataLoader can access them.
train_data = TensorDataset(torch.tensor(X_train_np, dtype=torch.float32), torch.tensor(y_train_np, dtype=torch.long))
test_data = TensorDataset(torch.tensor(X_test_np, dtype=torch.float32), torch.tensor(y_test_np, dtype=torch.long))

# Create DataLoaders
# DataLoaders handle feeding data to the model in "batches" (e.g., 32 at a time)
# shuffle=True is crucial for training, so the model doesn't memorize the order of data.
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)


# --- 6. Define Neural Network Architecture ---
# In PyTorch, we define a class that holds our model structure.
class PhilosophyNet(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(PhilosophyNet, self).__init__()

        # Layer 1: Input (300) -> Hidden (256)
        # nn.Linear represents weighted connections between neurons.
        self.layer1 = nn.Linear(input_dim, hidden_dim1)

        # Activation: ReLU
        # This allows the model to learn complex, non-linear patterns.
        # Without this, the neural net is just a fancy linear regression.
        self.relu = nn.ReLU()

        # Dropout: Randomly turns off 30% of neurons during training.
        # This prevents the model from "memorizing" the training data (overfitting).
        self.dropout1 = nn.Dropout(0.3)

        # Layer 2: Hidden (256) -> Hidden (128)
        self.layer2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.dropout2 = nn.Dropout(0.3)

        # Output Layer: Hidden (128) -> Output (13 classes)
        # The raw output numbers are called "logits".
        self.output_layer = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        # This function defines how data flows through the network
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        x = self.layer2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        x = self.output_layer(x)
        return x


# Initialize the model and move it to the GPU (if available)
model = PhilosophyNet(input_dim=300, hidden_dim1=256, hidden_dim2=128, output_dim=num_classes)
model.to(device)
print("\nModel Structure:")
print(model)

# --- 7. Loss and Optimizer ---
# CrossEntropyLoss: The standard error metric for multi-class classification.
# It automatically handles the "Softmax" calculation to turn outputs into probabilities.
criterion = nn.CrossEntropyLoss()

# Adam: An adaptive optimizer that figures out how much to adjust the weights.
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 8. Training Loop ---
print("\nStarting training...")
start_train_time = time.time()

for epoch in range(EPOCHS):
    model.train()  # Set model to 'training' mode (enables Dropout)
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        # Move this batch of data to the M2 GPU
        inputs, labels = inputs.to(device), labels.to(device)

        # 1. Clear old gradients (PyTorch accumulates them by default, so we must reset)
        optimizer.zero_grad()

        # 2. Forward Pass: Ask the model for predictions
        outputs = model(inputs)

        # 3. Calculate Loss: How wrong was the model?
        loss = criterion(outputs, labels)

        # 4. Backward Pass: Calculate how to adjust weights to reduce error
        loss.backward()

        # 5. Step: Actually update the weights
        optimizer.step()

        # Track stats for printing
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

print(f"Training finished in {time.time() - start_train_time:.2f} seconds.")

# --- 9. Evaluation ---
print("\nEvaluating on Test Set...")
model.eval()  # Set model to 'eval' mode (disables Dropout for consistent results)
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        # Move results back to CPU to convert to numpy for reporting
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Generate Report
print("\nClassification Report (PyTorch FFNN + Embeddings):")
print(classification_report(all_labels, all_preds, target_names=encoder.classes_, zero_division=0))
