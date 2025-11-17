import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction import text
import time
import os

# --- Configuration ---
SAMPLE_FILE_PATH = 'data/philosophy_sample_50k.csv'
# Add 'pron' to the default English stop word list
my_stop_words = list(text.ENGLISH_STOP_WORDS) + ['pron']

# --- 1. Load Data ---
print(f"Loading sample data from: {SAMPLE_FILE_PATH}")
if not os.path.exists(SAMPLE_FILE_PATH):
    print(f"Error: Sample file not found at {SAMPLE_FILE_PATH}")
    exit()

try:
    df = pd.read_csv(SAMPLE_FILE_PATH)
    df = df.dropna()
    print(f"Successfully loaded {len(df)} rows.")
except Exception as e:
    print(f"An error occurred loading the CSV: {e}")
    exit()

# --- 2. Define Features (X) and Target (y) ---
X = df['lemmatized_str']
y = df['school']
print(f"Using {len(df)} samples for training and testing.")

# --- 3. Preprocessing & Splitting ---
print("\nStarting TF-IDF Vectorization (with 'pron' removed)...")
start_time = time.time()

# --- UPDATED VECTORIZER ---
# Now using our custom stop word list
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words=my_stop_words
)

X_tfidf = tfidf_vectorizer.fit_transform(X)
print(
    f"TF-IDF vectorization complete in {time.time() - start_time:.2f} seconds.")

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)
print(
    f"Data split into {X_train.shape[0]} training and {X_test.shape[0]} test samples.")


# --- 4. Model Training & Evaluation ---
def train_and_evaluate(model, model_name):
    """Helper function to train and report on a model."""
    print(f"\n--- Training {model_name} ---")
    start_time = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    end_time = time.time()

    print(f"{model_name} training complete in {end_time - start_time:.2f} seconds.")
    print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification Report:")
    # We use zero_division=0 to prevent warnings for classes with no predictions
    print(classification_report(y_test, y_pred, zero_division=0))


# Model 1: Decision Tree
dt_classifier = DecisionTreeClassifier(random_state=42)
train_and_evaluate(dt_classifier, "Decision Tree")

# Model 2: Random Forest
rf_classifier = RandomForestClassifier(
    n_estimators=100, random_state=42, n_jobs=-1)
train_and_evaluate(rf_classifier, "Random Forest")

# Model 3: Logistic Regression
# We use 'saga' solver as it's good for large datasets, and max_iter=1000 to ensure convergence
lr_classifier = LogisticRegression(
    solver='saga', max_iter=1000, random_state=42, n_jobs=-1)
train_and_evaluate(lr_classifier, "Logistic Regression")

# Model 4: Linear Support Vector Machine (SVM)
# LinearSVC is often the best and fastest for text classification
svm_classifier = LinearSVC(random_state=42)
train_and_evaluate(svm_classifier, "Linear SVM (LinearSVC)")

print("\n--- Model Comparison Complete ---")
