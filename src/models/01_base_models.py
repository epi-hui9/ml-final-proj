import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import time
import os

# --- Configuration ---
# Path to our new sample file
SAMPLE_FILE_PATH = 'data/philosophy_sample_50k.csv'

# --- 1. Load Data ---
print(f"Loading sample data from: {SAMPLE_FILE_PATH}")
if not os.path.exists(SAMPLE_FILE_PATH):
    print(f"Error: Sample file not found at {SAMPLE_FILE_PATH}")
    print("Please make sure '01_data_preprocessing.py' ran successfully.")
    exit()

try:
    df = pd.read_csv(SAMPLE_FILE_PATH)
    print(f"Successfully loaded {len(df)} rows from sample.")
except Exception as e:
    print(f"An error occurred loading the CSV: {e}")
    exit()

# Drop any rows with missing values (though our EDA showed none)
df = df.dropna()

# --- 2. Define Features (X) and Target (y) ---
X = df['lemmatized_str']
y = df['school']
print(f"Using {len(df)} samples for training and testing.")

# --- 3. Preprocessing & Splitting ---
print("\nStarting TF-IDF Vectorization...")
start_time = time.time()

# We'll limit features to the top 5000 to keep it manageable
# and remove common English stop words.
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

# Fit the vectorizer and transform the text into numerical features
X_tfidf = tfidf_vectorizer.fit_transform(X)

print(
    f"TF-IDF vectorization complete in {time.time() - start_time:.2f} seconds.")
print(f"Feature matrix shape: {X_tfidf.shape}")

# Split data into training and testing sets (80% train, 20% test)
# We use stratify=y to ensure the class imbalance is preserved in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)
print(
    f"Data split into {X_train.shape[0]} training and {X_test.shape[0]} test samples.")

# --- 4. Model 1: Decision Tree ---
print("\n--- Training Decision Tree ---")
start_time = time.time()
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)
y_pred_dt = dt_classifier.predict(X_test)
print(
    f"Decision Tree training complete in {time.time() - start_time:.2f} seconds.")

print("\nDecision Tree - Evaluation:")
print(f"Overall Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_dt))

# --- 5. Model 2: Random Forest ---
print("\n--- Training Random Forest ---")
start_time = time.time()
# n_jobs=-1 uses all available CPU cores
rf_classifier = RandomForestClassifier(
    n_estimators=100, random_state=42, n_jobs=-1)
rf_classifier.fit(X_train, y_train)
y_pred_rf = rf_classifier.predict(X_test)
print(
    f"Random Forest training complete in {time.time() - start_time:.2f} seconds.")

print("\nRandom Forest - Evaluation:")
print(f"Overall Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_rf))

# --- 6. Feature Importance (from Random Forest) ---
print("\n--- Top 10 Most Important Features (Words) ---")
try:
    importances = rf_classifier.feature_importances_
    feature_names = tfidf_vectorizer.get_feature_names_out()
    feature_importance_df = pd.DataFrame(
        {'feature': feature_names, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values(
        by='importance', ascending=False)

    print(feature_importance_df.head(10))
except Exception as e:
    print(f"Could not calculate feature importance: {e}")
