import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction import text
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns

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

# --- 2. Preprocessing & Splitting ---
X = df['lemmatized_str']
y = df['school']

print("\nStarting TF-IDF Vectorization...")
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000, 
    stop_words=my_stop_words
)
X_tfidf = tfidf_vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Data split into {X_train.shape[0]} training and {X_test.shape[0]} test samples.")

# --- 3. Train Our Best Model ---
print("\n--- Training Best Model: Linear SVM (LinearSVC) ---")
start_time = time.time()
svm_classifier = LinearSVC(random_state=42)
svm_classifier.fit(X_train, y_train)
y_pred = svm_classifier.predict(X_test)
print(f"Training complete in {time.time() - start_time:.2f} seconds.")

# --- 4. Show Top Words per Class ---
print("\n--- Top 10 Features (Words) per School ---")
feature_names = tfidf_vectorizer.get_feature_names_out()
class_labels = svm_classifier.classes_

for i, class_label in enumerate(class_labels):
    # Get coefficients for this class
    coef = svm_classifier.coef_[i]
    # Sort coefficients by value
    top_coef_indices = coef.argsort()[-10:][::-1] # Top 10
    
    top_words = [feature_names[idx] for idx in top_coef_indices]
    
    print(f"[{class_label.upper()}]: {', '.join(top_words)}")

# --- 5. Generate and Save Confusion Matrix ---
print("\n--- Generating Confusion Matrix ---")
try:
    cm = confusion_matrix(y_test, y_pred, labels=class_labels)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=class_labels, 
        yticklabels=class_labels
    )
    plt.title('Confusion Matrix - Linear SVM (LinearSVC)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save the figure
    output_filename = 'plots/confusion_matrix_svm.png'
    plt.savefig(output_filename)
    
    print(f"\n--- SUCCESS! ---")
    print(f"Confusion matrix saved to: {output_filename}")
    print("This plot shows which classes are being confused with each other.")
    print("The diagonal (top-left to bottom-right) shows correct predictions.")

except Exception as e:
    print(f"Error generating plot. Do you have matplotlib and seaborn installed?")
    print(f"You can install them with: pip install matplotlib seaborn")
    print(f"Error: {e}")