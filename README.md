# Classification of Philosophical Schools Using Text-Based Models

The central goal of this project is to investigate whether different schools of philosophical thought possess distinct linguistic patterns that can be identified and classified by machine learning algorithms. We aim to answer the question: Can a model accurately predict the philosophical school (e.g., Rationalism, Empiricism, Stoicism) to which a sentence belongs, based solely on its text?

## Usage Instructions

## Data Source

This project uses the [History of Philosophy](https://www.kaggle.com/datasets/kouroshalizadeh/history-of-philosophy) dataset from Kaggle as the primary data source. Please download the dataset and place it in the `data/` directory before running the code.

## 01 Data Preprocessing

From the EDA of the original full dataset, we learned that:

- No missing values exist in the dataset.
- The classes are not evenly distributed. "Analytic" has 55k entries, while "Stoicism" only has 2.5k. Our models will be much better at predicting the common classes. When we look at the results, we must check the F1-score for "Stoicism" to see if the model is just ignoring it.
- The average sentence is 151 characters, but the max is 2,649. This is a wide range, but TF-IDF is generally robust to this.

## 02 Model Training

This very first step involves basic model training using TF-IDF vectorization and two classifiers: Decision Tree and Random Forest. This serves as a baseline for more complex models.

1. Decision Tree (Accuracy: 36.8%):

    - This is our low-end baseline. With 13 classes, random guessing would be ~7.7% (1/13), so 37% is significantly better than chance.
    - However, the F1-scores are low across the board (e.g., stoicism: 0.10), which confirms this model is struggling.

2. Random Forest (Accuracy: 50.6%):

    - We got a ~14% jump in accuracy by moving from one tree to 100 trees. It seems that a single tree is overfitting, while the ensemble generalizes better.
    - We have imbalanced classes. 'stoicism' has a precision of 0.60, but a recall of 0.09. This means that when the model predicts 'stoicism', it's usually right, but it misses most of the actual 'stoicism' sentences.
    - From the feature importance results, the top feature is 'pron' (0.028408). This is an artifact from the lemmatizer — it's the token for "pronoun". It is noise and not a real word. We need to remove it.
    - Other top features are exactly what we'd hope to see: 'woman' (likely from "feminism"), 'god', 'man', 'idea', 'madness' (likely "nietzsche" or "continental"). This proves our core concept is working.
