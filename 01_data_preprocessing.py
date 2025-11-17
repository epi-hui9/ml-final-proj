"""
This script loads the columns needed from the full dataset, performs EDA, and creates a
stratified sample.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import os

# --- Configuration ---
# Path to the full dataset CSV
SOURCE_FILE_PATH = 'data/philosophy_data.csv'

# Columns we want to load for EDA
# We load 'author' and 'sentence_length' for EDA
# We load 'lemmatized_str' and 'school' for the sample
COLUMNS_TO_LOAD = ['lemmatized_str', 'school', 'author', 'sentence_length']

# The size of the sample we want to create
SAMPLE_SIZE = 50000

# The name of the new sample file this script will create
SAMPLE_FILE_PATH = 'data/philosophy_sample_50k.csv'


def load_data(file_path, use_cols):
    """
    Loads the specified columns from the CSV file.
    """
    print(f"Attempting to load data from: {file_path}")
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        print("Please make sure to download the dataset from Kaggle and place it in the 'data/' directory.")
        return None

    try:
        df = pd.read_csv(file_path, usecols=use_cols)
        print(
            f"Successfully loaded {len(df)} rows and {len(df.columns)} columns.")
        return df
    except ValueError as e:
        print(f"Error loading CSV. Are the columns correct? {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during loading: {e}")
        return None


def perform_eda(df):
    """
    Prints a simple Exploratory Data Analysis (EDA) report.
    """
    print("\n--- Exploratory Data Analysis (Full Dataset) ---")

    # 1. Basic Info
    print("\n[1. Basic Info]")
    df.info(memory_usage='deep')

    # 2. Missing Values
    print("\n[2. Missing Values]")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])

    # 3. Target Distribution (School)
    print("\n[3. Target Distribution ('school')]")
    print("This shows our main classes. We'll need to clean/dropna.")
    print(df['school'].value_counts(dropna=False).head(15))

    # 4. Author Distribution
    print("\n[4. Author Distribution ('author')]")
    print(df['author'].value_counts(dropna=False).head(15))

    # 5. Sentence Length Statistics
    print("\n[5. Sentence Length Statistics ('sentence_length')]")
    print(df['sentence_length'].describe())
    print("--------------------------------------------------")


def create_sample(df, sample_size, output_path):
    """
    Cleans the data and creates a stratified sample, saving it to a new CSV.
    """
    print(f"\nCreating sample of {sample_size} rows...")

    # 1. Clean data: We can only model rows that have both a feature and a label.
    df_clean = df.dropna(subset=['lemmatized_str', 'school']).copy()
    print(f"Removed missing values. Usable rows: {len(df_clean)}")

    # 2. Handle rare classes for stratification
    # 'stratify' requires at least 2 members per class.
    class_counts = df_clean['school'].value_counts()
    classes_to_keep = class_counts[class_counts > 1].index
    df_stratify = df_clean[df_clean['school'].isin(classes_to_keep)]
    print(
        f"Removed single-instance classes. Rows for sampling: {len(df_stratify)}")

    # 3. Create the stratified sample
    try:
        sample_df, _ = train_test_split(
            df_stratify,
            train_size=sample_size,
            stratify=df_stratify['school'],
            random_state=42
        )
    except ValueError as e:
        print(f"Error during stratification. {e}")
        print("This can happen if sample_size is larger than available data.")
        print("Trying to create sample with a smaller size if possible.")
        if len(df_stratify) < sample_size:
            sample_size = len(df_stratify)

        sample_df, _ = train_test_split(
            df_stratify,
            train_size=sample_size,
            stratify=df_stratify['school'],
            random_state=42
        )

    print(f"Successfully created stratified sample of {len(sample_df)} rows.")

    # 4. Filter to *only* columns needed for modeling
    final_sample = sample_df[['lemmatized_str', 'school']]

    # 5. Save the sample to a new CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_sample.to_csv(output_path, index=False)
    print(f"--- SUCCESS! ---")
    print(f"Sample file saved to: {output_path}")


def main():
    df = load_data(SOURCE_FILE_PATH, COLUMNS_TO_LOAD)

    if df is not None:
        perform_eda(df)
        create_sample(df, SAMPLE_SIZE, SAMPLE_FILE_PATH)


if __name__ == "__main__":
    main()
