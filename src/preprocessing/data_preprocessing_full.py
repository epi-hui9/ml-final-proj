"""
This script loads the columns needed from the full dataset, performs EDA,
cleans missing values, and saves the full processed dataset without sampling.
"""

import pandas as pd
import os

# --- Configuration ---
# Path to the full dataset CSV
SOURCE_FILE_PATH = '../../data/philosophy_data.csv'

# Columns we want to load for EDA
# We load 'author' and 'sentence_length' for EDA
# We load 'lemmatized_str' and 'school' for the final output
COLUMNS_TO_LOAD = ['lemmatized_str', 'school', 'author', 'sentence_length']

# The name of the new full processed file this script will create
OUTPUT_FILE_PATH = '../../data/philosophy_full.csv'


def load_data(file_path, use_cols):
    """
    Loads the specified columns from the CSV file.
    """
    print(f"Attempting to load data from: {file_path}")
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        print("Please make sure the dataset is in the 'data/' directory.")
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


def process_full_data(df, output_path):
    """
    Cleans the data (removes NaNs) and saves the full dataset to a new CSV.
    """
    print(f"\nProcessing full dataset...")

    # 1. Clean data: We can only model rows that have both a feature and a label.
    # We drop rows where either 'lemmatized_str' or 'school' is missing.
    df_clean = df.dropna(subset=['lemmatized_str', 'school']).copy()
    print(f"Removed missing values. Final row count: {len(df_clean)}")

    # 2. Filter to *only* columns needed for modeling
    final_df = df_clean[['lemmatized_str', 'school']]

    # 3. Save the full cleaned dataset to a new CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)
    print(f"--- SUCCESS! ---")
    print(f"Full processed file saved to: {output_path}")


def main():
    df = load_data(SOURCE_FILE_PATH, COLUMNS_TO_LOAD)

    if df is not None:
        perform_eda(df)
        process_full_data(df, OUTPUT_FILE_PATH)


if __name__ == "__main__":
    main()
