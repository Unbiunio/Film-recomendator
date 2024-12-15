import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import os

def load_data(filepath):
    """
    Load the movie dataset from a CSV file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} not found.")
    return pd.read_csv(filepath)

def clean_data(data):
    """
    Clean the dataset:
    - Remove duplicates.
    - Handle missing values.
    """
    # Drop duplicates
    data = data.drop_duplicates()

    # Handle missing values
    if data.isnull().sum().any():
        # Fill missing 'rating' with the median value
        if 'rating' in data.columns:
            data['rating'] = data['rating'].fillna(data['rating'].median())
        
        # Drop rows with missing values in critical columns
        data = data.dropna(subset=['name', 'user'])

    return data

def encode_theme(data):
    """
    Encode the 'theme' column using One-Hot Encoding.
    """
    if 'theme' in data.columns:
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        theme_encoded = encoder.fit_transform(data[['theme']])
        theme_encoded_df = pd.DataFrame(theme_encoded, columns=encoder.get_feature_names_out(['theme']))
        data = pd.concat([data.reset_index(drop=True), theme_encoded_df], axis=1)
        data = data.drop(columns=['theme'])
    return data

def scale_ratings(data):
    """
    Normalize the 'rating' column to a scale between 0 and 1.
    """
    if 'rating' in data.columns:
        data['rating'] = (data['rating'] - data['rating'].min()) / (data['rating'].max() - data['rating'].min())
    return data

def save_cleaned_data(data, output_path):
    """
    Save the cleaned dataset to a new CSV file.
    """
    data.to_csv(output_path, index=False)

def process_data(input_filepath, output_filepath):
    """
    Full pipeline for processing the movie dataset.
    """
    print("Loading data...")
    data = load_data(input_filepath)

    print("Cleaning data...")
    data = clean_data(data)

    print("Encoding 'theme' column...")
    data = encode_theme(data)

    print("Scaling ratings...")
    data = scale_ratings(data)

    print("Saving cleaned data...")
    save_cleaned_data(data, output_filepath)
    print(f"Cleaned data saved to {output_filepath}")

# Example usage
if __name__ == "__main__":
    input_path = "./data/movies_dataset.csv"
    output_path = "./data/movies_cleaned.csv"
    process_data(input_path, output_path)
