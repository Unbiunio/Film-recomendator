import pandas as pd
import os
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import pickle

def load_cleaned_data(filepath):
    """
    Load the cleaned movie dataset from a CSV file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} not found.")
    return pd.read_csv(filepath)

def prepare_surprise_data(data):
    """
    Convert the dataset into a format compatible with Surprise library.
    """
    reader = Reader(rating_scale=(0, 1))  # Assuming 'rating' is normalized between 0 and 1
    surprise_data = Dataset.load_from_df(data[['user', 'name', 'rating']], reader)
    return surprise_data

def train_model(surprise_data):
    """
    Train an SVD model (collaborative filtering) using the Surprise library.
    """
    print("Splitting data into train and test sets...")
    trainset, testset = train_test_split(surprise_data, test_size=0.2, random_state=42)
    
    print("Training the SVD model...")
    model = SVD()
    model.fit(trainset)
    
    print("Model training complete!")
    return model

def save_model(model, output_path):
    """
    Save the trained model to a file using pickle.
    """
    with open(output_path, 'wb') as model_file:
        pickle.dump(model, model_file)
    print(f"Model saved to {output_path}")

def train_and_save_model(input_filepath, model_output_path):
    """
    Full pipeline for training and saving the recommendation model.
    """
    print("Loading cleaned data...")
    data = load_cleaned_data(input_filepath)
    
    print("Preparing data for Surprise...")
    surprise_data = prepare_surprise_data(data)
    
    print("Training recommendation model...")
    model = train_model(surprise_data)
    
    print("Saving trained model...")
    save_model(model, model_output_path)

# Example usage
if __name__ == "__main__":
    input_path = "./data/movies_cleaned.csv"
    model_output_path = "./models/model.pkl"
    train_and_save_model(input_path, model_output_path)
