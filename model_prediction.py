import pandas as pd
import pickle
import json
import os

def load_model(model_filepath):
    """
    Load the trained model from a pickle file.
    """
    if not os.path.exists(model_filepath):
        raise FileNotFoundError(f"{model_filepath} not found.")
    with open(model_filepath, 'rb') as model_file:
        model = pickle.load(model_file)
    print("Model loaded successfully!")
    return model

def load_cleaned_data(filepath):
    """
    Load the cleaned movie dataset from a CSV file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} not found.")
    return pd.read_csv(filepath)

def generate_recommendations(model, data, top_n=3):
    """
    Generate movie recommendations for each user.
    """
    print("Generating recommendations...")
    # Get unique users and movies
    unique_users = data['user'].unique()
    unique_movies = data['name'].unique()

    # Create a dictionary to store recommendations
    recommendations = {}

    for user in unique_users:
        # Predict ratings for all movies the user hasn't rated yet
        user_recommendations = []
        for movie in unique_movies:
            # Check if the user has already rated the movie
            if not ((data['user'] == user) & (data['name'] == movie)).any():
                # Predict the rating for the user-movie pair
                predicted_rating = model.predict(user, movie).est
                user_recommendations.append((movie, predicted_rating))

        # Sort movies by predicted ratings in descending order
        user_recommendations = sorted(user_recommendations, key=lambda x: x[1], reverse=True)
        
        # Select the top N movies
        recommendations[user] = [movie for movie, _ in user_recommendations[:top_n]]

    print("Recommendations generated successfully!")
    return recommendations

def save_recommendations(recommendations, output_filepath):
    """
    Save the recommendations as a JSON file.
    """
    with open(output_filepath, 'w') as json_file:
        json.dump({"target": recommendations}, json_file, indent=4)
    print(f"Recommendations saved to {output_filepath}")

def recommend_and_save(input_data_filepath, model_filepath, output_filepath):
    """
    Full pipeline for generating and saving movie recommendations.
    """
    print("Loading model...")
    model = load_model(model_filepath)
    
    print("Loading cleaned data...")
    data = load_cleaned_data(input_data_filepath)
    
    print("Generating recommendations...")
    recommendations = generate_recommendations(model, data)
    
    print("Saving recommendations...")
    save_recommendations(recommendations, output_filepath)

# Example usage
if __name__ == "__main__":
    input_data_path = "./data/movies_cleaned.csv"
    model_path = "./models/model.pkl"
    output_path = "./predictions/predictions.json"
    recommend_and_save(input_data_path, model_path, output_path)
