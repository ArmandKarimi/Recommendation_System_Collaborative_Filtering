import pandas as pd 
import numpy as np 
import torch 


def recommend_movies(model, user_id, all_movie_ids, rated_movie_ids=set(), top_n=10, device="cpu"):
    """
    Recommend top_n movies for a given user_id.

    Parameters:
      model (nn.Module): The trained recommendation model.
      user_id (int): The ID of the user for whom to recommend movies.
      all_movie_ids (list or array): A list/array of all movie IDs.
      rated_movie_ids (set): A set of movie IDs already rated by the user (to filter out).
      top_n (int): The number of recommendations to return.
      device (str): Device to run the model on.
      
    Returns:
      List of tuples: Each tuple contains (movie_id, predicted_rating).
    """
    # Filter out movies the user has already rated
    candidate_movie_ids = [m for m in all_movie_ids if m not in rated_movie_ids]
    
    # Convert candidate list to tensor and create user tensor
    candidate_tensor = torch.tensor(candidate_movie_ids, dtype=torch.long, device=device)
    user_tensor = torch.tensor([user_id] * len(candidate_movie_ids), dtype=torch.long, device=device)
    
    # Get predictions from the model
    model.eval()
    with torch.no_grad():
        predictions = model(user_tensor, candidate_tensor)
    
    # Convert predictions to numpy array for sorting
    predictions_np = predictions.cpu().numpy()
    
    # Get indices for top_n highest predictions
    top_indices = predictions_np.argsort()[::-1][:top_n]
    
    # Prepare the list of recommendations: (movie_id, predicted_rating)
    recommendations = [(candidate_movie_ids[i], predictions_np[i]) for i in top_indices]
    return recommendations
