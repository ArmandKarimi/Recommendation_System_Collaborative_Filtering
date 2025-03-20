# src/main.py
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd 
import numpy as np
import logging
import sys
import os
import json
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE

# Import necessary modules
from src.utils import movie_encoder, data_split, make_dataloader, CFModel, model_train, model_test, recommend_movies
from src.utils.dataset import MovieLensDataset

# import params
from src.config import DEVICE, EPOCHS, N_FACTORS, FRACTION

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG if needed
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

#import data
df_ratings = pd.read_csv("src/data/rating.csv")
df_movies = pd.read_csv("src/data/movie.csv")

ratings = df_ratings.sample(frac = FRACTION)
logger.info(f"✅ 10,000 samples of the Data Fetched")

ratings = movie_encoder(ratings)
logger.info(f"✅ Movie titles encoded")

train_data, val_data, test_data = data_split(ratings)
logger.info(f"✅ Data splitted")

train_loader = make_dataloader(train_data)
val_loader = make_dataloader(val_data)
test_loader = make_dataloader(test_data)
logger.info(f"✅ Dataloaders created")

#create vecotr space
n_users = len(np.unique(ratings.user))
n_movies = len(np.unique(ratings.movie))
n_factors = 5 #dimension of the vector space

# instantite the model
model = CFModel(n_users, n_movies, N_FACTORS)
logger.info(f"Collaborative Filtering Model Instantiated")

# Optimizer and Loss
LOSS_FN = nn.MSELoss()
OPTIMIZER = optim.Adam(model.parameters(), lr=0.01)


# Train model
logger.info(f"🚀🚀🚀 Training for {EPOCHS} epochs...🚀🚀🚀")
model = model_train(model, train_loader, val_loader, loss_fn = LOSS_FN, optimizer = OPTIMIZER, device = DEVICE, epochs = EPOCHS)
logger.info("✅ Training complete.")

# --- Evaluation ---
logger.info("Evaluating model...")
rmse = model_test(model, test_loader, loss_fn = LOSS_FN, device = DEVICE)
logger.info("✅ Evaluation complete.")
logger.info(f" RMSE of the model is {np.round(rmse,3)}")


# Recommand for a given user 
logger.info(f"Choose a userID from 1 to {max(ratings.user.values)}")
user_id = int(input("Enter User ID: "))
logger.info(f"User ID entered: {user_id}")


# Recommand Movies
all_movie_ids = np.unique(df_ratings['movieId'].values)
# Suppose we have a set of movies that user_id already rated:
rated_movie_ids = df_ratings[(df_ratings.userId == user_id)]['movieId'].values
#use the mode to recommand movies
recs = recommend_movies(model, user_id, all_movie_ids, rated_movie_ids, top_n=5, device=DEVICE)
logger.info(f"Top recommendations for user {user_id} : {recs}")

recommended_movie_ids = [movie_id for movie_id, _ in recs]

if recommended_movie_ids:
    filtered_movies = df_movies[df_movies.movieId.isin(recommended_movie_ids)]
    print(filtered_movies)  # Display recommended movies
else:
    print("No recommendations available.")
