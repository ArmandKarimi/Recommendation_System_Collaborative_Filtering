import pandas as pd 
import numpy as np 
import logging
import os 
import sys

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from .dataset import MovieLensDataset


# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG if needed
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def movie_encoder(ratings): 
    # Encode user and movie IDs as categorical indices
    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()

    ratings['user'] = user_encoder.fit_transform(ratings['userId'])  # Convert userId to index
    ratings['movie'] = movie_encoder.fit_transform(ratings['movieId'])  # Convert movieId to index

    # Drop timestamp since it's not needed
    ratings = ratings[['user', 'movie', 'rating']]

    return ratings

def data_split(ratings):
    """
    split data into train, val and test datasets
    """
    # Split into training+validation and test sets first
    train_val_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

    # Then split train+validation into training and validation sets (e.g., 80/20 of the remaining)
    train_data, val_data = train_test_split(train_val_data, test_size=0.25, random_state=42)  

    return train_data, val_data, test_data



def make_dataloader(df):
    dataset = MovieLensDataset(df)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    return data_loader



if __name__ == "__main__":
    #import data
    ratings = pd.read_csv("../data/rating.csv")
    ratings = ratings.sample(n = 10000)
    logger.info(f"✅ 10,000 samples of the Data Fetched")

    ratings = movie_encoder(ratings)
    logger.info(f"✅ Movie titles encoded")

    train_data, val_data, test_data = data_split(ratings)
    logger.info(f"✅ Data splitted")

    train_dataloader = make_dataloader(train_data)
    val_dataloader = make_dataloader(val_data)
    test_dataloader = make_dataloader(test_data)
    logger.info(f"✅ Dataloaders created")
    



