# src/utils/__init__.py

# Import functions from modules to simplify imports in main.py
from .data_processing import movie_encoder, data_split, make_dataloader
from .dataset import MovieLensDataset
from .model import CFModel
from .train_model import model_train
from .test_model import model_test
from .recommendation import recommend_movies