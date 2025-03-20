import torch
from torch.utils.data import Dataset

class MovieLensDataset(Dataset):
    # Initialize the data objects
    def __init__(self, data):
        self.users = torch.tensor(data['user'].values, dtype=torch.long)    # integers
        self.movies = torch.tensor(data['movie'].values, dtype=torch.long)   # integers
        self.ratings = torch.tensor(data['rating'].values, dtype=torch.float32)

    # Return the total number of samples
    def __len__(self):
        return len(self.ratings)

    # Get a single sample for a given index
    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]