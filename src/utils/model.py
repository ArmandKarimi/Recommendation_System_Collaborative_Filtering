import torch.nn as nn 

class CFModel(nn.Module):
    def __init__(self, n_users, n_movies, n_factors):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, n_factors)
        self.movie_embedding = nn.Embedding(n_movies, n_factors)

    def forward(self, users, movies): #data objects data[['user', 'movie', 'rating']]
        user_vecs = self.user_embedding(users)   # (batch_size, n_factors)
        movie_vecs = self.movie_embedding(movies)  # (batch_size, n_factors)
        preds = (user_vecs * movie_vecs).sum(dim=1)
        return preds