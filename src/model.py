import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset


class RatingsDataset(Dataset):
    def __init__(self, df, user_to_idx, movie_to_idx, genre_feature_matrix):
        self.user_indices = df["userId"].map(user_to_idx).to_numpy(dtype=np.int64)
        self.movie_indices = df["movieId"].map(movie_to_idx).to_numpy(dtype=np.int64)
        self.ratings = df["rating"].to_numpy(dtype=np.float32)
        self.genre_feature_matrix = genre_feature_matrix

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        user_idx = self.user_indices[idx]
        movie_idx = self.movie_indices[idx]
        genre_features = self.genre_feature_matrix[movie_idx]
        rating = self.ratings[idx]
        return (
            torch.tensor(user_idx, dtype=torch.long),
            torch.tensor(movie_idx, dtype=torch.long),
            torch.tensor(genre_features, dtype=torch.float32),
            torch.tensor(rating, dtype=torch.float32),
        )


class NCFModel(nn.Module):
    def __init__(self, num_users, num_movies, num_genres, embed_dim=32):
        super().__init__()

        self.user_embed  = nn.Embedding(num_users, embed_dim)
        self.movie_embed = nn.Embedding(num_movies, embed_dim)

        self.fc1 = nn.Linear(embed_dim * 2 + num_genres, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.out = nn.Linear(16, 1)

    def forward(self, user_ids, movie_ids, genre_flags):
        user_vec  = self.user_embed(user_ids)
        movie_vec = self.movie_embed(movie_ids)

        x = torch.cat([user_vec, movie_vec, genre_flags], dim=1)

        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)

        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)

        x = self.fc3(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        return self.out(x).squeeze(1)

    def get_movie_vector(self, movie_ids):
        with torch.no_grad():
            return self.movie_embed(movie_ids)
