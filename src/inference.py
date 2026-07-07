import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

from data_utils import Mappings
from model import NCFModel


MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def load_trained_artifacts(models_dir: Path = MODELS_DIR, device="cpu"):
    with open(models_dir / "mappings.json", "r") as f:
        payload = json.load(f)

    user_to_idx = {int(k): v for k, v in payload["user_to_idx"].items()}
    idx_to_user = {int(k): v for k, v in payload["idx_to_user"].items()}
    movie_to_idx = {int(k): v for k, v in payload["movie_to_idx"].items()}
    idx_to_movie = {int(k): v for k, v in payload["idx_to_movie"].items()}
    genre_names = payload["genre_names"]
    embed_dim = payload["embed_dim"]

    mappings = Mappings(
        user_to_idx=user_to_idx,
        idx_to_user=idx_to_user,
        movie_to_idx=movie_to_idx,
        idx_to_movie=idx_to_movie,
        genre_names=genre_names,
    )

    genre_feature_matrix = np.load(models_dir / "genre_feature_matrix.npy")

    model = NCFModel(
        num_users=len(user_to_idx),
        num_movies=len(movie_to_idx),
        num_genres=len(genre_names),
        embed_dim=embed_dim,
    )
    model.load_state_dict(torch.load(models_dir / "ncf_model.pt", map_location=device))
    model.to(device)
    model.eval()

    return model, mappings, genre_feature_matrix


def get_user_seen_movie_ids(ratings_df, user_id):
    return set(ratings_df[ratings_df["userId"] == user_id]["movieId"].tolist())


def score_all_unseen_items(user_id, model, mappings, genre_feature_matrix, seen_movie_ids, device="cpu"):
    if user_id not in mappings.user_to_idx:
        return []

    user_idx = mappings.user_to_idx[user_id]
    num_movies = len(mappings.movie_to_idx)

    candidate_movie_indices = [
        idx for idx in range(num_movies)
        if mappings.idx_to_movie[idx] not in seen_movie_ids
    ]

    if not candidate_movie_indices:
        return []

    user_indices_tensor = torch.full((len(candidate_movie_indices),), user_idx, dtype=torch.long, device=device)
    movie_indices_tensor = torch.tensor(candidate_movie_indices, dtype=torch.long, device=device)
    genre_features_tensor = torch.tensor(
        genre_feature_matrix[candidate_movie_indices], dtype=torch.float32, device=device
    )

    with torch.no_grad():
        predictions = model(user_indices_tensor, movie_indices_tensor, genre_features_tensor)

    predictions = predictions.cpu().numpy()

    scored = [
        {"movieId": mappings.idx_to_movie[idx], "predicted_rating": float(score)}
        for idx, score in zip(candidate_movie_indices, predictions)
    ]
    scored.sort(key=lambda item: item["predicted_rating"], reverse=True)
    return scored


def get_top_n_recommendations(
    user_id,
    ratings_df,
    model,
    mappings,
    genre_feature_matrix,
    top_n=10,
    device="cpu",
):
    seen_movie_ids = get_user_seen_movie_ids(ratings_df, user_id)
    scored = score_all_unseen_items(user_id, model, mappings, genre_feature_matrix, seen_movie_ids, device=device)
    return scored[:top_n]


def build_item_feature_vectors(model, genre_feature_matrix, device="cpu"):
    model.eval()
    all_movie_indices = torch.arange(genre_feature_matrix.shape[0], dtype=torch.long, device=device)
    item_embeddings = model.get_movie_vector(all_movie_indices).cpu().numpy()

    normalized_embeddings = item_embeddings / (np.linalg.norm(item_embeddings, axis=1, keepdims=True) + 1e-8)
    normalized_genres = genre_feature_matrix / (np.linalg.norm(genre_feature_matrix, axis=1, keepdims=True) + 1e-8)

    combined = np.concatenate([normalized_embeddings, normalized_genres], axis=1)
    return combined


def get_similar_movies(
    movie_id,
    model,
    mappings,
    genre_feature_matrix,
    top_n=10,
    device="cpu",
):
    if movie_id not in mappings.movie_to_idx:
        return []

    movie_idx = mappings.movie_to_idx[movie_id]
    item_features = build_item_feature_vectors(model, genre_feature_matrix, device=device)

    target_vector = item_features[movie_idx].reshape(1, -1)
    similarities = cosine_similarity(target_vector, item_features)[0]

    ranked_indices = np.argsort(-similarities)
    ranked_indices = ranked_indices[ranked_indices != movie_idx]

    results = []
    for idx in ranked_indices[:top_n]:
        results.append({
            "movieId": mappings.idx_to_movie[int(idx)],
            "score": float(similarities[idx]),
        })
    return results
