from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass
class Mappings:
    user_to_idx: dict
    idx_to_user: dict
    movie_to_idx: dict
    idx_to_movie: dict
    genre_names: list


def load_raw_data(data_dir: Path = DEFAULT_DATA_DIR):
    movies = pd.read_csv(data_dir / "movies.csv")
    ratings = pd.read_csv(data_dir / "ratings.csv")
    tags = pd.read_csv(data_dir / "tags.csv")
    links = pd.read_csv(data_dir / "links.csv")
    return movies, ratings, tags, links


def multi_hot_encode_genres(movies: pd.DataFrame):
    genres_split = movies["genres"].apply(
        lambda g: [] if g == "(no genres listed)" else g.split("|")
    )
    genre_names = sorted({genre for genres in genres_split for genre in genres})
    genre_matrix = np.zeros((len(movies), len(genre_names)), dtype=np.float32)
    genre_index = {genre: i for i, genre in enumerate(genre_names)}
    for row, genres in enumerate(genres_split):
        for genre in genres:
            genre_matrix[row, genre_index[genre]] = 1.0
    genre_df = pd.DataFrame(genre_matrix, columns=genre_names, index=movies.index)
    movies_encoded = pd.concat([movies, genre_df], axis=1)
    return movies_encoded, genre_names


def build_id_mappings(ratings: pd.DataFrame):
    unique_users = np.sort(ratings["userId"].unique())
    unique_movies = np.sort(ratings["movieId"].unique())

    user_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
    idx_to_user = {idx: user_id for user_id, idx in user_to_idx.items()}

    movie_to_idx = {movie_id: idx for idx, movie_id in enumerate(unique_movies)}
    idx_to_movie = {idx: movie_id for movie_id, idx in movie_to_idx.items()}

    return user_to_idx, idx_to_user, movie_to_idx, idx_to_movie


def merge_datasets(movies: pd.DataFrame, ratings: pd.DataFrame, links: pd.DataFrame):
    merged = ratings.merge(movies, on="movieId", how="left")
    merged = merged.merge(links, on="movieId", how="left")
    return merged


def leave_last_n_out_split(ratings: pd.DataFrame, n: int = 1, min_ratings: int = 3):
    ratings_sorted = ratings.sort_values(["userId", "timestamp"])
    counts = ratings_sorted.groupby("userId")["movieId"].transform("count")
    eligible = counts >= min_ratings

    rank_desc = ratings_sorted.groupby("userId").cumcount(ascending=False)
    is_test = eligible & (rank_desc < n)

    test_df = ratings_sorted[is_test].reset_index(drop=True)
    train_df = ratings_sorted[~is_test].reset_index(drop=True)
    return train_df, test_df


def build_genre_feature_matrix(movies_encoded: pd.DataFrame, genre_names: list, movie_to_idx: dict):
    num_movies = len(movie_to_idx)
    num_genres = len(genre_names)
    feature_matrix = np.zeros((num_movies, num_genres), dtype=np.float32)
    for _, row in movies_encoded.iterrows():
        movie_id = row["movieId"]
        if movie_id in movie_to_idx:
            idx = movie_to_idx[movie_id]
            feature_matrix[idx] = row[genre_names].to_numpy(dtype=np.float32)
    return feature_matrix


def prepare_full_pipeline(data_dir: Path = DEFAULT_DATA_DIR, test_n: int = 1, min_ratings: int = 3):
    movies, ratings, tags, links = load_raw_data(data_dir)
    movies_encoded, genre_names = multi_hot_encode_genres(movies)
    user_to_idx, idx_to_user, movie_to_idx, idx_to_movie = build_id_mappings(ratings)
    train_df, test_df = leave_last_n_out_split(ratings, n=test_n, min_ratings=min_ratings)
    genre_feature_matrix = build_genre_feature_matrix(movies_encoded, genre_names, movie_to_idx)

    mappings = Mappings(
        user_to_idx=user_to_idx,
        idx_to_user=idx_to_user,
        movie_to_idx=movie_to_idx,
        idx_to_movie=idx_to_movie,
        genre_names=genre_names,
    )

    return {
        "movies": movies,
        "movies_encoded": movies_encoded,
        "ratings": ratings,
        "tags": tags,
        "links": links,
        "train_df": train_df,
        "test_df": test_df,
        "genre_feature_matrix": genre_feature_matrix,
        "mappings": mappings,
    }


def get_user_rating_history(ratings: pd.DataFrame, movies: pd.DataFrame, user_id: int):
    user_ratings = ratings[ratings["userId"] == user_id]
    history = user_ratings.merge(movies, on="movieId", how="left")
    history = history.sort_values("timestamp", ascending=False)
    return history[["movieId", "title", "genres", "rating", "timestamp"]]


def get_movie_metadata(movies: pd.DataFrame, ratings: pd.DataFrame, tags: pd.DataFrame, movie_id: int):
    movie_row = movies[movies["movieId"] == movie_id]
    if movie_row.empty:
        return None

    title = movie_row.iloc[0]["title"]
    genres = movie_row.iloc[0]["genres"]

    movie_ratings = ratings[ratings["movieId"] == movie_id]
    avg_rating = movie_ratings["rating"].mean() if not movie_ratings.empty else None
    num_ratings = len(movie_ratings)

    movie_tags = tags[tags["movieId"] == movie_id]["tag"].tolist()

    return {
        "movieId": movie_id,
        "title": title,
        "genres": genres,
        "avg_rating": avg_rating,
        "num_ratings": num_ratings,
        "tags": movie_tags,
    }
