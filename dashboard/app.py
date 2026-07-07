import sys
from pathlib import Path

import pandas as pd
import streamlit as st

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.append(str(SRC_DIR))

from data_utils import load_raw_data, get_user_rating_history, get_movie_metadata
from inference import load_trained_artifacts, get_top_n_recommendations, get_similar_movies


st.set_page_config(
    page_title="MovieLens Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.08);
        border-radius: 8px;
        padding: 12px 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model_artifacts():
    return load_trained_artifacts()


@st.cache_data
def load_data():
    return load_raw_data()


def render_genre_badges(genres_str):
    genres = [g for g in genres_str.split("|") if g and g != "(no genres listed)"]
    if not genres:
        st.caption("No genres listed")
        return
    cols = st.columns(len(genres))
    for col, genre in zip(cols, genres):
        with col:
            st.badge(genre, color="blue")


def render_user_page(model, mappings, genre_feature_matrix, movies, ratings):
    st.header("🧑 User Recommendations")

    unique_users = sorted(ratings["userId"].unique().tolist())
    selected_user = st.selectbox("Select a user", unique_users, key="user_select")

    history = get_user_rating_history(ratings, movies, selected_user)

    col1, col2 = st.columns(2)
    col1.metric("Movies rated", len(history))
    col2.metric("Average rating given", f"{history['rating'].mean():.2f}" if not history.empty else "N/A")

    with st.expander(f"Rating history for user {selected_user}", expanded=False):
        display_history = history.copy()
        display_history["timestamp"] = pd.to_datetime(display_history["timestamp"], unit="s")
        st.dataframe(
            display_history.rename(columns={
                "movieId": "Movie ID", "title": "Title", "genres": "Genres",
                "rating": "Rating", "timestamp": "Rated on",
            }),
            width="stretch",
            hide_index=True,
        )

    st.subheader("Top Recommendations")
    top_n = st.slider("Number of recommendations", min_value=5, max_value=50, value=10, key="user_top_n")

    recs = get_top_n_recommendations(
        selected_user, ratings, model, mappings, genre_feature_matrix, top_n=top_n
    )

    recs_df = pd.DataFrame(recs)
    if recs_df.empty:
        st.info("No recommendations available for this user.")
        return

    recs_df = recs_df.merge(movies, on="movieId", how="left")
    recs_df = recs_df[["title", "genres", "predicted_rating"]]
    recs_df.insert(0, "Rank", range(1, len(recs_df) + 1))
    recs_df.columns = ["Rank", "Title", "Genres", "Predicted Rating"]

    st.dataframe(
        recs_df,
        width="stretch",
        hide_index=True,
        column_config={
            "Predicted Rating": st.column_config.NumberColumn(format="%.2f ⭐"),
        },
    )


def render_item_page(model, mappings, genre_feature_matrix, movies, ratings, tags):
    st.header("🎬 Item Similarity")

    search_query = st.text_input("Search movie by title", key="movie_search", placeholder="e.g. Harry Potter")
    matching_movies = (
        movies[movies["title"].str.contains(search_query, case=False, na=False, regex=False)]
        if search_query
        else movies
    )
    matching_movies = matching_movies.sort_values("title")

    if matching_movies.empty:
        st.warning("No movies match your search.")
        return

    title_options = [f"{row.title} (id {row.movieId})" for row in matching_movies.itertuples()]
    title_to_movie_id = {
        option: movie_id for option, movie_id in zip(title_options, matching_movies["movieId"].tolist())
    }
    selected_title = st.selectbox("Select movie", title_options, key="movie_select")
    selected_movie = title_to_movie_id[selected_title]

    metadata = get_movie_metadata(movies, ratings, tags, selected_movie)
    if metadata is None:
        st.warning("Movie not found.")
        return

    st.subheader(metadata["title"])
    render_genre_badges(metadata["genres"])

    col1, col2 = st.columns(2)
    avg_rating = metadata["avg_rating"]
    col1.metric("Average rating", f"{avg_rating:.2f} / 5" if avg_rating is not None else "N/A")
    col2.metric("Number of ratings", metadata["num_ratings"])

    if metadata["tags"]:
        st.caption("Tags: " + ", ".join(f"`{tag}`" for tag in metadata["tags"]))

    st.divider()
    st.subheader("Similar Movies")
    top_n = st.slider("Number of similar movies", min_value=5, max_value=50, value=10, key="item_top_n")

    sims = get_similar_movies(selected_movie, model, mappings, genre_feature_matrix, top_n=top_n)

    sims_df = pd.DataFrame(sims)
    if sims_df.empty:
        st.info("No similar movies found.")
        return

    sims_df = sims_df.merge(movies, on="movieId", how="left")
    sims_df = sims_df[["title", "genres", "score"]]
    sims_df.insert(0, "Rank", range(1, len(sims_df) + 1))
    sims_df.columns = ["Rank", "Title", "Genres", "Similarity"]

    st.dataframe(
        sims_df,
        width="stretch",
        hide_index=True,
        column_config={
            "Similarity": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.2f"),
        },
    )


def main():
    model, mappings, genre_feature_matrix = load_model_artifacts()
    movies, ratings, tags, links = load_data()

    with st.sidebar:
        st.title("🎬 MovieLens Recommender")
        st.caption("Neural collaborative filtering over the MovieLens dataset")
        page = st.radio(
            "Navigate",
            ["User Recommendations", "Item Similarity"],
            key="nav_page",
            label_visibility="collapsed",
        )
        st.divider()
        st.caption(f"{len(movies):,} movies · {ratings['userId'].nunique():,} users · {len(ratings):,} ratings")

    if page == "User Recommendations":
        render_user_page(model, mappings, genre_feature_matrix, movies, ratings)
    else:
        render_item_page(model, mappings, genre_feature_matrix, movies, ratings, tags)


if __name__ == "__main__":
    main()
