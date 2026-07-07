import pandas as pd
import pickle
import os
from surprise import Reader, Dataset, SVD

def load_clean_data(data_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
    df = pd.read_csv(data_path)
    return df

def build_and_train_model(df):
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()
    
    model = SVD()
    model.fit(trainset)
    
    return model

def save_model(model, model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    DATA_PATH = os.path.join("data", "ratings.csv")
    MODEL_PATH = os.path.join("models", "svd_model.pkl")
    try:
        data = load_clean_data(DATA_PATH)
        trained_model = build_and_train_model(data)
        save_model(trained_model, MODEL_PATH)
        print("Success")
    except Exception as e:
        print(f"Error: {e}")