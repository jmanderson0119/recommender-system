import os
import numpy as np
import pandas as pd

from surprise import Dataset, Reader
from surprise.model_selection import KFold


pwd = os.path.dirname(os.path.abspath(__file__))
ratings = os.path.join(pwd, "..", "data", "ratings.csv")
data_directory = os.path.join(pwd, "..", "data")
output_directory = os.path.join(pwd, "..", "outputs")
os.makedirs(output_directory, exist_ok=True)


def load_ratings():
    return pd.read_csv(ratings)


# Contiguous index maps for NN embedding layers
def build_id_maps(ratings_df):
    user_ids = sorted(ratings_df["userId"].unique())
    movie_ids = sorted(ratings_df["movieId"].unique())
    user_map = {uid: i for i, uid in enumerate(user_ids)}
    movie_map = {mid: i for i, mid in enumerate(movie_ids)}
    return user_map, movie_map


def build_surprise_dataset(ratings_df):
    reader = Reader(rating_scale=(0.5, 5.0))
    return Dataset.load_from_df(ratings_df[["userId", "movieId", "rating"]], reader)


# These are the 5 shared folds that need to be used by all algorithms
def build_folds(dataset):
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    return list(kf.split(dataset))


# Converts a Surprise trainset to numpy arrays
def trainset_to_arrays(trainset, user_map, movie_map):
    users, movies, ratings = [], [], []
    for u, i, r in trainset.all_ratings():
        users.append(user_map[int(trainset.to_raw_uid(u))])
        movies.append(movie_map[int(trainset.to_raw_iid(i))])
        ratings.append(r)
    return (np.array(users), np.array(movies), np.array(ratings, dtype=np.float32))


# Converts a Surprise testset to numpy arrays
def testset_to_arrays(testset, user_map, movie_map):
    users, movies, ratings = [], [], []
    for uid, iid, r in testset:
        users.append(user_map[int(uid)])
        movies.append(movie_map[int(iid)])
        ratings.append(r)
    return (np.array(users), np.array(movies), np.array(ratings, dtype=np.float32))


def compute_rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def compute_mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def save_predictions(predictions_df, filename):
    predictions_df.to_csv(os.path.join(output_directory, filename), index=False)


def format_time(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return (f"{h}:{m:02d}:{s:02d}")


def report_cv_results(fold_results):
    avg_rmse = np.mean([r["rmse"] for r in fold_results])
    avg_mae = np.mean([r["mae"] for r in fold_results])
    avg_time = np.mean([r["time"] for r in fold_results])

    print(f"{'Fold':<8} {'RMSE':>8} {'MAE':>8} {'Time':>10}")
    for r in fold_results:
        print(f"{r['fold']:<8} {r['rmse']:>8.4f} {r['mae']:>8.4f} {format_time(r['time']):>10}")
    print(f"{'Mean':<8} {avg_rmse:>8.4f} {avg_mae:>8.4f} {format_time(avg_time):>10}")
