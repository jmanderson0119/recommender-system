import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import ParameterGrid

from data_utils import (load_ratings, build_surprise_dataset, build_folds, build_id_maps,
                        trainset_to_arrays, testset_to_arrays, compute_rmse, compute_mae,
                        save_predictions, report_cv_results)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Architecture parameters, optimizer settings held fixed
architecture_parameter_space = {
    "embedding_dim": [32, 64, 128],
    "hidden_layers": [[128, 64], [256, 128], [256, 128, 64]]
}


# Optimizer parameters, architecture parameters are fixed optimally from the previous search
optimizer_parameter_space = {
    "lr": [0.001, 0.005, 0.01],
    "dropout": [0.1, 0.2, 0.4],
    "batch_size": [128, 256, 512]
}


# Fixed during search, carried into cross-validation
epochs = 20
learning_rate = 0.001
dropout = 0.2
batch_size = 256


class EmbeddingNet(nn.Module):
    def __init__(self, n_users, n_movies, embedding_dim, hidden_layers, dropout):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.movie_embedding = nn.Embedding(n_movies, embedding_dim)
        self.embedding_dropout = nn.Dropout(dropout)

        layers = []
        input_dim = embedding_dim * 2

        for hidden_dim in hidden_layers:
            layers += [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))
        
        self.hidden = nn.Sequential(*layers)
        self.min_rating = 0.5
        self.max_rating = 5.0

    def forward(self, user_ids, movie_ids):
        user_emb = self.user_embedding(user_ids)
        movie_emb = self.movie_embedding(movie_ids)
        x = self.embedding_dropout(torch.cat([user_emb, movie_emb], dim=1))
        
        # Scaled to the rating range
        return self.min_rating + (self.max_rating - self.min_rating) * torch.sigmoid(self.hidden(x).squeeze(1))


def make_tensors(users, movies, ratings):
    return (
        torch.tensor(users, dtype=torch.long),
        torch.tensor(movies, dtype=torch.long),
        torch.tensor(ratings, dtype=torch.float32),
    )


def train_and_evaluate(n_users, n_movies, train_users, train_movies, train_ratings,
                       test_users, test_movies, test_ratings,
                       embedding_dim, hidden_layers, dropout, lr, batch_size, n_epochs):
    torch.manual_seed(42)

    model = EmbeddingNet(n_users, n_movies, embedding_dim, hidden_layers, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_dataset = TensorDataset(*make_tensors(train_users, train_movies, train_ratings))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(n_epochs):
        for batch_users, batch_movies, batch_ratings in train_loader:
            batch_users = batch_users.to(device)
            batch_movies = batch_movies.to(device)
            batch_ratings = batch_ratings.to(device)
            optimizer.zero_grad()
            loss_fn(model(batch_users, batch_movies), batch_ratings).backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        t_users, t_movies, _ = make_tensors(test_users, test_movies, test_ratings)
        preds = model(t_users.to(device), t_movies.to(device)).cpu().numpy()

    rmse = compute_rmse(test_ratings, preds)
    return rmse, preds


# Architecture search over all folds with fixed optimizer settings
def architecture_search(folds, user_map, movie_map, n_users, n_movies):
    best_rmse = float("inf")
    best_params = None

    for params in ParameterGrid(architecture_parameter_space):
        fold_rmses = []

        for trainset, testset in folds:
            train_users, train_movies, train_ratings = trainset_to_arrays(trainset, user_map, movie_map)
            test_users, test_movies, test_ratings = testset_to_arrays(testset, user_map, movie_map)
            
            rmse, _ = train_and_evaluate(
                n_users, n_movies,
                train_users, train_movies, train_ratings,
                test_users, test_movies, test_ratings,
                embedding_dim=params["embedding_dim"],
                hidden_layers=params["hidden_layers"],
                dropout=dropout,
                lr=learning_rate,
                batch_size=batch_size,
                n_epochs=epochs,
            )

            fold_rmses.append(rmse)

        average_rmse = np.mean(fold_rmses)
        if average_rmse < best_rmse:
            best_rmse = average_rmse
            best_params = params

    print(f"Architecture search:")
    print(f"Optimal RMSE: {best_rmse:.4f}")
    print(f"Optimal parameters: {best_params}")

    return best_params


# Optimizer search over all 5 folds with architecture fixed at best from architecture search
def optimizer_search(folds, user_map, movie_map, n_users, n_movies, embedding_dim, hidden_layers):
    best_rmse = float("inf")
    best_params = None

    for params in ParameterGrid(optimizer_parameter_space):
        fold_rmses = []

        for trainset, testset in folds:
            train_users, train_movies, train_ratings = trainset_to_arrays(trainset, user_map, movie_map)
            test_users, test_movies, test_ratings = testset_to_arrays(testset, user_map, movie_map)
            
            rmse, _ = train_and_evaluate(
                n_users, n_movies,
                train_users, train_movies, train_ratings,
                test_users, test_movies, test_ratings,
                embedding_dim=embedding_dim,
                hidden_layers=hidden_layers,
                dropout=params["dropout"],
                lr=params["lr"],
                batch_size=params["batch_size"],
                n_epochs=epochs,
            )

            fold_rmses.append(rmse)

        average_rmse = np.mean(fold_rmses)
        if average_rmse < best_rmse:
            best_rmse = average_rmse
            best_params = params

    print(f"Optimizer search:")
    print(f"Optimal RMSE: {best_rmse:.4f}")
    print(f"Optimal parameters: {best_params}")

    return best_params


def nn_cv(folds, user_map, movie_map, n_users, n_movies,
           embedding_dim, hidden_layers, dropout, lr, batch_size):
    fold_results = []
    predictions = []

    for fold_i, (trainset, testset) in enumerate(folds):
        train_users, train_movies, train_ratings = trainset_to_arrays(trainset, user_map, movie_map)
        test_users, test_movies, test_ratings = testset_to_arrays(testset, user_map, movie_map)

        start = time.time()
        rmse, preds = train_and_evaluate(
            n_users, n_movies,
            train_users, train_movies, train_ratings,
            test_users, test_movies, test_ratings,
            embedding_dim=embedding_dim,
            hidden_layers=hidden_layers,
            dropout=dropout,
            lr=lr,
            batch_size=batch_size,
            n_epochs=epochs,
        )
        elapsed = time.time() - start

        mae = compute_mae(test_ratings, preds)

        fold_results.append({
            "fold": fold_i + 1,
            "rmse": rmse,
            "mae": mae,
            "time": elapsed,
        })

        predictions.extend([
            {
                "userId": int(uid),
                "movieId": int(iid),
                "predicted_rating": float(p),
            }
            for (uid, iid, _), p in zip(testset, preds)
        ])

    return fold_results, pd.DataFrame(predictions)


def main():
    ratings = load_ratings()
    dataset = build_surprise_dataset(ratings)
    folds = build_folds(dataset)
    user_map, movie_map = build_id_maps(ratings)

    n_users = len(user_map)
    n_movies = len(movie_map)

    architecture = architecture_search(folds, user_map, movie_map, n_users, n_movies)

    optimizers = optimizer_search(
        folds, user_map, movie_map, n_users, n_movies,
        embedding_dim=architecture["embedding_dim"],
        hidden_layers=architecture["hidden_layers"],
    )

    fold_results, predictions = nn_cv(
        folds, user_map, movie_map, n_users, n_movies,
        embedding_dim=architecture["embedding_dim"],
        hidden_layers=architecture["hidden_layers"],
        dropout=optimizers["dropout"],
        lr=optimizers["lr"],
        batch_size=optimizers["batch_size"],
    )

    report_cv_results(fold_results)
    save_predictions(predictions, "nn_predictions.csv")


if __name__ == "__main__": main()