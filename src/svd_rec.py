import time
import pandas as pd

from surprise import SVD, accuracy
from surprise.model_selection import GridSearchCV

from data_utils import load_ratings, build_surprise_dataset, build_folds, save_predictions, report_cv_results


hyperparameter_space = {
    "n_factors": [50, 100, 150, 200],
    "n_epochs": [20, 50, 100],
    "lr_all": [0.002, 0.005, 0.01],
    "reg_all": [0.02, 0.05, 0.1]
}


def search_svd_params(dataset):
    gs = GridSearchCV(SVD, hyperparameter_space, measures=["rmse"], cv=5, n_jobs=1)
    gs.fit(dataset)
    best_params = gs.best_params["rmse"]

    print(f"Optimal RMSE: {gs.best_score['rmse']:.4f}")
    print(f"Optimal parameters: {best_params}")
    
    return best_params


def svd_cv(folds, n_factors, n_epochs, lr_all, reg_all):
    fold_results = []
    predictions = []

    for fold_i, (trainset, testset) in enumerate(folds):
        algo = SVD(
            n_factors=n_factors,
            n_epochs=n_epochs,
            lr_all=lr_all,
            reg_all=reg_all,
            random_state=42,
        )

        start = time.time()
        algo.fit(trainset)
        fold_predictions = algo.test(testset)
        elapsed = time.time() - start

        rmse = accuracy.rmse(fold_predictions, verbose=False)
        mae = accuracy.mae(fold_predictions, verbose=False)

        fold_results.append({
            "fold": fold_i + 1,
            "rmse": rmse,
            "mae": mae,
            "time": elapsed,
        })

        predictions.extend([
            {
                "userId": int(pred.uid),
                "movieId": int(pred.iid),
                "predicted_rating": pred.est,
            }
            for pred in fold_predictions
        ])

    return fold_results, pd.DataFrame(predictions)


def main():
    ratings = load_ratings()
    dataset = build_surprise_dataset(ratings)
    folds = build_folds(dataset)

    best_params = search_svd_params(dataset)

    fold_results, predictions = svd_cv(
        folds,
        n_factors=best_params["n_factors"],
        n_epochs=best_params["n_epochs"],
        lr_all=best_params["lr_all"],
        reg_all=best_params["reg_all"],
    )

    report_cv_results(fold_results)
    save_predictions(predictions, "svd_predictions.csv")


if __name__ == "__main__": main()
