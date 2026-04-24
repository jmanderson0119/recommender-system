import time
import pandas as pd

from surprise import NormalPredictor, accuracy

from data_utils import load_ratings, build_surprise_dataset, build_folds, save_predictions, report_cv_results


def random_cv(folds):
    fold_results = []
    all_predictions = []

    for fold_i, (trainset, testset) in enumerate(folds):
        algo = NormalPredictor()

        start = time.time()
        algo.fit(trainset)
        predictions = algo.test(testset)
        elapsed = time.time() - start

        rmse = accuracy.rmse(predictions, verbose=False)
        mae = accuracy.mae(predictions, verbose=False)

        fold_results.append({
            "fold": fold_i + 1,
            "rmse": rmse,
            "mae": mae,
            "time": elapsed,
        })

        all_predictions.extend([
            {
                "userId": int(pred.uid),
                "movieId": int(pred.iid),
                "predicted_rating": pred.est,
            }
            for pred in predictions
        ])

    return fold_results, pd.DataFrame(all_predictions)


def main():
    ratings = load_ratings()
    dataset = build_surprise_dataset(ratings)
    folds = build_folds(dataset)
    fold_results, predictions = random_cv(folds)

    report_cv_results(fold_results)
    save_predictions(predictions, "random_predictions.csv")


if __name__ == "__main__": main()
