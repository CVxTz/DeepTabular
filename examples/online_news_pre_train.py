# Data : https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity
import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from deeptabular.deeptabular import DeepTabularRegressor, DeepTabularUnsupervised

if __name__ == "__main__":
    n = 5
    cols = [
        "url",
        "timedelta",
        "n_tokens_title",
        "n_tokens_content",
        "n_unique_tokens",
        "n_non_stop_words",
        "n_non_stop_unique_tokens",
        "num_hrefs",
        "num_self_hrefs",
        "num_imgs",
        "num_videos",
        "average_token_length",
        "num_keywords",
        "data_channel_is_lifestyle",
        "data_channel_is_entertainment",
        "data_channel_is_bus",
        "data_channel_is_socmed",
        "data_channel_is_tech",
        "data_channel_is_world",
        "kw_min_min",
        "kw_max_min",
        "kw_avg_min",
        "kw_min_max",
        "kw_max_max",
        "kw_avg_max",
        "kw_min_avg",
        "kw_max_avg",
        "kw_avg_avg",
        "self_reference_min_shares",
        "self_reference_max_shares",
        "self_reference_avg_sharess",
        "weekday_is_monday",
        "weekday_is_tuesday",
        "weekday_is_wednesday",
        "weekday_is_thursday",
        "weekday_is_friday",
        "weekday_is_saturday",
        "weekday_is_sunday",
        "is_weekend",
        "LDA_00",
        "LDA_01",
        "LDA_02",
        "LDA_03",
        "LDA_04",
        "global_subjectivity",
        "global_sentiment_polarity",
        "global_rate_positive_words",
        "global_rate_negative_words",
        "rate_positive_words",
        "rate_negative_words",
        "avg_positive_polarity",
        "min_positive_polarity",
        "max_positive_polarity",
        "avg_negative_polarity",
        "min_negative_polarity",
        "max_negative_polarity",
        "title_subjectivity",
        "title_sentiment_polarity",
        "abs_title_subjectivity",
        "abs_title_sentiment_polarity",
        "shares",
    ]

    data = pd.read_csv("../data/news/OnlineNewsPopularity.csv", names=cols, skiprows=1)
    data["shares"] = np.log(data["shares"] + 1)

    train, test = train_test_split(data, test_size=0.2, random_state=1337)

    targets = ["shares"]
    num_cols = [
        "n_tokens_title",
        "n_tokens_content",
        "n_unique_tokens",
        "n_non_stop_words",
        "n_non_stop_unique_tokens",
        "num_hrefs",
        "num_self_hrefs",
        "num_imgs",
        "num_videos",
        "average_token_length",
        "num_keywords",
        "data_channel_is_lifestyle",
        "data_channel_is_entertainment",
        "data_channel_is_bus",
        "data_channel_is_socmed",
        "data_channel_is_tech",
        "data_channel_is_world",
        "kw_min_min",
        "kw_max_min",
        "kw_avg_min",
        "kw_min_max",
        "kw_max_max",
        "kw_avg_max",
        "kw_min_avg",
        "kw_max_avg",
        "kw_avg_avg",
        "self_reference_min_shares",
        "self_reference_max_shares",
        "self_reference_avg_sharess",
        "weekday_is_monday",
        "weekday_is_tuesday",
        "weekday_is_wednesday",
        "weekday_is_thursday",
        "weekday_is_friday",
        "weekday_is_saturday",
        "weekday_is_sunday",
        "is_weekend",
        "LDA_00",
        "LDA_01",
        "LDA_02",
        "LDA_03",
        "LDA_04",
        "global_subjectivity",
        "global_sentiment_polarity",
        "global_rate_positive_words",
        "global_rate_negative_words",
        "rate_positive_words",
        "rate_negative_words",
        "avg_positive_polarity",
        "min_positive_polarity",
        "max_positive_polarity",
        "avg_negative_polarity",
        "min_negative_polarity",
        "max_negative_polarity",
        "title_subjectivity",
        "title_sentiment_polarity",
        "abs_title_subjectivity",
        "abs_title_sentiment_polarity",
    ]
    cat_cols = []

    for k in num_cols:
        mean = train[k].mean()
        std = train[k].std()
        train[k] = (train[k] - mean) / std
        test[k] = (test[k] - mean) / std

    train = train.sample(frac=1)

    pretrain = DeepTabularUnsupervised(
        num_layers=10, cat_cols=cat_cols, num_cols=num_cols, n_targets=1,
    )

    pretrain.fit(
        train, save_path="news", epochs=512,
    )

    sizes = [200, 500, 1000, 2000, 5000]

    scratch_mae = []
    pretrain_mae = []

    for size in sizes:

        mae = 0

        for _ in range(n):
            regressor = DeepTabularRegressor(
                num_layers=10,
                cat_cols=cat_cols,
                num_cols=num_cols,
                n_targets=len(targets),
            )

            regressor.fit(train[:size], target_cols=targets, epochs=256)

            pred = regressor.predict(test).ravel()

            y_true = np.array(test[targets].values).ravel()

            mae += mean_absolute_error(y_true, pred) / n

            del regressor

        scratch_mae.append(mae)

    for size in sizes:

        mae = 0

        for _ in range(n):
            regressor = DeepTabularRegressor(n_targets=len(targets))
            regressor.load_config("news_config.json")
            regressor.load_weights("news_weights.h5", by_name=True)

            regressor.fit(train[:size], target_cols=targets, epochs=256)

            pred = regressor.predict(test).ravel()

            y_true = np.array(test[targets].values).ravel()

            mae += mean_absolute_error(y_true, pred) / n

            del regressor

        pretrain_mae.append(mae)

    print("sizes", sizes)
    print("scratch_accuracies", scratch_mae)
    print("pretrain_accuracies", pretrain_mae)

    with open("news_performance.json", "w") as f:
        json.dump(
            {"sizes": sizes, "scratch_mae": scratch_mae, "pretrain_mae": pretrain_mae,},
            f,
            indent=4,
        )
