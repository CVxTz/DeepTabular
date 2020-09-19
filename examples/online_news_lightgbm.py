# Data : https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    n = 10
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

    for feature in cat_cols:
        train[feature] = pd.Series(train[feature], dtype="category")
        test[feature] = pd.Series(test[feature], dtype="category")

    sizes = [200, 500, 1000, 2000, 5000]

    scratch_mae = []

    lgbm_params = {
        "objective": "regression_l1",
        "metric": "l1",
        # 'max_depth': 5,
        "num_leaves": 300,
        "feature_fraction": 0.5,
        "bagging_fraction": 0.75,
        "bagging_freq": 2,
        "learning_rate": 0.0001,
        "verbose": 0,
    }

    for size in sizes:

        mae = 0

        for _ in range(n):

            tr, val = train_test_split(train[:size], test_size=0.1)

            pred = np.zeros((test.shape[0], len(targets)))

            for i, target in enumerate(targets):
                train_data = lgb.Dataset(
                    tr[num_cols + cat_cols],
                    label=tr[target],
                    feature_name=num_cols + cat_cols,
                    categorical_feature=cat_cols,
                )
                validation_data = lgb.Dataset(
                    val[num_cols + cat_cols],
                    label=val[target],
                    feature_name=num_cols + cat_cols,
                    categorical_feature=cat_cols,
                )

                num_round = 1000

                bst = lgb.train(
                    lgbm_params,
                    train_data,
                    num_round,
                    valid_sets=[validation_data],
                    early_stopping_rounds=20,
                    verbose_eval=10,
                )

                pred[:, i] = bst.predict(test[num_cols + cat_cols])

            pred = pred.ravel()
            y_true = np.array(test[targets].values).ravel()

            mae += mean_absolute_error(y_true, pred) / n

        scratch_mae.append(mae)

    print("sizes", sizes)
    print("scratch_mae", scratch_mae)

    with open("news_lgbm_performance.json", "w") as f:
        json.dump(
            {"sizes": sizes, "scratch_mae": scratch_mae}, f, indent=4,
        )
