# Data : http://www.gaussianprocess.org/gpml/data/
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.io
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    n = 5

    train_np = scipy.io.loadmat("../data/sarco/sarcos_inv.mat")["sarcos_inv"]
    train = pd.DataFrame(
        data=train_np,
        columns=["col%s" % (i + 1) for i in range(21)]
        + ["target%s" % (i + 1) for i in range(7)],
    )

    test_np = scipy.io.loadmat("../data/sarco/sarcos_inv_test.mat")["sarcos_inv_test"]
    test = pd.DataFrame(
        data=test_np,
        columns=["col%s" % (i + 1) for i in range(21)]
        + ["target%s" % (i + 1) for i in range(7)],
    )

    targets = ["target%s" % (i + 1) for i in range(7)]
    num_cols = ["col%s" % (i + 1) for i in range(21)]
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

    with open("sarco_lgbm_performance.json", "w") as f:
        json.dump(
            {"sizes": sizes, "scratch_mae": scratch_mae}, f, indent=4,
        )
