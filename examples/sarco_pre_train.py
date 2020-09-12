# Data : http://www.gaussianprocess.org/gpml/data/
import json
import scipy.io
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from deeptabular.deeptabular import DeepTabularRegressor, DeepTabularUnsupervised

if __name__ == "__main__":
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

    pretrain = DeepTabularUnsupervised(
        num_layers=6, cat_cols=cat_cols, num_cols=num_cols, n_targets=1,
    )

    pretrain.fit(
        train, save_path=None, epochs=34,
    )

    pretrain.save_config("sarco_config.json")
    pretrain.save_weigts("sarco_weights.h5")

    sizes = [500, 1000, 2000, 4000, 8000, 16000]

    scratch_mae = []
    pretrain_mae = []

    for size in sizes:
        regressor = DeepTabularRegressor(
            num_layers=6, cat_cols=cat_cols, num_cols=num_cols, n_targets=len(targets),
        )

        regressor.fit(train[:size], target_cols=targets, epochs=128)

        pred = regressor.predict(test).ravel()

        y_true = np.array(test[targets].values).ravel()

        scratch_mae.append(mean_absolute_error(y_true, pred))

        del regressor

    for size in sizes:
        regressor = DeepTabularRegressor(n_targets=len(targets))
        regressor.load_config("sarco_config.json")
        regressor.load_weights("sarco_weights.h5", by_name=True)

        regressor.fit(train[:size], target_cols=targets, epochs=128)

        pred = regressor.predict(test).ravel()

        y_true = np.array(test[targets].values).ravel()

        pretrain_mae.append(mean_absolute_error(y_true, pred))

        del regressor

    print("sizes", sizes)
    print("scratch_accuracies", scratch_mae)
    print("pretrain_accuracies", pretrain_mae)

    with open("sarco_performance.json", "w") as f:
        json.dump(
            {
                "sizes": sizes,
                "scratch_accuracies": scratch_mae,
                "pretrain_accuracies": pretrain_mae,
            },
            f,
        )
