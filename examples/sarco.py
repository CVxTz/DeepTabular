# Data : http://www.gaussianprocess.org/gpml/data/
import numpy as np
import pandas as pd
import scipy.io
from sklearn.metrics import mean_absolute_error

from deeptabular.deeptabular import DeepTabularRegressor

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

    regressor = DeepTabularRegressor(
        num_layers=6, cat_cols=cat_cols, num_cols=num_cols, n_targets=len(targets),
    )

    regressor.fit(train, target_cols=targets, epochs=64)

    pred = regressor.predict(test).ravel()

    y_true = np.array(test[targets].values).ravel()

    mae = mean_absolute_error(y_true, pred)

    print("mae", mae)
