import pandas as pd
import scipy.io
import numpy as np

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
    cal_cols = []

    for k in num_cols:
        train[k] = (train[k] - train[k].mean()) / train[k].std()
        test[k] = (test[k] - test[k].mean()) / test[k].std()

    regressor = DeepTabularRegressor(num_layers=6)

    regressor.fit(train, cat_cols=cal_cols, num_cols=num_cols, target_cols=targets)
