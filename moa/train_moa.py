# Data : https://www.kaggle.com/c/lish-moa/overview/evaluation
import json

import numpy as np
import pandas as pd

from deeptabular.deeptabular import DeepTabularMultiLabel

if __name__ == "__main__":

    data_f = pd.read_csv("../data/lish-moa/train_features.csv")
    data_t = pd.read_csv("../data/lish-moa/train_targets_scored.csv")

    test = pd.read_csv("../data/lish-moa/test_features.csv")

    train = pd.merge(data_f, data_t, how="inner", on="sig_id")

    cols = json.load(open("cols.json", "r"))

    targets = cols["targets"]
    num_cols = cols["num_cols"]
    cat_cols = cols["cat_cols"]

    for k in num_cols:
        mean = train[k].mean()
        std = train[k].std()
        train[k] = (train[k] - mean) / std
        test[k] = (test[k] - mean) / std

    classifier = DeepTabularMultiLabel(
        num_layers=8,
        cat_cols=cat_cols,
        num_cols=num_cols,
        n_targets=len(targets),
        d_model=64,
        dropout=0.1,
    )

    classifier.fit(train, target_cols=targets, epochs=32, batch_size=32)

    classifier.save_config("moa.json")
    classifier.save_weigts("moa.h5")

    pred_test = classifier.predict(test)

    y_true_test = np.array(test[targets].values)
