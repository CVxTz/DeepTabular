# Data : https://www.kaggle.com/c/lish-moa/overview/evaluation
import json

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

from deeptabular.deeptabular import DeepTabularMultiLabel

if __name__ == "__main__":

    data_f = pd.read_csv("../data/lish-moa/train_features.csv")
    data_t = pd.read_csv("../data/lish-moa/train_targets_scored.csv")

    test = pd.read_csv("../data/lish-moa/test_features.csv")

    data = pd.merge(data_f, data_t, how="inner", on="sig_id")

    cols = json.load(open("cols.json", "r"))

    targets = cols["targets"]
    num_cols = cols["num_cols"]
    cat_cols = cols["cat_cols"]

    train, val = train_test_split(data, test_size=0.1, random_state=1337)

    for k in num_cols:
        mean = train[k].mean()
        std = train[k].std()
        train[k] = (train[k] - mean) / std
        test[k] = (test[k] - mean) / std

    classifier = DeepTabularMultiLabel(
        num_layers=12, cat_cols=cat_cols, num_cols=num_cols, n_targets=len(targets),
    )

    classifier.fit(train, target_cols=targets, epochs=64)

    pred_test = classifier.predict(test)
    pred_val = classifier.predict(val)

    y_true_test = np.array(test[targets].values)
    y_true_val = np.array(test[targets].values)

    score = 0
    for i in range(len(targets)):
        score_ = log_loss(y_true_val[:, i], pred_val[:, i])
        score += score_ / len(targets)

    print(score)
