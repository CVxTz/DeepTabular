# Data : https://www.kaggle.com/uciml/adult-census-income/
import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from deeptabular.deeptabular import DeepTabularClassifier, DeepTabularUnsupervised

if __name__ == "__main__":

    n = 5

    data = pd.read_csv("../data/census/adult.csv")

    train, test = train_test_split(data, test_size=0.2, random_state=1337)

    target = "income"

    num_cols = ["age", "fnlwgt", "capital.gain", "capital.loss", "hours.per.week"]
    cat_cols = [
        "workclass",
        "education",
        "education.num",
        "marital.status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native.country",
    ]

    for k in num_cols:
        mean = train[k].mean()
        std = train[k].std()
        train[k] = (train[k] - mean) / std
        test[k] = (test[k] - mean) / std

    train[target] = train[target].map({"<=50K": 0, ">50K": 1})
    test[target] = test[target].map({"<=50K": 0, ">50K": 1})

    train = train.sample(frac=1)

    pretrain = DeepTabularUnsupervised(
        num_layers=10, cat_cols=cat_cols, num_cols=num_cols, n_targets=1,
    )

    pretrain.fit(
        train, save_path="census", epochs=512,
    )

    sizes = [200, 500, 1000, 2000, 5000]

    scratch_accuracies = []
    pretrain_accuracies = []

    for size in sizes:

        acc = 0

        for _ in range(n):
            classifier = DeepTabularClassifier(
                num_layers=10, cat_cols=cat_cols, num_cols=num_cols, n_targets=1,
            )
            classifier.fit(train.sample(n=size), target_col=target, epochs=256)

            pred = classifier.predict(test)

            y_true = np.array(test[target].values).ravel()

            acc += accuracy_score(y_true, pred) / n

            del classifier

        scratch_accuracies.append(acc)

    for size in sizes:

        acc = 0

        for _ in range(n):
            classifier = DeepTabularClassifier(n_targets=1)
            classifier.load_config("census_config.json")
            classifier.load_weights("census_weights.h5", by_name=True)

            classifier.fit(train.sample(n=size), target_col=target, epochs=256)

            pred = classifier.predict(test)

            y_true = np.array(test[target].values).ravel()

            acc += accuracy_score(y_true, pred) / n

            del classifier

        pretrain_accuracies.append(acc)

    print("sizes", sizes)
    print("scratch_accuracies", scratch_accuracies)
    print("pretrain_accuracies", pretrain_accuracies)

    with open("census_pretrain_performance.json", "w") as f:
        json.dump(
            {
                "sizes": sizes,
                "scratch_accuracies": scratch_accuracies,
                "pretrain_accuracies": pretrain_accuracies,
            },
            f,
            indent=4,
        )
