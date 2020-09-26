# Data : https://www.kaggle.com/uciml/forest-cover-type-dataset
import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from deeptabular.deeptabular import DeepTabularClassifier, DeepTabularUnsupervised

if __name__ == "__main__":

    n = 5

    data = pd.read_csv("../data/cover_type/covtype.csv")

    train, test = train_test_split(data, test_size=0.1, random_state=1337)

    target = "Cover_Type"
    num_cols = [
        "Elevation",
        "Aspect",
        "Slope",
        "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Hillshade_9am",
        "Hillshade_Noon",
        "Hillshade_3pm",
        "Horizontal_Distance_To_Fire_Points",
    ]
    cat_cols = ["Soil_Type%s" % (i + 1) for i in range(40)] + [
        "Wilderness_Area%s" % (i + 1) for i in range(4)
    ]

    train[target] = train[target] - 1
    test[target] = test[target] - 1

    for k in num_cols:
        mean = train[k].mean()
        std = train[k].std()
        train[k] = (train[k] - mean) / std
        test[k] = (test[k] - mean) / std

    train = train.sample(frac=1)

    pretrain = DeepTabularUnsupervised(
        num_layers=12, cat_cols=cat_cols, num_cols=num_cols, n_targets=1,
    )

    pretrain.fit(
        train, save_path="cover", epochs=128,
    )

    sizes = [800, 1000, 2000, 5000]

    scratch_accuracies = []
    pretrain_accuracies = []

    for size in sizes:

        acc = 0

        for _ in range(n):
            classifier = DeepTabularClassifier(
                num_layers=12,
                cat_cols=cat_cols,
                num_cols=num_cols,
                n_targets=int(train[target].max() + 1),
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
            classifier = DeepTabularClassifier(n_targets=int(train[target].max() + 1))
            classifier.load_config("cover_config.json")
            classifier.load_weights("cover_weights.h5", by_name=True)

            classifier.fit(train.sample(n=size), target_col=target, epochs=256)

            pred = classifier.predict(test)

            y_true = np.array(test[target].values).ravel()

            acc += accuracy_score(y_true, pred) / n

            del classifier

        pretrain_accuracies.append(acc)

    print("sizes", sizes)
    print("scratch_accuracies", scratch_accuracies)
    print("pretrain_accuracies", pretrain_accuracies)

    with open("cover_pretrain_performance.json", "w") as f:
        json.dump(
            {
                "sizes": sizes,
                "scratch_accuracies": scratch_accuracies,
                "pretrain_accuracies": pretrain_accuracies,
            },
            f,
            indent=4,
        )
