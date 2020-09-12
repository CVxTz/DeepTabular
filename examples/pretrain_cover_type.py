# Data : https://www.kaggle.com/uciml/forest-cover-type-dataset
import pandas as pd
import numpy as np
import json

from deeptabular.deeptabular import DeepTabularClassifier, DeepTabularUnsupervised
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    data = pd.read_csv("../data/cover_type/covtype.csv")

    train, test = train_test_split(data, test_size=0.2, random_state=1337)

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
    cal_cols = ["Soil_Type%s" % (i + 1) for i in range(40)] + [
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

    pretrain = DeepTabularUnsupervised(num_layers=6)

    pretrain.fit(
        train,
        cat_cols=cal_cols,
        num_cols=num_cols,
        save_path="unsupervised.h5",
        epochs=64,
    )

    pretrain.save_config("config.json")

    sizes = [1000, 2000, 4000, 8000, 16000]

    scratch_accuracies = []
    pretrain_accuracies = []

    for size in sizes:

        classifier = DeepTabularClassifier(num_layers=6)

        classifier.fit(
            train[:size],
            cat_cols=cal_cols,
            num_cols=num_cols,
            target_col=target,
            mapping=pretrain.mapping,
        )

        pred = classifier.predict(test)

        y_true = np.array(test[target].values).ravel()

        scratch_accuracies.append(accuracy_score(y_true, pred))

        del classifier

    for size in sizes:
        classifier = DeepTabularClassifier(num_layers=6)

        classifier.fit(
            train[:size],
            cat_cols=cal_cols,
            num_cols=num_cols,
            target_col=target,
            mapping=pretrain.mapping,
            weights="unsupervised.h5",
        )

        pred = classifier.predict(test)

        y_true = np.array(test[target].values).ravel()

        pretrain_accuracies.append(accuracy_score(y_true, pred))

        del classifier

    print("sizes", sizes)
    print("scratch_accuracies", scratch_accuracies)
    print("pretrain_accuracies", pretrain_accuracies)

    with open("pretrain.json", "w") as f:
        json.dump(
            {
                "sizes": sizes,
                "scratch_accuracies": scratch_accuracies,
                "pretrain_accuracies": pretrain_accuracies,
            },
            f,
        )
