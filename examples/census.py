# Data : https://www.kaggle.com/uciml/adult-census-income/
import pandas as pd
from sklearn.model_selection import train_test_split

from deeptabular.deeptabular import DeepTabularClassifier

if __name__ == "__main__":
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

    print(train[target].mean())

    classifier = DeepTabularClassifier(
        num_layers=10, cat_cols=cat_cols, num_cols=num_cols, n_targets=1,
    )

    classifier.fit(train, target_col=target, epochs=128)

    pred = classifier.predict(test)

    classifier.save_config("census_config.json")
    classifier.save_weigts("census_weights.h5")

    new_classifier = DeepTabularClassifier()

    new_classifier.load_config("census_config.json")
    new_classifier.load_weights("census_weights.h5")

    new_pred = new_classifier.predict(test)

    print(pred[:100])

    print(new_pred[:100])
