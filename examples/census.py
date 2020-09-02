# Data : https://www.kaggle.com/uciml/adult-census-income/
import pandas as pd


from deeptabular.deeptabular import DeepTabularClassifier

if __name__ == "__main__":
    train = pd.read_csv("../data/census/adult.csv")

    target = "income"

    num_cols = ["age", "fnlwgt", "capital.gain", "capital.loss", "hours.per.week"]
    cal_cols = [
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
        train[k] = (train[k] - train[k].mean()) / train[k].std()

    train[target] = train[target].map({"<=50K": 0, ">50K": 1})

    print(train[target].mean())

    classifier = DeepTabularClassifier(num_layers=10)

    classifier.fit(train, cat_cols=cal_cols, num_cols=num_cols, target_col=target)
