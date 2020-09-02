# Data : https://www.kaggle.com/c/otto-group-product-classification-challenge/
import pandas as pd


from deeptabular.deeptabular import DeepTabularClassifier

if __name__ == "__main__":
    train = pd.read_csv("../data/otto/train.csv")

    target = "target"

    num_cols = []
    cal_cols = ["feat_%s" % (i + 1) for i in range(93)]

    for k in num_cols:
        train[k] = (train[k] - train[k].mean()) / train[k].std()

    train[target] = train[target].map({"Class_%s" % (i + 1): i for i in range(9)})

    classifier = DeepTabularClassifier(num_layers=4)

    classifier.fit(train, cat_cols=cal_cols, num_cols=num_cols, target_col=target)
