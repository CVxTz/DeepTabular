# Data : https://www.kaggle.com/uciml/forest-cover-type-dataset
import pandas as pd

from deeptabular.deeptabular import DeepTabularClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    data = pd.read_csv("../data/cover_type/covtype.csv", nrows=10000)

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

    classifier = DeepTabularClassifier(num_layers=6)

    classifier.fit(train, cat_cols=cal_cols, num_cols=num_cols, target_col=target)

    pred = classifier.predict(test)

    print("acc", accuracy_score(test[target], pred))

