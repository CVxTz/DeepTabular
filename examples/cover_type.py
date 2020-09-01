import pandas as pd

from deeptabular.deeptabular import DeepTabularClassifier

if __name__ == "__main__":
    train = pd.read_csv("../data/cover_type/covtype.csv")

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

    for k in num_cols:
        train[k] = (train[k] - train[k].mean()) / train[k].std()

    classifier = DeepTabularClassifier(num_layers=4)

    classifier.fit(train, cat_cols=cal_cols, num_cols=num_cols, target_col=target)

    print(classifier.frequency)
    print(classifier.mapping)
