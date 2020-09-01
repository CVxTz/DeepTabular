import pandas as pd

from deeptabular.deeptabular import DeepTabularClassifier

if __name__ == "__main__":
    train = pd.read_csv(
        "../data/poker/poker-hand-training-true.data",
        header=None,
        names=[
            "col1",
            "col2",
            "col3",
            "col4",
            "col5",
            "col6",
            "col7",
            "col8",
            "col9",
            "col10",
            "target",
        ],
    )

    print(train)

    classifier = DeepTabularClassifier(num_layers=1)

    classifier.fit(
        train,
        cat_cols=[
            "col1",
            "col2",
            "col3",
            "col4",
            "col5",
            "col6",
            "col7",
            "col8",
            "col9",
            "col10",
        ],
        num_cols=[],
        target_col="target",
    )

    print(classifier.frequency)
    print(classifier.mapping)
