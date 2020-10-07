from deeptabular.deeptabular import (
    DeepTabular,
    DeepTabularClassifier,
    DeepTabularRegressor,
)
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, mean_absolute_error


def test_build_deeptabular():
    base_model = DeepTabular(
        cat_cols=["C1", "C2", "C3"], num_cols=["N1", "N2"], n_targets=3
    )
    assert isinstance(base_model, DeepTabular)


def test_build_mapping():
    base_model = DeepTabular(cat_cols=["C1", "C2"], num_cols=["N1", "N2"], n_targets=3)
    df = pd.DataFrame(
        {
            "C1": np.random.randint(0, 10, size=5000),
            "C2": np.random.randint(0, 10, size=5000),
            "N1": np.random.uniform(0, 10, size=5000),
            "N2": np.random.uniform(0, 10, size=5000),
        }
    )
    base_model.fit_mapping(df)

    mapping = base_model.mapping
    expected = {
        "C2_4",
        "C2_5",
        "C1_0",
        "C1_4",
        "C1_8",
        "C1_6",
        "C1_9",
        "C1_2",
        "C2_7",
        "C2_1",
        "C2_2",
        "C1_1",
        "C1_3",
        "C1_5",
        "C2_8",
        "C1_7",
        "C2_6",
        "C2_9",
        "C2_3",
        "N2",
        "N1",
        "C2_0",
    }

    assert set(mapping.keys()) == expected


def test_build_save():
    base_model = DeepTabular(cat_cols=["C1", "C2"], num_cols=["N1", "N2"], n_targets=3)
    df = pd.DataFrame(
        {
            "C1": np.random.randint(0, 10, size=5000),
            "C2": np.random.randint(0, 10, size=5000),
            "N1": np.random.uniform(0, 10, size=5000),
            "N2": np.random.uniform(0, 10, size=5000),
        }
    )
    base_model.fit_mapping(df)
    base_model.save_config("config.json")

    base_model_new = DeepTabular()
    base_model_new.load_config("config.json")

    assert base_model.mapping == base_model_new.mapping
    assert base_model.cat_cols == base_model_new.cat_cols
    assert base_model.num_cols == base_model_new.num_cols
    assert base_model.n_targets == base_model_new.n_targets


def test_build_classifier():
    classifier = DeepTabularClassifier(
        cat_cols=["C1", "C2"], num_cols=["N1", "N2"], n_targets=1, num_layers=1
    )
    df = pd.DataFrame(
        {
            "C1": np.random.randint(0, 10, size=5000),
            "C2": np.random.randint(0, 10, size=5000),
            "N1": np.random.uniform(-1, 1, size=5000),
            "N2": np.random.uniform(-1, 1, size=5000),
            "target": np.random.uniform(-1, 1, size=5000),
        }
    )
    df["target"] = df.apply(
        lambda x: 1 if (x["C1"] == 4 and x["N1"] < 0.5) else 0, axis=1
    )

    test = pd.DataFrame(
        {
            "C1": np.random.randint(0, 10, size=5000),
            "C2": np.random.randint(0, 10, size=5000),
            "N1": np.random.uniform(-1, 1, size=5000),
            "N2": np.random.uniform(-1, 1, size=5000),
            "target": np.random.uniform(-1, 1, size=5000),
        }
    )
    test["target"] = test.apply(
        lambda x: 1 if (x["C1"] == 4 and x["N1"] < 0.5) else 0, axis=1
    )

    classifier.fit(df, target_col="target", epochs=100)

    pred = classifier.predict(test)

    acc = accuracy_score(test["target"], pred)

    assert isinstance(classifier.model, tf.keras.models.Model)
    assert acc > 0.9


def test_build_regressor():
    classifier = DeepTabularRegressor(
        cat_cols=["C1", "C2"], num_cols=["N1", "N2"], n_targets=1, num_layers=1
    )
    df = pd.DataFrame(
        {
            "C1": np.random.randint(0, 10, size=5000),
            "C2": np.random.randint(0, 10, size=5000),
            "N1": np.random.uniform(-1, 1, size=5000),
            "N2": np.random.uniform(-1, 1, size=5000),
            "target": np.random.uniform(-1, 1, size=5000),
        }
    )
    df["target"] = df.apply(
        lambda x: x["N1"] * x["N2"] * x["C1"] / (x["C2"] + 1), axis=1
    )

    test = pd.DataFrame(
        {
            "C1": np.random.randint(0, 10, size=5000),
            "C2": np.random.randint(0, 10, size=5000),
            "N1": np.random.uniform(-1, 1, size=5000),
            "N2": np.random.uniform(-1, 1, size=5000),
            "target": np.random.uniform(-1, 1, size=5000),
        }
    )
    test["target"] = test.apply(
        lambda x: x["N1"] * x["N2"] * x["C1"] / (x["C2"] + 1), axis=1
    )

    classifier.fit(df, target_cols=["target"], epochs=100)

    pred = classifier.predict(test)

    mae = mean_absolute_error(test["target"], pred)

    assert isinstance(classifier.model, tf.keras.models.Model)
    assert mae < 0.5
