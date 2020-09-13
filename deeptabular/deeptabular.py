import json
import random
from collections import Counter
from typing import List, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tqdm import tqdm

from .models import transformer_tabular


class DeepTabular:
    def __init__(
        self, cat_cols=None, num_cols=None, n_targets=None, num_layers=4, dropout=0.2,
    ):
        self.num_layers = num_layers
        self.dropout = dropout
        self.model = None
        self.mapping = None
        self.frequency = None
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.n_targets = n_targets

    def fit_mapping(self, df: pd.DataFrame):

        self.frequency = dict()

        for col in tqdm(self.cat_cols):
            values = df[col].apply(lambda x: "%s_%s" % (col, str(x))).tolist()
            count = Counter(values)

            self.frequency.update(count)

        self.mapping = {
            k: i + 1 for i, k in enumerate(list(self.frequency.keys()) + self.num_cols)
        }

    def prepare_data(self, df, add_distractors=False):
        data_x1 = []
        data_x2 = []

        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            sample_x1 = []
            sample_x2 = []

            sample_x1 += ["%s_%s" % (col, str(row[col])) for col in self.cat_cols]
            sample_x1 += [col for col in self.num_cols]

            sample_x2 += [1 for _ in self.cat_cols]
            sample_x2 += [row[col] for col in self.num_cols]

            if add_distractors and len(self.cat_cols):
                distractors_x1 = random.sample(list(self.mapping), len(self.cat_cols))
                distractors_x2 = [1 if a in sample_x1 else 0 for a in distractors_x1]

                sample_x1 += distractors_x1
                sample_x2 += distractors_x2

            sample_x1 = [self.mapping.get(x, 0) for x in sample_x1]
            data_x1.append(sample_x1)
            data_x2.append(sample_x2)

        return data_x1, data_x2

    @staticmethod
    def build_callbacks(monitor, patience_early, patience_reduce, save_path):
        callbacks = []
        if save_path is not None:
            checkpoint = ModelCheckpoint(
                save_path,
                monitor=monitor,
                verbose=1,
                save_best_only=True,
                save_weights_only=True,
            )
            callbacks.append(checkpoint)

        reduce = ReduceLROnPlateau(
            monitor=monitor, patience=patience_reduce, min_lr=1e-7
        )
        callbacks.append(reduce)

        early = EarlyStopping(monitor=monitor, patience=patience_early)

        callbacks.append(early)

        return callbacks

    def build_model(self):
        raise NotImplementedError

    def save_config(self, path):
        with open(path, "w") as f:
            json.dump(
                {
                    "mapping": self.mapping,
                    "cat_cols": self.cat_cols,
                    "num_cols": self.num_cols,
                    "n_targets": self.n_targets,
                    "num_layers": self.num_layers,
                    "dropout": self.dropout,
                    "frequency": self.frequency,
                },
                f,
                indent=4,
            )

    def save_weigts(self, path):
        self.model.save_weights(path)

    def load_config(self, path):
        with open(path, "r") as f:
            config = json.load(f)

        self.mapping = config["mapping"]
        self.cat_cols = config["cat_cols"]
        self.num_cols = config["num_cols"]
        self.n_targets = (
            config["n_targets"] if self.n_targets is None else self.n_targets
        )
        self.num_layers = config["num_layers"]
        self.dropout = config["dropout"]
        self.frequency = config["frequency"]

    def load_weights(self, path, by_name=False):

        self.build_model()
        self.model.load_weights(path, by_name=by_name)


class DeepTabularClassifier(DeepTabular):
    def __init__(
        self, cat_cols=None, num_cols=None, n_targets=None, num_layers=4, dropout=0.2,
    ):
        super().__init__(
            cat_cols=cat_cols,
            num_cols=num_cols,
            n_targets=n_targets,
            num_layers=num_layers,
            dropout=dropout,
        )

    def build_model(self):
        model = transformer_tabular(
            n_categories=len(self.mapping) + 1,
            n_targets=self.n_targets,
            num_layers=self.num_layers,
            dropout=self.dropout,
            seq_len=(0 if self.cat_cols is None else len(self.cat_cols))
            + (0 if self.num_cols is None else len(self.num_cols)),
            embeds_size=50,
            flatten=True,
        )
        self.model = model

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str,
        monitor: str = "val_acc",
        patience_early: int = 15,
        patience_reduce: int = 9,
        save_path: Union[str, None] = "classifier.h5",
        epochs=128,
    ):

        if self.mapping is None:

            self.fit_mapping(df)

        self.build_model()

        data_x1, data_x2 = self.prepare_data(df)

        data_y = df[target_col].tolist()

        train_x1, val_x1, train_x2, val_x2, train_y, val_y = train_test_split(
            data_x1, data_x2, data_y, test_size=0.1, random_state=1337, stratify=data_y
        )

        train_x1 = np.array(train_x1)
        val_x1 = np.array(val_x1)
        train_x2 = np.array(train_x2)[..., np.newaxis]
        val_x2 = np.array(val_x2)[..., np.newaxis]
        train_y = np.array(train_y)[..., np.newaxis]
        val_y = np.array(val_y)[..., np.newaxis]

        callbacks = self.build_callbacks(
            monitor, patience_early, patience_reduce, save_path
        )

        self.model.fit(
            [train_x1, train_x2],
            train_y,
            validation_data=([val_x1, val_x2], val_y),
            epochs=epochs,
            callbacks=callbacks,
            batch_size=128,
        )

    def predict(self, test):
        data_x1, data_x2 = self.prepare_data(test)

        data_x1 = np.array(data_x1)
        data_x2 = np.array(data_x2)[..., np.newaxis]

        predict = self.model.predict([data_x1, data_x2])

        if self.n_targets > 1:
            pred_classes = np.argmax(predict.squeeze(), axis=-1).ravel()
        else:
            pred_classes = (predict.squeeze() > 0.5).ravel().astype(np.int)

        return pred_classes


class DeepTabularRegressor(DeepTabular):
    def __init__(
        self, cat_cols=None, num_cols=None, n_targets=None, num_layers=4, dropout=0.2,
    ):
        super().__init__(
            cat_cols=cat_cols,
            num_cols=num_cols,
            n_targets=n_targets,
            num_layers=num_layers,
            dropout=dropout,
        )

    def build_model(self):
        model = transformer_tabular(
            n_categories=len(self.mapping) + 1,
            n_targets=self.n_targets,
            num_layers=self.num_layers,
            dropout=self.dropout,
            seq_len=(0 if self.cat_cols is None else len(self.cat_cols))
            + (0 if self.num_cols is None else len(self.num_cols)),
            embeds_size=50,
            flatten=True,
            task="regression",
        )
        self.model = model

    def fit(
        self,
        df: pd.DataFrame,
        target_cols: List[str],
        monitor: str = "val_loss",
        patience_early: int = 15,
        patience_reduce: int = 9,
        save_path: Union[str, None] = "regressor.h5",
        epochs=128,
    ):

        if self.mapping is None:

            self.fit_mapping(df)

        self.build_model()

        data_x1, data_x2 = self.prepare_data(df)

        data_y = df[target_cols].values

        train_x1, val_x1, train_x2, val_x2, train_y, val_y = train_test_split(
            data_x1, data_x2, data_y, test_size=0.1, random_state=1337
        )

        train_x1 = np.array(train_x1)
        val_x1 = np.array(val_x1)
        train_x2 = np.array(train_x2)[..., np.newaxis]
        val_x2 = np.array(val_x2)[..., np.newaxis]
        train_y = np.array(train_y)
        val_y = np.array(val_y)

        callbacks = self.build_callbacks(
            monitor, patience_early, patience_reduce, save_path
        )

        self.model.fit(
            [train_x1, train_x2],
            train_y,
            validation_data=([val_x1, val_x2], val_y),
            epochs=epochs,
            callbacks=callbacks,
            batch_size=128,
        )

    def predict(self, test):
        data_x1, data_x2 = self.prepare_data(test)

        data_x1 = np.array(data_x1)
        data_x2 = np.array(data_x2)[..., np.newaxis]

        predict = self.model.predict([data_x1, data_x2])

        return predict


class DeepTabularUnsupervised(DeepTabular):
    def __init__(
        self, cat_cols=None, num_cols=None, n_targets=None, num_layers=4, dropout=0.2,
    ):
        super().__init__(
            cat_cols=cat_cols,
            num_cols=num_cols,
            n_targets=n_targets,
            num_layers=num_layers,
            dropout=dropout,
        )

    def build_model(self):
        model = transformer_tabular(
            n_categories=len(self.mapping) + 1,
            n_targets=self.n_targets,
            num_layers=self.num_layers,
            dropout=self.dropout,
            seq_len=2 * (0 if self.cat_cols is None else len(self.cat_cols))
            + (0 if self.num_cols is None else len(self.num_cols)),
            embeds_size=50,
            flatten=False,
            task="pretrain",
        )
        self.model = model

    def fit(
        self,
        df: pd.DataFrame,
        monitor: str = "loss",
        patience_early: int = 15,
        patience_reduce: int = 9,
        save_path: Union[str, None] = "unsupervised.h5",
        epochs=128,
    ):

        if self.mapping is None:

            self.fit_mapping(df)

        self.build_model()

        data_x1, data_x2 = self.prepare_data(df, add_distractors=True)

        train_x1, val_x1, train_x2, val_x2 = train_test_split(
            data_x1, data_x2, test_size=0.1, random_state=1337
        )

        train_x1 = np.array(train_x1)
        val_x1 = np.array(val_x1)
        train_x2 = np.array(train_x2)[..., np.newaxis]
        val_x2 = np.array(val_x2)[..., np.newaxis]

        callbacks = self.build_callbacks(
            monitor, patience_early, patience_reduce, save_path
        )

        self.model.fit(
            [train_x1, train_x2],
            train_x2,
            validation_data=([val_x1, val_x2], val_x2),
            epochs=epochs,
            callbacks=callbacks,
            batch_size=128,
        )
