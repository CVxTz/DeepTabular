from collections import Counter
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .models import transformer_classifier


class DeepTabularClassifier:
    def __init__(self, num_layers=4):
        self.model = None
        self.num_layers = num_layers
        self.mapping = None
        self.frequency = None

        self.cat_cols = None
        self.num_cols = None

    def fit(self, df: pd.DataFrame, cat_cols: List, num_cols: List, target_col: str):

        self.cat_cols = cat_cols
        self.num_cols = num_cols

        self.fit_mapping(df, cat_cols)

        data_x1, data_x2 = self.prepare_data(df)

        data_y = df[target_col].tolist()

        n_targets = df[target_col].max() + 1

        train_x1, val_x1, train_x2, val_x2, train_y, val_y = train_test_split(data_x1, data_x2, data_y, test_size=0.1,
                                                                              random_state=1337, stratify=data_y)

        train_x1 = np.array(train_x1)
        val_x1 = np.array(val_x1)
        train_x2 = np.array(train_x2)[..., np.newaxis]
        val_x2 = np.array(val_x2)[..., np.newaxis]
        train_y = np.array(train_y)[..., np.newaxis]
        val_y = np.array(val_y)[..., np.newaxis]

        self.model = transformer_classifier(n_categories=len(self.mapping) + 1, n_targets=n_targets,
                                            num_layers=self.num_layers)

        self.model.fit([train_x1, train_x2], train_y, validation_data=([val_x1, val_x2], val_y), nb_epoch=64)

    def fit_mapping(self, df: pd.DataFrame, cat_cols: List):

        self.frequency = dict()

        for col in cat_cols:
            values = df[col].apply(lambda x: '%s_%s' % (col, str(x))).tolist()
            count = Counter(values)

            self.frequency.update(count)

        self.mapping = {k: i + 1 for i, (k, v) in enumerate(self.frequency.items())}

    def prepare_data(self, df):
        data_x1 = []
        data_x2 = []

        for index, row in df.iterrows():
            sample_x1 = []
            sample_x2 = []

            sample_x1 += ['%s_%s' % (col, str(row[col])) for col in self.cat_cols]
            sample_x1 += [col for col in self.num_cols]

            sample_x1 = [self.mapping.get(x, 0) for x in sample_x1]

            sample_x2 += [1 for _ in self.cat_cols]
            sample_x2 += [row[col] for col in self.num_cols]

            data_x1.append(sample_x1)
            data_x2.append(sample_x2)

        return data_x1, data_x2
