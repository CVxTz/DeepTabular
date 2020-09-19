# Data : https://www.kaggle.com/c/lish-moa/overview/evaluation
import json

import pandas as pd

if __name__ == "__main__":
    data_f = pd.read_csv("../data/lish-moa/train_features.csv")
    data_t = pd.read_csv("../data/lish-moa/train_targets_scored.csv")

    cols = dict()

    cols["targets"] = data_t.columns.values[1:].tolist()
    cols["cat_cols"] = ["cp_type", "cp_time", "cp_dose"]
    cols["num_cols"] = ["g-%s" % i for i in range(772)] + [
        "c-%s" % i for i in range(100)
    ]

    data = pd.merge(data_f, data_t, how="inner", on="sig_id")

    json.dump(cols, open("cols.json", "w"), indent=4)
