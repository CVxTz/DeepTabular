from tensorflow.keras.layers import (
    Input,
    GlobalMaxPool1D,
    Dense,
    Flatten,
    Embedding,
    Concatenate,
)
from tensorflow.keras.losses import (
    sparse_categorical_crossentropy,
    binary_crossentropy,
    mse,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from .transformer import Encoder


def transformer_tabular(
    n_categories,
    n_targets,
    embeds_size=10,
    num_layers=4,
    num_dense_layers=3,
    d_model=128,
    num_heads=8,
    dff=256,
    task="classification",
    lr=0.0001,
    dropout=0.1,
    seq_len=None,
    flatten=False,
):
    input_cols = Input(shape=(seq_len,))
    input_values = Input(shape=(seq_len, 1))

    x1 = Embedding(n_categories, embeds_size, name="embed")(input_cols)
    x2 = Dense(embeds_size, activation="linear", name="d1")(input_values)

    x = Concatenate(axis=-1)([x1, x2])

    x = Dense(d_model, activation="relu", name="d2")(x)

    encoder = Encoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        rate=dropout,
        maximum_position_encoding=seq_len*5 if seq_len is not None else 5000
    )

    x_encoded = encoder(x)

    if task in ['classification', 'regression']:

        if flatten:
            x = Dense(embeds_size, name="d3")(x_encoded)
            x = Flatten()(x)
        else:

            x = GlobalMaxPool1D()(x_encoded)

        for i in range(num_dense_layers):

            x = Dense(d_model, activation="relu", name="d4_%s" % i)(x)

    if task == "classification":
        if n_targets > 1:

            out = Dense(n_targets, activation="softmax", name="d5")(x)

        else:

            out = Dense(n_targets, activation="sigmoid", name="d6")(x)

    elif task == "regression":
        out = Dense(n_targets, activation="linear", name="d7")(x)

    elif task == "pretrain":
        out = Dense(1, activation="linear", name="d8")(x_encoded)

    else:
        raise NotImplementedError

    model = Model(inputs=[input_cols, input_values], outputs=out)

    opt = Adam(lr)

    if task == "classification":

        if n_targets > 1:

            model.compile(
                optimizer=opt, loss=sparse_categorical_crossentropy, metrics=["acc"]
            )

        else:

            model.compile(optimizer=opt, loss=binary_crossentropy, metrics=["acc"])

    elif task == "regression":

        model.compile(optimizer=opt, loss=mse)

    elif task == "pretrain":
        model.compile(optimizer=opt, loss=mse)

    model.summary()

    return model

