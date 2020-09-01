from tensorflow.keras.layers import (
    Input,
    GlobalMaxPool1D,
    Dense,
    Dropout,
    Embedding,
    Concatenate,
)
from tensorflow.keras.losses import sparse_categorical_crossentropy, binary_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from .transformer import Encoder


def transformer_classifier(
    n_categories,
    n_targets,
    embeds_size=10,
    num_layers=4,
    d_model=128,
    num_heads=8,
    dff=256,
    task="classification",
    lr=0.0001,
    dropout=0.01,
):
    input_cols = Input(shape=(None,))
    input_values = Input(shape=(None, 1))

    x1 = Embedding(n_categories, embeds_size)(input_cols)
    x2 = Dense(embeds_size, activation="linear")(input_values)

    x = Concatenate(axis=-1)([x1, x2])

    x = Dense(d_model, activation="relu")(x)

    encoder = Encoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        rate=dropout,
    )

    x = encoder(x)

    x = GlobalMaxPool1D()(x)

    x = Dense(4 * n_targets, activation="selu")(x)

    if task == "classification":
        if n_targets > 1:

            out = Dense(n_targets, activation="softmax")(x)

        else:

            out = Dense(n_targets, activation="sigmoid")(x)

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

    model.summary()

    return model
