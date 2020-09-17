from tensorflow.keras.layers import (
    Input,
    GlobalMaxPool1D,
    Dense,
    Flatten,
    Embedding,
    Add,
    Concatenate
)
from tensorflow.keras.losses import (
    sparse_categorical_crossentropy,
    binary_crossentropy,
    mse,
    mae,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from .transformer import EncoderLayer


def transformer_tabular(
    n_categories,
    n_targets,
    embeds_size=10,
    num_layers=4,
    num_dense_layers=3,
    d_model=64,
    num_heads=4,
    dff=256,
    task="classification",
    lr=0.0001,
    dropout=0.01,
    seq_len=None,
):
    input_cols = Input(shape=(seq_len,))
    input_values = Input(shape=(seq_len, 1))

    x1 = Embedding(n_categories, embeds_size, name="embed")(input_cols)
    x2 = Dense(embeds_size, activation="linear", name="d1")(input_values)

    x = Add()([x1, x2])

    x = Dense(d_model, activation="relu", name="d2")(x)

    l_encoded = []

    for i in range(num_layers):
        x = EncoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, rate=dropout, name="encoder_%s" % i)(x)
        l_encoded.append(x)

    x_encoded = l_encoded[-1]

    if task in ["classification", "regression"]:

        x = Concatenate()([GlobalMaxPool1D()(a) for a in l_encoded])

        x = Dense(d_model, name="d3", activation="relu")(x)

        for i in range(num_dense_layers):
            x_to_add = Dense(d_model, activation="relu", name="d4_%s" % i)(x)
            x = Add()([x, x_to_add])

    if task == "classification":
        if n_targets > 1:

            out = Dense(n_targets, activation="softmax", name="d5")(x)

        else:

            out = Dense(n_targets, activation="sigmoid", name="d6")(x)

    elif task == "regression":
        out = Dense(n_targets, activation="linear", name="d7")(x)

    elif task == "pretrain":
        out = Dense(n_targets, activation="linear", name="d8")(x_encoded)

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
        model.compile(optimizer=opt, loss=mae)

    model.summary()

    return model
