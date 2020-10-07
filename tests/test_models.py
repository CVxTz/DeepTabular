from deeptabular.models import transformer_tabular
import tensorflow as tf


def test_build_model():
    model = transformer_tabular(
        n_categories=100,
        n_targets=10,
        embeds_size=10,
        num_layers=4,
        num_dense_layers=4,
        d_model=64,
        num_heads=4,
        task="classification",
        lr=0.0001,
        dropout=0.01,
        seq_len=None,
        att_heads=2,
    )
    assert isinstance(model, tf.keras.models.Model)
