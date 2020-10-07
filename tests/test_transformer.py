from deeptabular.transformer import EncoderLayer, Encoder, MultiHeadAttention
import tensorflow as tf


def test_build_encoder_layer():
    encoder = EncoderLayer(d_model=128, num_heads=8, dff=256)
    assert isinstance(encoder, tf.keras.layers.Layer)


def test_build_encoder():
    encoder = Encoder(num_layers=5, d_model=128, num_heads=8, dff=256)
    assert isinstance(encoder, tf.keras.layers.Layer)


def test_build_mha():
    mha = MultiHeadAttention(d_model=128, num_heads=8)
    assert isinstance(mha, tf.keras.layers.Layer)
