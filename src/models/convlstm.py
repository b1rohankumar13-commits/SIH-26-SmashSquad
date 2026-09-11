"""TensorFlow/Keras ConvLSTM for lead-specific forecast-bust probabilities."""

from __future__ import annotations


def build_convlstm(
    input_channels: int,
    *,
    sequence_length: int = 9,
    grid_shape: tuple[int, int] = (66, 70),
    hidden_channels: tuple[int, int] = (16, 32),
    kernel_size: int = 3,
    dropout: float = 0.15,
    recurrent_dropout: float = 0.0,
    l2_regularization: float = 1e-4,
):
    """Build a ConvLSTM with one probability per grid cell and lead.

    Inputs use TensorFlow's channels-last layout:
    ``[batch, lead, latitude, longitude, channel]``.  The recurrent stack is
    forward-only. All forecast maps are available at issuance time.
    Output layout is ``[batch, lead, latitude, longitude, 1]``.
    Nine leads remain the default for the time-aligned rainfall pilot;
    pass ten only when ten complete verification windows are available.
    """
    if input_channels <= 0:
        raise ValueError("input_channels must be positive")
    if sequence_length <= 0 or any(size <= 0 for size in grid_shape):
        raise ValueError("sequence length and grid dimensions must be positive")
    if len(hidden_channels) != 2 or any(value <= 0 for value in hidden_channels):
        raise ValueError("hidden_channels must contain two positive filter counts")

    import tensorflow as tf

    regularizer = tf.keras.regularizers.l2(l2_regularization)
    inputs = tf.keras.Input(
        shape=(sequence_length, *grid_shape, input_channels),
        name="forecast_sequence",
    )
    values = tf.keras.layers.ConvLSTM2D(
        filters=hidden_channels[0],
        kernel_size=kernel_size,
        padding="same",
        return_sequences=True,
        activation="tanh",
        recurrent_activation="sigmoid",
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        kernel_regularizer=regularizer,
        recurrent_regularizer=regularizer,
        name="convlstm_1",
    )(inputs)
    values = tf.keras.layers.LayerNormalization(axis=-1, name="channel_norm_1")(values)
    values = tf.keras.layers.ConvLSTM2D(
        filters=hidden_channels[1],
        kernel_size=kernel_size,
        padding="same",
        return_sequences=True,
        activation="tanh",
        recurrent_activation="sigmoid",
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        kernel_regularizer=regularizer,
        recurrent_regularizer=regularizer,
        name="convlstm_2",
    )(values)
    values = tf.keras.layers.LayerNormalization(axis=-1, name="channel_norm_2")(values)
    probabilities = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv2D(1, 1, activation="sigmoid", dtype="float32"),
        name="grid_bust_probability",
    )(values)
    return tf.keras.Model(inputs=inputs, outputs=probabilities, name="rainfall_convlstm")
