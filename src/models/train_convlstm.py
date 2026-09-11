"""Train the TensorFlow gridded spatiotemporal branch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

from src.models.convlstm import build_convlstm


@tf.keras.utils.register_keras_serializable(package="sih")
class PositiveWeightedBinaryCrossentropy(tf.keras.losses.Loss):
    """Binary cross-entropy with a training-fold weight on each positive lead."""

    def __init__(self, positive_weight: float, name="positive_weighted_bce", **kwargs):
        kwargs.setdefault("reduction", "mean_with_sample_weight")
        super().__init__(name=name, **kwargs)
        if not np.isfinite(positive_weight) or positive_weight <= 0:
            raise ValueError("positive_weight must be positive")
        self.positive_weight = float(positive_weight)

    def call(self, y_true, y_pred):
        elementwise_loss = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        elementwise_weight = tf.where(
            tf.equal(y_true, 1.0),
            tf.cast(self.positive_weight, elementwise_loss.dtype),
            tf.cast(1.0, elementwise_loss.dtype),
        )
        return tf.reduce_mean(elementwise_loss * elementwise_weight, axis=-1)

    def get_config(self):
        return {**super().get_config(), "positive_weight": self.positive_weight}


def positive_class_weight(labels) -> float:
    labels = np.asarray(labels, dtype=np.float32)
    valid = ~np.isnan(labels)
    if not np.isin(labels[valid], [0, 1]).all():
        raise ValueError("Labels must be binary, with NaN only for unverified cells")
    positives = int(np.sum(labels[valid] == 1))
    negatives = int(np.sum(labels[valid] == 0))
    if positives == 0 or negatives == 0:
        raise ValueError("Training labels must contain both bust and non-bust cases")
    return float(negatives / positives)


def _mixed_precision_policy(mode: str):
    import tensorflow as tf

    if mode not in {"auto", "on", "off"}:
        raise ValueError("mixed_precision must be auto, on, or off")
    has_gpu = bool(tf.config.list_physical_devices("GPU"))
    enabled = mode == "on" or (mode == "auto" and has_gpu)
    tf.keras.mixed_precision.set_global_policy(
        "mixed_float16" if enabled else "float32"
    )
    return enabled


def train_convlstm(
    train_inputs,
    train_labels,
    validation_inputs,
    validation_labels,
    config: dict,
    *,
    checkpoint_path: Path | None = None,
):
    """Fit spatial labels; NaN cells are excluded from loss and metrics.

    Labels have shape batch×lead×latitude×longitude×1. Inputs must already
    be standardized/imputed using training-only statistics. The caller owns
    chronological splitting and must keep verification windows disjoint.
    """
    train_inputs = np.asarray(train_inputs, dtype=np.float32)
    validation_inputs = np.asarray(validation_inputs, dtype=np.float32)
    train_labels = np.asarray(train_labels, dtype=np.float32)
    validation_labels = np.asarray(validation_labels, dtype=np.float32)
    if train_inputs.ndim != 5 or validation_inputs.ndim != 5:
        raise ValueError("ConvLSTM inputs must be batch×lead×latitude×longitude×channel")
    if train_labels.shape != (*train_inputs.shape[:4], 1):
        raise ValueError("Training labels must have shape batch×lead×latitude×longitude×1")
    if validation_labels.shape != (*validation_inputs.shape[:4], 1):
        raise ValueError("Validation labels must have shape batch×lead×latitude×longitude×1")
    if train_inputs.shape[1:] != validation_inputs.shape[1:]:
        raise ValueError("Training and validation tensor shapes do not match")
    if not np.isfinite(train_inputs).all() or not np.isfinite(validation_inputs).all():
        raise ValueError("Inputs must be finite after training-only standardization/imputation")
    if int(config.get("sequence_length", train_inputs.shape[1])) != train_inputs.shape[1]:
        raise ValueError("Configured lead count does not match the available data")
    class_weight = positive_class_weight(train_labels)
    positive_class_weight(validation_labels)
    train_mask = (~np.isnan(train_labels[..., 0])).astype(np.float32)
    validation_mask = (~np.isnan(validation_labels[..., 0])).astype(np.float32)
    train_labels = np.nan_to_num(train_labels, nan=0.0)
    validation_labels = np.nan_to_num(validation_labels, nan=0.0)

    tf.keras.utils.set_random_seed(int(config.get("random_state", 42)))
    _mixed_precision_policy(str(config.get("mixed_precision", "auto")))
    sequence_length, latitude_cells, longitude_cells, channels = train_inputs.shape[1:]
    model = build_convlstm(
        channels,
        sequence_length=sequence_length,
        grid_shape=(latitude_cells, longitude_cells),
        hidden_channels=tuple(config["hidden_channels"]),
        kernel_size=int(config["kernel_size"]),
        dropout=float(config.get("dropout", 0.15)),
        recurrent_dropout=float(config.get("recurrent_dropout", 0.0)),
        l2_regularization=float(config.get("l2_regularization", 1e-4)),
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(float(config["learning_rate"])),
        loss=PositiveWeightedBinaryCrossentropy(class_weight),
        weighted_metrics=[
            tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
            tf.keras.metrics.AUC(curve="ROC", name="roc_auc"),
            tf.keras.metrics.MeanSquaredError(name="brier_score"),
        ],
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_pr_auc",
            mode="max",
            patience=int(config.get("early_stopping_patience", 12)),
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            patience=int(config.get("reduce_lr_patience", 5)),
            factor=0.5,
            min_lr=1e-6,
        ),
    ]
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor="val_pr_auc",
                mode="max",
                save_best_only=True,
            )
        )
    history = model.fit(
        train_inputs,
        train_labels,
        sample_weight=train_mask,
        validation_data=(validation_inputs, validation_labels, validation_mask),
        batch_size=int(config["batch_size"]),
        epochs=int(config.get("epochs", 100)),
        callbacks=callbacks,
        shuffle=False,
        verbose=int(config.get("verbose", 1)),
    )
    return model, history
