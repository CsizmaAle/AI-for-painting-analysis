import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

from label_mapping import ALL_TAGS
from generate_multilabel_csv import generate

# ── Config ───────────────────────────────────────────────────
ARCHIVE_DIR  = "archive"
CSV_PATH     = "multilabel_dataset.csv"
MODEL_SAVE   = "art_model.keras"
TFLITE_SAVE  = "art_model.tflite"
TAGS_FILE    = "tags.txt"

IMG_SIZE     = (224, 224)
BATCH_SIZE   = 32
EPOCHS       = 20
THRESHOLD    = 0.4


def prepare_csv():
    if not os.path.exists(CSV_PATH):
        print("multilabel_dataset.csv not found — generating...")
        generate()
    else:
        print(f"Found {CSV_PATH}")


def load_data():
    df = pd.read_csv(CSV_PATH)
    tag_cols = [c for c in df.columns if c not in ("filename", "subset")]

    train_df = df[df["subset"] == "train"].reset_index(drop=True)
    val_df   = df[df["subset"] == "test"].reset_index(drop=True)

    print(f"Train: {len(train_df)}  |  Val: {len(val_df)}  |  Tags: {len(tag_cols)}")
    return train_df, val_df, tag_cols


def make_dataset(dataframe, tag_cols, shuffle):
    paths  = [os.path.join(ARCHIVE_DIR, f) for f in dataframe["filename"]]
    labels = dataframe[tag_cols].values.astype("float32")

    ds = tf.data.Dataset.zip((
        tf.data.Dataset.from_tensor_slices(paths),
        tf.data.Dataset.from_tensor_slices(labels),
    ))

    def load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = preprocess_input(img)
        return img, label

    def augment(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.15)
        img = tf.image.random_contrast(img, 0.85, 1.15)
        return img, label

    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(2048).map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def build_model(num_tags):
    base = MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet")
    base.trainable = False
    for layer in base.layers[-30:]:
        layer.trainable = True

    inputs  = layers.Input(shape=(*IMG_SIZE, 3))
    x       = base(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.Dropout(0.3)(x)
    x       = layers.Dense(256, activation="relu")(x)
    x       = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_tags, activation="sigmoid")(x)

    return Model(inputs, outputs)


def weighted_bce_fn(pos_weights):
    pw = tf.constant(pos_weights, dtype=tf.float32)

    def loss(y_true, y_pred):
        bce     = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        weights = y_true * pw + (1.0 - y_true)
        return tf.reduce_mean(weights * bce)

    return loss


def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"],     label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].legend()

    axes[1].plot(history.history["auc"],     label="train")
    axes[1].plot(history.history["val_auc"], label="val")
    axes[1].set_title("AUC"); axes[1].legend()

    plt.tight_layout()
    plt.savefig("training_history.png")
    print("Training plot saved: training_history.png")


def export_tflite(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(TFLITE_SAVE, "wb") as f:
        f.write(tflite_model)
    size_mb = os.path.getsize(TFLITE_SAVE) / 1024 / 1024
    print(f"TFLite model saved: {TFLITE_SAVE}  ({size_mb:.1f} MB)")


def save_tags(tag_cols):
    with open(TAGS_FILE, "w") as f:
        f.write("\n".join(tag_cols))
    print(f"Tag list saved: {TAGS_FILE}")


if __name__ == "__main__":
    # GPU check
    gpus = tf.config.list_physical_devices("GPU")
    print("GPUs detected:", gpus if gpus else "None — training on CPU")

    # Step 1 — CSV
    prepare_csv()

    # Step 2 — Data
    train_df, val_df, tag_cols = load_data()
    train_ds = make_dataset(train_df, tag_cols, shuffle=True)
    val_ds   = make_dataset(val_df,   tag_cols, shuffle=False)

    # Step 3 — Class weights
    tag_counts  = train_df[tag_cols].sum().values
    pos_weights = (len(train_df) - tag_counts) / np.clip(tag_counts, 1, None)
    pos_weights = np.clip(pos_weights, 1.0, 20.0).astype("float32")

    # Step 4 — Model
    model = build_model(len(tag_cols))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=weighted_bce_fn(pos_weights),
        metrics=[
            tf.keras.metrics.AUC(multi_label=True, name="auc"),
            tf.keras.metrics.BinaryAccuracy(threshold=THRESHOLD, name="bin_acc"),
        ]
    )
    model.summary()

    # Step 5 — Train
    callbacks = [
        EarlyStopping(monitor="val_auc", patience=4, mode="max", restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_auc", factor=0.5, patience=2, mode="max", min_lr=1e-6),
        ModelCheckpoint(MODEL_SAVE, monitor="val_auc", mode="max", save_best_only=True),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # Step 6 — Evaluate
    plot_history(history)

    y_true, y_pred = [], []
    for imgs, labels in val_ds:
        y_pred.append(model.predict(imgs, verbose=0))
        y_true.append(labels.numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    f1 = f1_score(y_true, y_pred > THRESHOLD, average="macro", zero_division=0)
    print(f"Macro F1 @ {THRESHOLD}: {f1:.4f}")

    # Step 7 — Export
    export_tflite(model)
    save_tags(tag_cols)
