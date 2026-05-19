import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARCHIVE_DIR  = os.path.join(BASE_DIR, "../archive")
CSV_PATH     = os.path.join(BASE_DIR, "data/multilabel_dataset.csv")
MODEL_SAVE   = os.path.join(BASE_DIR, "art_model.keras")
TFLITE_SAVE  = os.path.join(BASE_DIR, "art_model.tflite")
TAGS_FILE    = os.path.join(BASE_DIR, "tags.txt")

IMG_SIZE     = (224, 224)
BATCH_SIZE   = 32
EPOCHS       = 20
THRESHOLD    = 0.4
RANDOM_SEED  = 42

def load_dataset():
    df = pd.read_csv(CSV_PATH)
    tag_cols = [c for c in df.columns if c not in ("filename", "subset")]

    mask = df["filename"].apply(lambda fn: os.path.exists(os.path.join(ARCHIVE_DIR, fn)))
    n_dropped = (~mask).sum()
    if n_dropped:
        print(f"Warning: dropping {n_dropped} rows with missing files")
    df = df[mask].reset_index(drop=True)

    train_df = df[df["subset"] == "train"].reset_index(drop=True)
    val_df   = df[df["subset"] == "validation"].reset_index(drop=True)
    test_df  = df[df["subset"] == "test"].reset_index(drop=True)
    return train_df, val_df, test_df, tag_cols

def build_tf_dataset(dataframe, tag_cols, shuffle):
    paths=[os.path.join(ARCHIVE_DIR, fn) for fn in dataframe["filename"]]
    labels=dataframe[tag_cols].values.astype(np.float32)
    ds= tf.data.Dataset.zip((
        tf.data.Dataset.from_tensor_slices(paths),
        tf.data.Dataset.from_tensor_slices(labels)
    ))
    
    def load(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image.set_shape((None, None, 3))
        image = tf.image.resize(image, IMG_SIZE)
        image = preprocess_input(image)
        return image, label
    
    def augment(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, 0.2)
        image = tf.image.random_contrast(image, 0.75, 1.25)
        image = tf.image.random_saturation(image, 0.75, 1.25)
        image = tf.image.random_hue(image, 0.05)
        return image, label

    ds=ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds=ds.shuffle(2048).map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

def build_model(num_tags):
    base = MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet")
    base.trainable = False
    for layer in base.layers[-10:]:
        layer.trainable = True

    inputs  = layers.Input(shape=(*IMG_SIZE, 3))
    x       = base(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.Dropout(0.6)(x)
    x       = layers.Dense(256, activation="relu")(x)
    x       = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_tags, activation="sigmoid")(x)

    return Model(inputs, outputs)


def save_tags(tag_cols):
    with open(TAGS_FILE, "w") as f:
        f.write("\n".join(tag_cols))
    print(f"Tag list saved: {TAGS_FILE}")
    

def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"], label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].plot(history.history["test_loss"], label="Test Loss")
    axes[0].set_title("Model Loss")
    axes[0].legend()
    
    axes[1].plot(history.history["auc"], label="Train AUC")
    axes[1].plot(history.history["val_auc"], label="Val AUC")
    axes[1].plot(history.history["test_auc"], label="Test AUC")
    axes[1].set_title("Model AUC")
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "training_history.png"))
    print("Training plot saved: training_history.png")
    
def export_tflite(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(TFLITE_SAVE, "wb") as f:
        f.write(tflite_model)
    size_mb = os.path.getsize(TFLITE_SAVE) / 1024 / 1024
    print(f"TFLite model saved: {TFLITE_SAVE}  ({size_mb:.1f} MB)")
    
   
def weighted_bce_fn(pos_weights):
    pw = tf.constant(pos_weights, dtype=tf.float32)

    def loss(y_true, y_pred):
        bce     = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        weights = y_true * pw + (1.0 - y_true)
        return tf.reduce_mean(weights * bce)

    return loss


class TestEvalCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_ds):
        self.test_ds = test_ds

    def on_epoch_end(self, epoch, logs=None):
        results = self.model.evaluate(self.test_ds, verbose=0)
        logs["test_loss"]    = results[0]
        logs["test_auc"]     = results[1]
        logs["test_bin_acc"] = results[2]
        print(f"  test_loss: {results[0]:.4f}  test_auc: {results[1]:.4f}")


def save_test_metrics(loss, auc, bin_acc, f1):
    path = os.path.join(BASE_DIR, "test_metrics.txt")
    with open(path, "w") as f:
        f.write(f"Test Loss:     {loss:.4f}\n")
        f.write(f"Test AUC:      {auc:.4f}\n")
        f.write(f"Test Accuracy: {bin_acc:.4f}\n")
        f.write(f"Macro F1:      {f1:.4f}\n")
    print(f"Test metrics saved: {path}")


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    gpus = tf.config.list_physical_devices("GPU")
    print("GPUs detected:", gpus if gpus else " None - training on CPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    train_df, val_df, test_df, tag_cols = load_dataset()
    train_ds = build_tf_dataset(train_df, tag_cols, shuffle=True)
    val_ds   = build_tf_dataset(val_df,   tag_cols, shuffle=False)
    test_ds  = build_tf_dataset(test_df,  tag_cols, shuffle=False)

    tag_counts  = train_df[tag_cols].sum().values
    pos_weights = (len(train_df) - tag_counts) / np.clip(tag_counts, 1, None)
    pos_weights = np.clip(pos_weights, 1.0, 20.0).astype("float32")

    
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

    callbacks = [
        TestEvalCallback(test_ds),
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

    plot_history(history)

    test_loss, test_auc, test_bin_acc = model.evaluate(test_ds, verbose=1)
    y_pred = model.predict(test_ds, verbose=1)
    y_true = np.concatenate([labels.numpy() for _, labels in test_ds])

    f1 = f1_score(y_true, y_pred > THRESHOLD, average="macro", zero_division=0)
    save_test_metrics(test_loss, test_auc, test_bin_acc, f1)

    export_tflite(model)
    save_tags(tag_cols)

if __name__ == "__main__":
    main()