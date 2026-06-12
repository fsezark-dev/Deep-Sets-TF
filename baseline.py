import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

from data.generator import generate_batch
from model.deep_set import build_deep_set

def build_naive_mlp(set_size, input_dim=1, hidden_units=[64, 64, 64]):
    model = keras.Sequential(name="naive_mlp")
    model.add(layers.Input(shape=(set_size * input_dim,)))
    for units in hidden_units:
        model.add(layers.Dense(units, activation="relu"))
    model.add(layers.Dense(1, activation="linear"))
    return model


def train_mlp(set_size=50, min_val=0.0, max_val=1000.0, epochs=50, steps_per_epoch=100, batch_size=64, learning_rate=1e-3):
    model = build_naive_mlp(set_size)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = keras.losses.MeanSquaredError()
    train_losses, val_losses = [], []

    print(f"Training Naive MLP on sets of size {set_size}, range [{min_val}, {max_val}]")

    for epoch in range(epochs):
        epoch_losses = []
        for _ in range(steps_per_epoch):
            X_batch, y_batch = generate_batch(batch_size, set_size, min_val, max_val)
            X_batch = X_batch.astype(np.float32).reshape(batch_size, -1) / max_val
            y_batch = y_batch.astype(np.float32) / max_val

            with tf.GradientTape() as tape:
                predictions = tf.squeeze(model(X_batch, training=True), axis=-1)
                loss = loss_fn(y_batch, predictions)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_losses.append(loss.numpy())

        X_val, y_val = generate_batch(512, set_size, min_val, max_val)
        X_val = X_val.astype(np.float32).reshape(512, -1) / max_val
        y_val = y_val.astype(np.float32) / max_val
        val_preds = tf.squeeze(model(X_val, training=False), axis=-1)
        val_loss = loss_fn(y_val, val_preds).numpy()

        train_loss = np.mean(epoch_losses)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} — "
                  f"Train Loss: {train_loss:.6f} — Val Loss: {val_loss:.6f}")

    return model, train_losses, val_losses


def train_deep_sets(set_size=50, min_val=0.0, max_val=1000.0, epochs=50, steps_per_epoch=100, batch_size=64, learning_rate=1e-3):
    model = build_deep_set(input_dim=1)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = keras.losses.MeanSquaredError()
    train_losses, val_losses = [], []
    scale = max_val

    print(f"\nTraining Deep Sets on sets of size {set_size}, range [{min_val}, {max_val}]")

    for epoch in range(epochs):
        epoch_losses = []
        for _ in range(steps_per_epoch):
            X_batch, y_batch = generate_batch(batch_size, set_size, min_val, max_val)
            X_batch = X_batch.astype(np.float32) / scale
            y_batch = y_batch.astype(np.float32) / scale

            with tf.GradientTape() as tape:
                predictions = model(X_batch, training=True)
                loss = loss_fn(y_batch, predictions)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_losses.append(loss.numpy())

        X_val, y_val = generate_batch(512, set_size, min_val, max_val)
        X_val = X_val.astype(np.float32) / scale
        y_val = y_val.astype(np.float32) / scale
        val_preds = model(X_val, training=False)
        val_loss = loss_fn(y_val, val_preds).numpy()

        train_loss = np.mean(epoch_losses)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} — "
                  f"Train Loss: {train_loss:.6f} — Val Loss: {val_loss:.6f}")

    return model, train_losses, val_losses, scale

if __name__ == "__main__":
    SET_SIZE = 50
    MIN_VAL = 0.0
    MAX_VAL = 1000.0

    mlp_model, mlp_train_losses, mlp_val_losses = train_mlp(
        set_size=SET_SIZE, min_val=MIN_VAL, max_val=MAX_VAL
    )
    ds_model, ds_train_losses, ds_val_losses, scale = train_deep_sets(
        set_size=SET_SIZE, min_val=MIN_VAL, max_val=MAX_VAL
    )

    print("\n===== PERMUTATION INVARIANCE COMPARISON =====")
    X_check, _ = generate_batch(100, set_size=SET_SIZE, min_val=MIN_VAL, max_val=MAX_VAL)
    X_check = X_check.astype(np.float32)
    X_shuffled = X_check[:, np.random.permutation(SET_SIZE), :]

    ds_orig = ds_model(X_check / scale, training=False).numpy()
    ds_shuf = ds_model(X_shuffled / scale, training=False).numpy()
    ds_diff = np.abs(ds_orig - ds_shuf).mean()

    mlp_orig = tf.squeeze(mlp_model(X_check.reshape(100, -1) / MAX_VAL, training=False)).numpy()
    mlp_shuf = tf.squeeze(mlp_model(X_shuffled.reshape(100, -1) / MAX_VAL, training=False)).numpy()
    mlp_diff = np.abs(mlp_orig - mlp_shuf).mean()

    print(f"Deep Sets  : mean output diff after shuffle: {ds_diff:.8f}")
    print(f"Naive MLP  : mean output diff after shuffle: {mlp_diff:.4f}")
    print(f"Deep Sets permutation invariant: {ds_diff < 1e-5}")
    print(f"Naive MLP  permutation invariant: {mlp_diff < 1e-5}")

    print("\nSaving plots...")
    os.makedirs("results", exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(ds_train_losses, label="Deep Sets Train", color="blue")
    axes[0].plot(ds_val_losses, label="Deep Sets Val", color="blue", linestyle="--")
    axes[0].plot(mlp_train_losses, label="Naive MLP Train", color="orange")
    axes[0].plot(mlp_val_losses, label="Naive MLP Val", color="orange", linestyle="--")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss (normalized)")
    axes[0].set_title("Training Convergence: Deep Sets vs Naive MLP")
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_yscale("log")

    axes[1].bar(["Deep Sets", "Naive MLP"], [ds_diff, mlp_diff], color=["blue", "orange"], width=0.4)
    axes[1].set_ylabel("Mean Output Difference After Shuffle")
    axes[1].set_title("Permutation Sensitivity")
    axes[1].grid(True, axis="y")
    for i, v in enumerate([ds_diff, mlp_diff]):
        axes[1].text(i, v + mlp_diff * 0.02, f"{v:.4f}", ha="center", fontsize=11)

    plt.tight_layout()
    plt.savefig("results/comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved to results/comparison.png")