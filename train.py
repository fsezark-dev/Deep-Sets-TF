import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
from tensorflow import keras

from data.generator import generate_batch
from model.deep_set import build_deep_set

def train(
    set_size=50,
    min_val=0.0,
    max_val=1000.0,
    normalize=True,
    phi_hidden=[64, 64],
    phi_output=32,
    rho_hidden=[64, 64],
    aggregation="sum",
    epochs=50,
    steps_per_epoch=100,
    batch_size=64,
    learning_rate=1e-3,
):
    model = build_deep_set(
        input_dim=1,
        phi_hidden=phi_hidden,
        phi_output=phi_output,
        rho_hidden=rho_hidden,
        aggregation=aggregation,
    )

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = keras.losses.MeanSquaredError()

    train_losses = []
    val_losses = []

    print(f"Training on sets of size {set_size}, range [{min_val}, {max_val}]")
    print(f"Aggregation: {aggregation}\n")

    for epoch in range(epochs):
        epoch_losses = []

        for _ in range(steps_per_epoch):
            X_batch, y_batch = generate_batch(batch_size, set_size, min_val, max_val)
            X_batch = X_batch.astype(np.float32)
            y_batch = y_batch.astype(np.float32)

            if normalize:
                X_batch = X_batch / max_val
                y_batch = y_batch / max_val

            with tf.GradientTape() as tape:
                predictions = model(X_batch, training=True)
                loss = loss_fn(y_batch, predictions)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_losses.append(loss.numpy())

        X_val, y_val = generate_batch(512, set_size, min_val, max_val)
        X_val = X_val.astype(np.float32)
        y_val = y_val.astype(np.float32)

        if normalize:
            X_val = X_val / max_val
            y_val = y_val / max_val

        val_preds = model(X_val, training=False)
        val_loss = loss_fn(y_val, val_preds).numpy()

        train_loss = np.mean(epoch_losses)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch+1:3d}/{epochs} — "
                f"Train Loss: {train_loss:.6f} — "
                f"Val Loss: {val_loss:.6f}"
            )

    return model, train_losses, val_losses, max_val if normalize else None


if __name__ == "__main__":
    model, train_losses, val_losses, scale = train()

    print("\n===== EVALUATION =====")
    X_test, y_test = generate_batch(8, set_size=50, min_val=0.0, max_val=1000.0)
    X_test = X_test.astype(np.float32)
    preds = model(X_test / scale, training=False).numpy() * scale

    for i in range(8):
        elements = X_test[i].squeeze()[:5]
        print(
            f"Set (first 5): {np.array2string(elements, precision=1, separator=', ')} ... "
            f"| True: {y_test[i]:.2f} "
            f"| Pred: {preds[i]:.2f} "
            f"| Error: {abs(y_test[i] - preds[i]):.2f}"
        )

    print("\n===== PERMUTATION INVARIANCE =====")
    X_check, _ = generate_batch(4, set_size=50, min_val=0.0, max_val=1000.0)
    X_check = X_check.astype(np.float32) / scale
    X_shuffled = X_check[:, np.random.permutation(50), :]

    out_original = model(X_check, training=False).numpy()
    out_shuffled = model(X_shuffled, training=False).numpy()
    max_diff = np.abs(out_original - out_shuffled).max()

    print(f"Max output difference after shuffling: {max_diff:.8f}")
    print("Permutation invariant:", max_diff < 1e-5)

    model.save("deep_set_model.keras")
    print("\nModel saved to deep_set_model.keras")