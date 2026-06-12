import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_phi(input_dim=1, hidden_units=[64, 64], output_dim=32):

    model = keras.Sequential(name="phi")
    model.add(layers.Input(shape=(input_dim,)))

    for units in hidden_units:
        model.add(layers.Dense(units, activation="relu"))

    model.add(layers.Dense(output_dim, activation="relu"))
    return model


def build_rho(input_dim=32, hidden_units=[64, 64], output_dim=1):
    model = keras.Sequential(name="rho")
    model.add(layers.Input(shape=(input_dim,)))

    for units in hidden_units:
        model.add(layers.Dense(units, activation="relu"))

    model.add(layers.Dense(output_dim, activation="linear"))
    return model


class DeepSet(keras.Model):
    def __init__(self, phi, rho, aggregation="sum"):
        super().__init__()
        self.phi = phi
        self.rho = rho
        self.aggregation = aggregation

    def get_config(self):
        return {
            "phi": keras.layers.serialize(self.phi),
            "rho": keras.layers.serialize(self.rho),
            "aggregation": self.aggregation,
        }
    def call(self, x):
        batch_size = tf.shape(x)[0]
        set_size = tf.shape(x)[1]
        input_dim = tf.shape(x)[2]

        x_flat = tf.reshape(x, (-1, input_dim))

        phi_out = self.phi(x_flat)

        phi_dim = tf.shape(phi_out)[-1]
        phi_out = tf.reshape(phi_out, (batch_size, set_size, phi_dim))

        if self.aggregation == "sum":
            aggregated = tf.reduce_sum(phi_out, axis=1)
        elif self.aggregation == "mean":
            aggregated = tf.reduce_mean(phi_out, axis=1)
        elif self.aggregation == "max":
            aggregated = tf.reduce_max(phi_out, axis=1)

        out = self.rho(aggregated)

        return tf.squeeze(out, axis=-1)


def build_deep_set(
    input_dim=1,
    phi_hidden=[64, 64],
    phi_output=32,
    rho_hidden=[64, 64],
    aggregation="sum"
):
    phi = build_phi(input_dim, phi_hidden, phi_output)
    rho = build_rho(phi_output, rho_hidden, output_dim=1)
    return DeepSet(phi, rho, aggregation)


if __name__ == "__main__":
    import numpy as np

    model = build_deep_set()

    dummy = np.random.uniform(0, 10, size=(4, 5, 1)).astype(np.float32)
    output = model(dummy)

    print("Input shape:", dummy.shape)
    print("Output shape:", output.shape)
    print("Predictions:", output.numpy())
    print("Actual sums:", dummy.squeeze(-1).sum(axis=1))

    shuffled = dummy[:, np.random.permutation(5), :]
    output_shuffled = model(shuffled)
    print("\nPermutation invariance check:")
    print("Original output:  ", output.numpy())
    print("Shuffled output:  ", output_shuffled.numpy())
    print("Max difference:   ", np.abs(output.numpy() - output_shuffled.numpy()).max())