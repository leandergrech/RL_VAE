import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Layer
from tensorflow.keras.models import Model


# Creates a sampling layer
class Sampling(Layer):
    """
    Uses (z_mean, z_log_var) to sample z, the vector encoding a digit.

    Notes:
        I think that the variance is stored in log scale just so that the
        sampling can be easily done since we replace a square root of the
        variance by a multiplication of 0.5 in the log scale followed by
        an exponentiation.
    """
    def call(self, inputs, **kwargs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[-1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))

        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# Builds the encoder network
def generate_encoder(input_dim, latent_dim=2, hidden_layers=None, name_prefix=None, is_variational=True):
    x = input_tensor = Input(shape=(input_dim,))

    if hidden_layers is None:
        hidden_layers = [32, 64, 1024]

    if name_prefix is None:
        name_prefix = 'z'

    for i, h in enumerate(hidden_layers):
        x = Dense(h, activation='relu', name=f'{name_prefix}_fc{i}')(x)

    if is_variational:
        z_mean = Dense(latent_dim, activation='tanh', name=f'{name_prefix}_mean')(x)
        z_log_var = Dense(latent_dim, activation='tanh', name=f'{name_prefix}_log_var')(x)
        z = Sampling(name=f'{name_prefix}_sample')([z_mean, z_log_var])

        encoder = Model(input_tensor, [z_mean, z_log_var, z], name=f'{name_prefix}_encoder')

        return encoder
    else:
        x = Dense(latent_dim, activation='tanh', name=f'{name_prefix}_latent')(x)
        return Model(input_tensor, x, name=f'{name_prefix}_encoder')


# Builds the decoder network
def generate_decoder(latent_dim, output_dim, hidden_layers=None, name_prefix=None):

    x = latent_input = Input(shape=(latent_dim,))

    if hidden_layers is None:
        hidden_layers = [1024, 64, 32]

    if name_prefix is None:
        name_prefix = 'z'

    for i, h in enumerate(hidden_layers):
        x = Dense(h, activation='relu', name=f'{name_prefix}_fc{i}')(x)

    decoder_output = Dense(output_dim, activation='tanh', name=f'{name_prefix}_out')(x)
    decoder = Model(latent_input, decoder_output, name=f'{name_prefix}_decoder')

    return decoder


def generate_simple_interactions(b, n):
    """
    Creates a generator which outputs horizontal lines at random y-intercepts.
    :param b: batch size, first dimension of
    :param n:
    :return:
    """
    batch = np.empty(shape=(b, n), dtype=np.float)
    while True:
        for i in range(b):
            batch[i] = [np.random.uniform(-1, 1)] * n  # + np.random.randn(batch.shape[-1]) * 1e-2

        yield batch