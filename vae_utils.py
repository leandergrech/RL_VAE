import os
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Layer, BatchNormalization
from tensorflow.keras.models import Model

from random_env import RandomEnv
from replay_buffer import ReplayBuffer


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
def generate_encoder(input_dim, latent_dim, hidden_layers, name_prefix=None, is_variational=True):
    x = input_tensor = Input(shape=(input_dim,))

    if name_prefix is None:
        name_prefix = 'z'

    for i, h in enumerate(hidden_layers):
        x = Dense(h, activation='relu', name=f'{name_prefix}_fc{i}')(x)
        x = BatchNormalization()(x)

    if is_variational:
        z_mean = Dense(latent_dim, activation='tanh', name=f'{name_prefix}_mean')(x)
        z_log_var = Dense(latent_dim, activation='tanh', name=f'{name_prefix}_log_var')(x)
        z = Sampling(name=f'{name_prefix}_sample')([z_mean, z_log_var])

        return Model(input_tensor, [z_mean, z_log_var, z], name=f'{name_prefix}_encoder')

    else:
        x = Dense(latent_dim, activation='tanh', name=f'{name_prefix}_latent')(x)
        return Model(input_tensor, x, name=f'{name_prefix}_encoder')


# Builds the decoder network
def generate_decoder(latent_dim, output_dim, hidden_layers, name_prefix=None):
    x = latent_input = Input(shape=(latent_dim,))

    if name_prefix is None:
        name_prefix = 'z'

    for i, h in enumerate(hidden_layers):
        x = Dense(h, activation='relu', name=f'{name_prefix}_fc{i}')(x)
        x = BatchNormalization()(x)

    decoder_output = Dense(output_dim, activation='tanh', name=f'{name_prefix}_out')(x)
    decoder = Model(latent_input, decoder_output, name=f'{name_prefix}_decoder')

    return decoder


# Generates horizontal lines
def data_gen_simple_interactions(b, n):
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


# Generates random walk RandomEnv trajectories
def generate_random_env_test_data():
    rm_name = 'random_env_5x5_reduced_linear_decay.npy'
    rm = np.load(os.path.join('random_env_rms', rm_name))

    n_obs = 5
    n_act = 5

    env = RandomEnv(n_obs, n_act, rm, noise_std=0.0)

    replay = ReplayBuffer(obs_dim=n_obs, act_dim=n_act, max_size=int(5e4))

    max_eps = 100
    max_steps = 50

    for ep_idx  in range(max_eps):
        o1 = env.reset()
        for step_idx in range(max_steps):
            a = env.action_space.sample()
            o2, r, d, _ = env.step(a)

            replay.store(o1, a, r, o2, d)

            o1 = o2

    replay.save_to_pkl(name=f'buffer_{n_obs}x{n_act}.pkl', directory='data')


def data_gen_random_env(b, buffer_data_name):
    replay = ReplayBuffer()
    replay.read_from_pkl(name=buffer_data_name, directory='data')

    while True:
        yield replay.sample_batch(batch_size=b)


def data_gen_random_env_obs1(b, buffer_data_name, scale=1.0):
    gen = data_gen_random_env(b, buffer_data_name)

    while True:
        yield next(gen)['obs1']*scale

def data_gen_random_env_acts(b, buffer_data_name, scale=1.0):
    gen = data_gen_random_env(b, buffer_data_name)

    while True:
        yield next(gen)['acts']*scale


def non_random_policy(batch_size, act_dim=5, dep_dim=3):
    matrix = np.zeros((act_dim, dep_dim))
    matrix[0, 0] = 1
    matrix[1, 1] = 1
    matrix[2, 2] = 1
    matrix[3, :] = 0.5 * matrix[0, :] + 0.5 * matrix[1, :]
    matrix[4, :] = 0.5 * matrix[1, :] + 0.25 * matrix[0, :] + 0.25 * matrix[2, :]

    acts = np.zeros((batch_size, act_dim))
    while True:
        for i in range(batch_size):
            # acts[i] = np.clip(np.dot(matrix, np.random.uniform(-1, 1, dep_dim)), -1, 1)
            acts[i] = np.dot(matrix, np.random.uniform(-1, 1, dep_dim))
        yield acts
