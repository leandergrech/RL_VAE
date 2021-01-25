import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Dense
from tensorflow.keras import Model
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam

tf.get_logger().setLevel('ERROR')


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
def generate_encoder(input_shape, latent_dim=2, hidden_layers=None, latent_name=None):
    x = input_tensor = Input(shape=input_shape)

    if hidden_layers is None:
        hidden_layers = [32, 64, 1024]

    if latent_name is None:
        latent_name = 'z'

    for h in hidden_layers:
        x = Dense(h, activation='relu')(x)

    z_mean = Dense(latent_dim, activation='tanh', name=f'{latent_name}_mean')(x)
    z_log_var = Dense(latent_dim, activation='tanh', name=f'{latent_name}_log_var')(x)
    z = Sampling()([z_mean, z_log_var])

    encoder = Model(input_tensor, [z_mean, z_log_var, z], name=f'{latent_name}_encoder')

    return encoder


# Builds the decoder network
def generate_decoder(latent_dim, output_dim, hidden_layers=None, latent_name=None):
    x = latent_input = Input(shape=(latent_dim,))

    if hidden_layers is None:
        hidden_layers = [1024, 64, 32]

    if latent_name is None:
        latent_name = 'z'

    for h in hidden_layers:
        x = Dense(h, activation='relu')(x)

    decoder_output = Dense(output_dim, activation='tanh')(x)
    decoder = Model(latent_input, decoder_output, name=f'{latent_name}_decoder')

    return decoder


# Defines VAE as a keras Model with a custom train_step method used in
# Model.fit()
class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = Mean(name='total_loss')
        self.reconstruction_loss_tracker = Mean(name='reconstruction_loss')
        self.kl_loss_tracker = Mean(name="kl_loss")

    def call(self, inputs, training=None, mask=None):
        return self.decoder(self.encoder(inputs))

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(tf.square(data - reconstruction))

            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


import os
from random_env import RandomEnv
from datetime import datetime as dt
if __name__ == '__main__':
    n_obs = 10
    n_act = 10
    rm = np.load(os.path.join('random_env_rms', f'random_env_{n_obs}x{n_act}.npy'))
    env = RandomEnv(n_obs, n_act, rm, noise_std=0.0)
    env.reset()

    latent_dim = 10
    hidden_enc = [32, 64, 256]
    hidden_dec = [256, 64, 32]
    latent_name = 'z_a'
    batch_size = 512
    encoder = generate_encoder(input_shape=(n_obs,),
                               latent_dim=latent_dim,
                               hidden_layers=hidden_enc,
                               latent_name=latent_name)
    decoder = generate_decoder(latent_dim=latent_dim,
                               output_dim=n_obs,
                               hidden_layers=hidden_dec,
                               latent_name=latent_name)

    # def generate_interactions(batch_size):
    #     global env
    #
    #     batch = np.empty(shape=(batch_size, env.obs_dimension), dtype=np.float)
    #     while True:
    #         for i in range(batch_size):
    #             a = env.action_space.sample()
    #             o, *_ = env.step(a)
    #             batch[i] = o
    #         yield batch

    def generate_interactions(batch_size):
        batch = np.empty(shape=(batch_size, env.obs_dimension), dtype=np.float)
        while True:
            for i in range(batch_size):
                batch[i] = [np.random.uniform(-1, 1)] * batch.shape[-1] + np.random.randn(batch.shape[-1]) * 1e-2

            yield batch

    data_generator = generate_interactions(batch_size=batch_size)

    vae = VAE(encoder, decoder)
    vae.compile(optimizer=Adam(learning_rate=1e-3))
    vae.fit(data_generator, steps_per_epoch=100, epochs=100)

    save_path = os.path.join('models', f'VAE_{n_obs}x{n_act}_{dt.strftime(dt.now(), "%m%d%y_%H%M")}')
    vae.save(save_path)










