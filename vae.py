import os
import numpy as np
from datetime import datetime as dt

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras import Model
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam

import vae_utils

tf.get_logger().setLevel('ERROR')

# Defines VAE as a keras Model with a custom train_step method used in
# Model.fit()
class VAE(Model):
    def __init__(self, input_dim, encoder_kwargs, decoder_kwargs, warmup_steps, **kwargs):
        super(VAE, self).__init__(**kwargs)
        # Generate the encoder and decoder networks
        self.encoder = vae_utils.generate_encoder(**encoder_kwargs)
        self.decoder = vae_utils.generate_decoder(**decoder_kwargs)

        self.latent_dim = encoder_kwargs['latent_dim']

        # Combine encoder and decoder to create VAE structure
        self.input_tensor = Input(shape=(input_dim,))

        if encoder_kwargs.get('is_variational', True):
            self.output_tensor = self.decoder(self.encoder(self.input_tensor)[-1])
        else:
            self.output_tensor = self.decoder(self.encoder(self.input_tensor))
        self.vae = Model(self.input_tensor, self.output_tensor, name='vae')

        # We changed call() method so that self is self.vae. We compile the above model.
        self.compile(optimizer=Adam(learning_rate=1e-3))

        # Logging
        self.total_loss_tracker = Mean(name='total_loss')
        self.reconstruction_loss_tracker = Mean(name='reconstruction_loss')
        self.kl_loss_tracker = Mean(name="kl_loss")

        # Hyper-parameters
        self.warmup_steps = tf.constant(warmup_steps, dtype=tf.int32)
        self.it = tf.Variable(0, dtype=tf.int32)

    # The model is actually the self.vae Model inside this class.
    def call(self, inputs, training=None, mask=None):
        return self.vae(inputs)

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    @tf.function
    def train_step(self, data):
        # return self.train_step_deterministic(data)
        return self.train_step_variational(data)

    def train_step_deterministic(self, data):
        with tf.GradientTape() as tape:
            # Get encoder outputs
            z = self.encoder(data)

            # How good are we at reconstruction?
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(tf.square(data - reconstruction))

            self.it.assign_add(1)

        # Gradients w.r.t. the total_loss in the GradientTape
        grads = tape.gradient(reconstruction_loss, self.trainable_weights)

        # Update the network parameters
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # Logging
        # self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        # self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


    def train_step_variational(self, data):
        with tf.GradientTape() as tape:
            # Get encoder outputs
            z_mean, z_log_var, z = self.encoder(data)

            # How good are we at reconstruction?
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(tf.square(data - reconstruction))

            # How much regularized is our latent space?
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

            # Mask kl_loss during warm-up phase
            kl_loss = tf.cond(tf.math.greater_equal(self.it, self.warmup_steps), lambda : kl_loss, lambda : 0.0)

            # We will minimize this loss
            total_loss = reconstruction_loss + kl_loss/10.0

            self.it.assign_add(1)

        # Gradients w.r.t. the total_loss in the GradientTape
        grads = tape.gradient(total_loss, self.vae.trainable_weights)

        # Update the network parameters
        self.optimizer.apply_gradients(zip(grads, self.vae.trainable_weights))

        # Logging
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


def train_vae():
    n_obs = 5
    n_act = 5
    # rm = np.load(os.path.join('random_env_rms', f'random_env_{n_obs}x{n_act}.npy'))
    # env = RandomEnv(n_obs, n_act, rm, noise_std=0.0)
    # env.reset()

    latent_dim = 5
    hidden_enc = [64, 64]
    hidden_dec = [64, 64]
    latent_name = 'z_state'
    batch_size = 64

    encoder_kwargs = dict(input_dim=n_obs,
                          latent_dim=latent_dim,
                          hidden_layers=hidden_enc,
                          name_prefix=latent_name,
                          is_variational=True)

    decoder_kwargs = dict(latent_dim=latent_dim,
                          output_dim=n_obs,
                          hidden_layers=hidden_dec,
                          name_prefix=latent_name)

    vae = VAE(input_dim=n_obs, encoder_kwargs=encoder_kwargs, decoder_kwargs=decoder_kwargs, warmup_steps=1000)

    # data_generator = vae_utils.generate_simple_interactions(b=batch_size, n=n_obs)
    data_generator = vae_utils.data_gen_random_env_obs1(batch_size, 'buffer_5x5.pkl', scale=1/1.9)

    try:
        vae.fit(data_generator, steps_per_epoch=100, epochs=100)
    except KeyboardInterrupt:
        pass

    save_path = os.path.join('models', f'VAE_{n_obs}x{n_act}_{dt.strftime(dt.now(), "%m%d%y_%H%M")}')
    vae.save(save_path)


def evaluate_vae():
    from tensorflow.keras.models import load_model
    model_name = 'VAE_5x5_012921_1105'

    model = load_model(os.path.join('models', model_name))

    gen = vae_utils.data_gen_random_env_obs1(100, 'buffer_5x5.pkl', 1/1.9)

    batch = next(gen)
    recon = model.encoder(batch)[-1] # Get only sample of latent space

    recon = np.mean(np.subtract(batch, recon), axis=0)
    print(recon)

if __name__ == '__main__':
    # train_vae()
    evaluate_vae()










