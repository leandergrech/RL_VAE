import os
import numpy as np
from datetime import datetime as dt

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras import Model
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam

from vae_utils import generate_encoder, generate_decoder, generate_simple_interactions

tf.get_logger().setLevel('ERROR')


# Defines VAE as a keras Model with a custom train_step method used in
# Model.fit()
class VAE(Model):
    def __init__(self, input_dim, encoder_kwargs, decoder_kwargs, warmup_steps, **kwargs):
        super(VAE, self).__init__(**kwargs)
        # Generate the encoder and decoder networks
        self.encoder = generate_encoder(**encoder_kwargs)
        self.decoder = generate_decoder(**decoder_kwargs)

        self.latent_dim = encoder_kwargs['latent_dim']

        # Combine encoder and decoder to create VAE structure
        self.input_tensor = Input(shape=(input_dim,))
        self.output_tensor = self.decoder(self.encoder(self.input_tensor)[-1])
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
            total_loss = reconstruction_loss + kl_loss/50.0

            self.it.assign_add(1)

        # Gradients w.r.t. the total_loss in the GradientTape
        grads = tape.gradient(total_loss, self.trainable_weights)

        # Update the network parameters
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # Logging
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

def test_vae():
    n_obs = 10
    n_act = 10
    # rm = np.load(os.path.join('random_env_rms', f'random_env_{n_obs}x{n_act}.npy'))
    # env = RandomEnv(n_obs, n_act, rm, noise_std=0.0)
    # env.reset()

    latent_dim = 1
    hidden_enc = [32, 64, 256]
    hidden_dec = [256, 64, 32]
    latent_name = 'z_a'
    batch_size = 32

    encoder_kwargs = dict(input_dim=n_obs,
                               latent_dim=latent_dim,
                               hidden_layers=hidden_enc,
                               name_prefix=latent_name)

    decoder_kwargs = dict(latent_dim=latent_dim,
                               output_dim=n_obs,
                               hidden_layers=hidden_dec,
                               name_prefix=latent_name)

    data_generator = generate_simple_interactions(b=batch_size, n=n_obs)

    vae = VAE(input_dim=n_obs, encoder_kwargs=encoder_kwargs, decoder_kwargs=decoder_kwargs, warmup_steps=600)

    try:
        vae.fit(data_generator, steps_per_epoch=1000, epochs=40)
    except KeyboardInterrupt:
        pass

    save_path = os.path.join('models', f'VAE_{n_obs}x{n_act}_{dt.strftime(dt.now(), "%m%d%y_%H%M")}')
    vae.save(save_path)

if __name__ == '__main__':
    test_vae()










