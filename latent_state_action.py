import os
import numpy as np
from datetime import datetime as dt

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean
from tensorflow.keras.callbacks import TensorBoard

import vae_utils
from random_env import RandomEnv

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class Encoder(Model):
    def __init__(self, input_dim, latent_dim, name_prefix, is_variational=True, **kwargs):
        super(Encoder, self).__init__(**kwargs)

        self.encoder = vae_utils.generate_encoder(input_dim=input_dim,
                                        latent_dim=latent_dim,
                                        hidden_layers=[64, 64],
                                        name_prefix=name_prefix,
                                        is_variational=is_variational)

    def call(self, inputs, training=None, mask=None):
        return self.encoder(inputs)


class Decoder(Model):
    def __init__(self, output_dim, latent_dim, name_prefix, **kwargs):
        super(Decoder, self).__init__(**kwargs)

        self.decoder = vae_utils.generate_decoder(output_dim=output_dim,
                                        latent_dim=latent_dim,
                                        hidden_layers=[64, 64],
                                        name_prefix=name_prefix)

    def call(self, inputs, training=None, mask=None):
        return self.decoder(inputs)


class LatentStateAction(Model):
    def __init__(self, n_obs, n_act, latent_dim, warmup_steps, **kwargs):
        super(LatentStateAction, self).__init__(**kwargs)

        self.action_encoder = Encoder(n_act, latent_dim, is_variational=False, name_prefix='action_encoder')
        self.state_encoder = Encoder(n_obs, latent_dim, name_prefix='state_encoder')

        self.action_decoder = Decoder(n_act, latent_dim, name_prefix='action_decoder')
        self.state_decoder = Decoder(n_obs, latent_dim, name_prefix='state_decoder')

        self.action_input = Input(shape=(n_act,))
        self.action_output = self.action_decoder(self.action_encoder(self.action_input))
        self.action_ae_model = Model(self.action_input, self.action_output, name='action_ae')

        self.state_input = Input(shape=(n_obs,))
        self.state_output = self.state_decoder(self.state_encoder(self.state_input)[-1])
        self.state_ae_model = Model(self.state_input, self.state_output, name='state_vae')

        self.model = Model(inputs=[self.action_input, self.state_input],
                           outputs=[self.action_output, self.state_output])
        self.compile(optimizer=Adam(learning_rate=1e-2))

        # Hyper-parameters
        self.warmup_steps = tf.constant(warmup_steps, dtype=tf.int32)
        self.it = tf.Variable(0, dtype=tf.int32)

        # Logging
        self.action_loss_tracker = Mean(name='action_loss')
        self.state_recon_loss_tracker = Mean(name='state_recon_loss')
        self.state_kl_loss_tracker = Mean(name='state_kl_loss')

    def call(self, inputs, training=None, mask=None):
        return self.model([inputs['acts'], inputs['obs1']])

    @property
    def metrics(self):
        return [self.action_loss_tracker,
                self.state_recon_loss_tracker,
                self.state_kl_loss_tracker]

    # @tf.function
    def train_step(self, data):
        data = data[0]

        with tf.GradientTape(persistent=True) as tape:
            # Get state encoder output
            zs_mean, zs_log_var, zs = self.state_encoder(data['obs1'])

            # How good are we at reconstructing state?
            state_reconstruction = self.state_decoder(zs)
            state_reconstruction_loss = tf.reduce_mean(tf.square(data['obs1'] - state_reconstruction))

            # How much regularized is our latent state space?
            state_kl_loss = -0.5 * (1 + zs_log_var - tf.square(zs_mean) - tf.exp(zs_log_var))
            state_kl_loss = tf.reduce_mean(tf.reduce_sum(state_kl_loss, axis=1))

            # Mask state_kl_loss during warm-up phase
            state_kl_loss = tf.cond(pred=tf.math.greater_equal(self.it, self.warmup_steps),
                                    true_fn=lambda : state_kl_loss,
                                    false_fn=lambda : 0.0)

            # Find state VAE total loss
            total_state_loss = state_reconstruction_loss + 4*state_kl_loss

            # Get action encoder output
            za = self.action_encoder(data['acts'])

            # How good are we at reconstructing action?
            action_reconstruction = self.action_decoder(za)
            action_reconstruction_loss = tf.reduce_mean(tf.square(data['acts'] - action_reconstruction))

            # Get encoded next state using current state encoder
            zs_tp1 = self.state_encoder(data['obs2'])[-1]

            # Predict next state assuming canonical representation
            pred_zs_tp1 = tf.add(za, zs_tp1)

            # Get action AE total loss
            latent_matching_loss = tf.reduce_mean(tf.square(zs_tp1 - pred_zs_tp1))

            total_action_loss = latent_matching_loss + action_reconstruction_loss

            self.it.assign_add(1)

        # Get partial derivative wrt to losses and network params
        state_grads = tape.gradient(total_state_loss, self.state_ae_model.trainable_weights)
        action_grads = tape.gradient(total_action_loss, self.action_ae_model.trainable_weights)

        # Apply gradients
        self.optimizer.apply_gradients(zip(state_grads, self.state_ae_model.trainable_weights))
        self.optimizer.apply_gradients(zip(action_grads, self.action_ae_model.trainable_weights))

        # Logging
        self.action_loss_tracker.update_state(total_action_loss)
        self.state_recon_loss_tracker.update_state(state_reconstruction_loss)
        self.state_kl_loss_tracker.update_state(state_kl_loss)

        return {
            'action_loss': self.action_loss_tracker.result(),
            'state_recon_loss': self.state_recon_loss_tracker.result(),
            'state_kl_loss': self.state_kl_loss_tracker.result()
        }

def verify_model(model):
    n_obs = 5
    n_act = 5

    rm = np.load(os.path.join('random_env_rms', 'random_env_5x5_reduced_linear_decay.npy'))
    env = RandomEnv(n_obs, n_act, rm, noise_std=0.0)

    nb_evals = 100
    action_recon_error = np.empty((nb_evals, n_act))
    state_recon_error = np.empty((nb_evals, n_obs))

    env.reset()
    for i in range(nb_evals):
        a = env.action_space.sample()
        o, *_ = env.step(a)

        aa, oo = model({'acts': a.reshape((1, -1)), 'obs1': o.reshape((1, -1))})
        aa = aa.numpy().flatten()
        oo = oo.numpy().flatten()

        action_recon_error[i] = a - aa
        state_recon_error[i] = o - oo

    action_error = np.mean(action_recon_error, axis=0)
    state_error = np.mean(state_recon_error, axis=0)

    rms = lambda x: np.sqrt(np.mean(np.square(x)))
    print(f'Actions diff: {action_error} - RMS: {rms(action_error)}')
    print(f'States diff:  {state_error} - RMS: {rms(state_error)}')


def train_model():
    from replay_buffer import ReplayBuffer

    n_obs = 5
    n_act = 5

    latent_dim = 5

    batch_size = 128



    model_name = f'RLVAE_{dt.now().strftime("%m%d%y-%H%M")}'
    log_dir = os.path.join('logs', model_name)
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    model = LatentStateAction(n_obs, n_act, latent_dim, warmup_steps=int(1e2))

    data = vae_utils.data_gen(batch_size, os.path.join('data', 'buffer_5x5.pkl'))

    model.fit(data, steps_per_epoch=100, epochs=100, callbacks=[tensorboard_callback])

    save_model(model, os.path.join('models', model_name))

    return model


if __name__ == '__main__':
    model = train_model()
    verify_model(model)