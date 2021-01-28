import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow as tf
import pandas as pd

from random_env import RandomEnv

if __name__ == '__main__':
    model_name = os.path.join('models', 'VAE_10x10_012721_1730')

    vae = load_model(model_name)
    #
    n_obs = 10
    latent_dim = 1
    # n_act = 10
    # rm = np.load(os.path.join('random_env_rms', f'random_env_{n_obs}x{n_act}.npy'))
    # env = RandomEnv(n_obs, n_act, rm, noise_std=0.0)
    obses = np.empty((1000, n_obs))
    for i in range(1000):
        # o =env.reset()
        # obses[i] = o
        obses[i] = [np.random.uniform(-1, 1)] * n_obs

    o_reconstructed = tf.squeeze(vae(obses)).numpy()

    print(pd.Series(obses.flatten()).corr(pd.Series(o_reconstructed.flatten())))

    fig, (ax1, ax2) = plt.subplots(2)

    n_samples = 20
    for i in range(n_samples):
        latent_sample = np.array([2 * (i / n_samples) - 1])
        ax1.plot(vae.decoder(latent_sample).numpy().flatten())
        ax2.scatter(0, latent_sample[0])

    fig.suptitle(model_name)
    ax1.set_title('Reconstruction')
    ax1.set_ylim((-1, 1))
    ax2.set_title('Latent variable')
    ax2.set_ylim((-1, 1))
    plt.show()




