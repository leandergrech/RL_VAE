import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow as tf
import pandas as pd

from random_env import RandomEnv

if __name__ == '__main__':
    model_name = os.path.join('models', 'VAE_10x10_012521_2159')

    vae = load_model(model_name)
    #
    n_obs = 10
    # n_act = 10
    # rm = np.load(os.path.join('random_env_rms', f'random_env_{n_obs}x{n_act}.npy'))
    # env = RandomEnv(n_obs, n_act, rm, noise_std=0.0)
    obses = np.empty((1000, n_obs))
    for i in range(1000):
        # o =env.reset()
        # obses[i] = o
        obses[i] = np.random.uniform(-1, 1, n_obs)

    o_reconstructed = tf.squeeze(vae(obses)).numpy()

    print(pd.Series(obses.flatten()).corr(pd.Series(o_reconstructed.flatten())))

    # fig, ax = plt.subplots()
    #
    # ax.plot(o, label='Real')
    # ax.plot(o_reconstructed, label='Reconstruction')
    # ax.legend(loc='best')
    #
    # plt.show()




