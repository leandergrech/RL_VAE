import os
import numpy as np
import pickle as pkl


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for NAF_debug agents.
    """

    def __init__(self, obs_dim, act_dim, max_size):
        self.obs1_buf = np.empty([max_size, obs_dim], dtype=np.float64)
        self.obs2_buf = np.empty([max_size, obs_dim], dtype=np.float64)
        self.acts_buf = np.empty([max_size, act_dim], dtype=np.float64)
        self.rews_buf = np.empty(max_size, dtype=np.float64)
        self.done_buf = np.empty(max_size, dtype=np.float64)
        self.ptr, self.size, self.max_size = 0, 0, max_size

        self.nb_calls = 0

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        if self.size < batch_size:
            idxs = np.arange(self.size)
        else:
            idxs = np.random.randint(0, self.size, size=batch_size)

        self.nb_calls += 1

        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def save_to_pkl(self, name, directory):
        idxs = np.arange(self.size)
        buffer_data = dict(obs1=self.obs1_buf[idxs],
                           obs2=self.obs2_buf[idxs],
                           acts=self.acts_buf[idxs],
                           rews=self.rews_buf[idxs],
                           done=self.done_buf[idxs])
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(os.path.join(directory, name), 'wb') as f:
            pkl.dump(buffer_data, f)

    def read_from_pkl(self, name, directory):
        with open(os.path.join(directory, name), 'rb') as f:
            buffer_data = pkl.load(f)

        obs1s, obs2s, acts, rews, dones = [buffer_data[key] for key in buffer_data]
        for i in range(len(obs1s)):
            self.store(obs1s[i], acts[i], rews[i], obs2s[i], dones[i])