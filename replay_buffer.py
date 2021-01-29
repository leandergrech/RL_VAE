import os
import numpy as np
import pickle as pkl


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for NAF_debug agents.
    """

    def __init__(self, obs_dim=None, act_dim=None, max_size=None):
        self.obs1_buf = None
        self.obs2_buf = None
        self.acts_buf = None
        self.rews_buf = None
        self.done_buf = None
        self.ptr, self.size = 0, 0
        self.nb_calls = 0

        if obs_dim is not None and act_dim is not None and max_size is not None:
            self.init_storage(max_size, obs_dim, act_dim)
        else:
            print('Buffer initialised without allocated storage')

    def init_storage(self, max_size, obs_dim, act_dim):
        self.obs1_buf = np.empty([max_size, obs_dim], dtype=np.float64)
        self.obs2_buf = np.empty([max_size, obs_dim], dtype=np.float64)
        self.acts_buf = np.empty([max_size, act_dim], dtype=np.float64)
        self.rews_buf = np.empty(max_size, dtype=np.float64)
        self.done_buf = np.empty(max_size, dtype=np.float64)

        self.max_size = max_size

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
        max_size, n_obs = obs1s.shape
        n_act = acts.shape[-1]
        self.init_storage(max_size, n_obs, n_act)

        for i in range(len(obs1s)):
            self.store(obs1s[i], acts[i], rews[i], obs2s[i], dones[i])