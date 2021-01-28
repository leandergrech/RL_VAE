import os
import numpy as np
import matplotlib.pyplot as plt


def check_rm():
    rm_name = os.path.join('random_env_rms', 'random_env_10x10.npy')

    rm = np.load(rm_name)

    u, s, vh = np.linalg.svd(rm)

    fig, ax = plt.subplots()
    ax.scatter(range(len(s)), s)

    ax.set_title(rm_name)
    ax.set_xlabel('Singular value index')
    ax.grid(True)
    plt.show()


def reduce_rm():
    rm_name = os.path.join('random_env_rms', 'random_env_5x5.npy')
    rm = np.load(rm_name)

    u, s, vh = np.linalg.svd(rm)

    half_index = 2

    weights = [1/(i+1) for i in range(rm.shape[0])]
    # weights = [np.exp(-i/half_index) for i in range(len(s))]
    sw = s * weights
    s_diag = np.diag(sw)

    rm_name = os.path.join('random_env_rms', 'random_env_5x5_reduced_linear_decay.npy')

    new_rm = u.dot(s_diag).dot(vh)
    np.save(rm_name, new_rm)

    fig, ax = plt.subplots()
    ax.scatter(range(len(s)), s)
    ax.scatter(range(len(s)), sw)

    ax.set_title(rm_name)
    ax.set_xlabel('Singular value index')
    ax.grid(True)
    plt.show()


def compare_rm_performance():
    from random_env import RandomEnv

    rm1 = np.load(os.path.join('random_env_rms', 'random_env_10x10.npy'))
    rm2 = np.load(os.path.join('random_env_rms', 'random_env_10x10_reduced_half_index_2.npy'))

    print(rm1.shape)
    print(rm2.shape)

    env1 = RandomEnv(10, 10, rm1)
    env2 = RandomEnv(10, 10, rm2)

    o = env1.reset()
    env2.reset(o)

    fig, (ax1, ax2) = plt.subplots(2, num=2)
    o1_line, = ax1.plot(np.zeros(10))
    o2_line, = ax2.plot(np.zeros(10))

    plt.ion()

    for _ in range(10):
        a = env1.action_space.sample()
        o1, *_ = env1.step(a)
        o2, *_ = env2.step(a)

        o1_line.set_ydata(o1)
        o2_line.set_ydata(o2)

        if not plt.fignum_exists(1):
            break
        plt.draw()
        plt.pause(1)

if __name__ == '__main__':
    # compare_rm_performance()
    reduce_rm()