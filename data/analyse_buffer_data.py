import os
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt


with open('buffer_5x5.pkl', 'rb') as f:
    data = pkl.load(f)

rms = lambda x: np.sqrt(np.mean(np.square(x)))

obs1_rmses = sorted([rms(item) for item in data['obs1']])

obs1_flat = data['obs1'].flatten()
print(f"Max obs value = {obs1_flat[np.argmax(np.abs(obs1_flat))]}")

fig, ax = plt.subplots()
ax.plot(obs1_rmses)
plt.show()


