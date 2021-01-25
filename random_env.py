import os, shutil
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import gym
from gym.spaces import Box


class RandomEnv(gym.Env):
	# in normalised space
	STATE_INIT_STD = 0.5
	GOAL = 0.05

	ACT_LIM = 0.1
	OBS_LIM = 5

	# reward_accumulated_limit = -10
	REW_SCALE = 0.1

	def __init__(self, n_obs, n_act, rm, noise_std=None):
		self.noise_std = noise_std
		self._last_action = None
		self._current_state = None
		self._prev_state = None
		self._reward = None
		self._reward_thresh = self.objective([RandomEnv.GOAL]*n_obs)
		self._reward_deque = deque(maxlen=5)

		self.rm = rm
		self.pi = np.linalg.pinv(rm)
		assert np.isfinite(self.pi).all(), 'Response matrix passed is not invertible'


		self.obs_dimension = n_obs
		self.act_dimension = n_act

		self.observation_space = Box(low=-np.ones(self.obs_dimension), high=np.ones(self.obs_dimension), dtype=np.float)
		self.action_space = Box(low=-RandomEnv.ACT_LIM, high=RandomEnv.ACT_LIM, shape=(n_act,), dtype=np.float)

		self.it = 0

	def reset(self, init_state=None):
		o = self._reset(init_state)
		return o

	def _reset(self, init_state=None):
		if init_state is None:
			# self._current_state = np.random.normal(0, self.Q_init_std, self.obs_dimension)
			self._current_state = self.observation_space.sample()
		else:
			self._current_state = init_state
		self._prev_state = self._current_state
		self._last_action = np.zeros(self.act_dimension)
		self.it = 0

		return self._current_state

	def step(self, action):
		if self.noise_std is not None:
			noise_std = self.noise_std

		self._last_action = action

		# # Convert action to rm units and add noise
		# action += np.random.normal(scale=noise_std, size=self.act_dimension)
		# action_scaled = np.multiply(action, RandomEnv.ACT_SCALE)

		# Calculate real trim
		trim_state = self.rm.T.dot(action)
		# Normalise trim obtained from action
		# trim_state = np.divide(trim_state, self.Q_limit_hz)

		self._prev_state = self._current_state
		self._current_state = self._current_state + trim_state

		self._current_state = np.clip(self._current_state, -self.OBS_LIM, self.OBS_LIM)

		self._reward = self.objective(self._current_state)
		self._reward_deque.append(self._reward)

		done = self.is_done()

		self.it += 1

		return self._current_state, self._reward, done, {}

	def objective(self, state):
		# state_reward = -np.sqrt(np.mean(np.power(self._current_state, 2)))
		# action_reward = -np.sqrt(np.mean(np.power(self._last_action, 2))) / 5
		state_reward = -np.square(np.sum(np.abs(state)) + 1)
		# for s in state:
		#     if np.abs(s) > 1:
		#         state_reward -= np.abs(s)
		# action_reward = -np.sum(np.abs(self._last_action)) / self.act_dimension

		return state_reward * RandomEnv.REW_SCALE #+ action_reward

	def is_done(self):
		# Reach goal
		if np.mean(self._reward_deque) > self._reward_thresh:# or np.max(np.abs(self._current_state)) > RandomEnv.OBS_LIM:
			done = True
		else:
			done = False
		return done

	def get_optimal_action(self, state):
		action_optimal = self.pi.T.dot(state)

		return -action_optimal


def main():
	ANIM_DELAY = 0.1
	n_obs = 5
	n_act = 5
	rm = np.random.uniform(-1.0, 1.0, (n_obs, n_act))

	env = RandomEnv(n_obs, n_act, rm=rm)
	opt_env = RandomEnv(n_obs, n_act, rm=rm)

	n_obs = env.obs_dimension
	n_act = env.act_dimension

	f, axes = plt.subplots(3, 2)
	# fig, ((ax1, ax4), (ax2, ax5), (ax3, ax6)) = plt.subplots(3, 2, num=1)
	fig, ((ax1, ax4), (ax2, ax5)) = plt.subplots(2, 2, num=1)
	fig2, axr = plt.subplots(num=2)
	o_x = range(n_obs)
	a_x = range(n_act)

	# ax1
	o_bars = ax1.bar(o_x, np.zeros(n_obs))
	ax1.axhline(0.0, color='k', ls='dashed')
	ax1.axhline(env.GOAL, color='g', ls='dashed')
	ax1.axhline(-env.GOAL, color='g', ls='dashed')
	ax1.set_title('State')
	ax1.set_ylim((-1, 1))
	# ax2
	a_bars = ax2.bar(a_x, np.zeros(n_act))
	ax2.set_title('Action')
	ax2.set_ylim((-1, 1))
	# ax3
	rew_line, = axr.plot([], [], label='Agent')
	axr.set_xlabel('Steps')
	axr.set_ylabel('Reward')
	# ax4
	opt_o_bars = ax4.bar(o_x, np.zeros(n_obs))
	ax4.axhline(0.0, color='k', ls='dashed')
	ax4.axhline(env.GOAL, color='g', ls='dashed')
	ax4.axhline(-env.GOAL, color='g', ls='dashed')
	ax4.set_title('Opt State')
	ax4.set_ylim((-1, 1))
	# ax5
	opt_a_bars = ax5.bar(a_x, np.zeros(n_act))
	ax5.set_title('Opt Action')
	ax5.set_ylim((-1, 1))
	# ax6
	opt_rew_line, = axr.plot([], [], label='Optimal')
	axr.axhline(env.objective([env.GOAL] * 2), color='g', ls='dashed', label='Reward threshold')
	axr.set_title('Opt Reward')
	axr.legend(loc='lower right')


	def update_bars(o, a, opo, opa):
		nonlocal o_bars, a_bars, opt_o_bars, opt_a_bars, o_x, a_x
		for bar in (o_bars, a_bars, opt_o_bars, opt_a_bars):
			bar.remove()

		o_bars = ax1.bar(o_x, o, color='b')
		a_bars = ax2.bar(a_x, a, color='r')
		opt_o_bars = ax4.bar(o_x, opo, color='b')
		opt_a_bars = ax5.bar(a_x, opa, color='r')


	plt.ion()

	n_episodes = 10
	max_steps = env.EP_LEN_LIM
	for ep in range(n_episodes):
		o = env.reset()
		opt_o = o.copy()
		opt_env.reset(opt_o)

		o_bars.remove()
		a_bars.remove()
		o_bars = ax1.bar(o_x, o, color='b')
		a_bars = ax2.bar(a_x, np.zeros(n_act))

		opt_o_bars.remove()
		opt_a_bars.remove()
		opt_o_bars = ax4.bar(o_x, opt_o, color='b')
		opt_a_bars = ax5.bar(a_x, np.zeros(n_act))

		plt.draw()
		plt.pause(2)

		rewards = []
		opt_rewards = []
		for step in range(max_steps):
			# Put some obs noise to test agent
			# FInd limiting noise
			a = env.action_space.sample()
			o, r, d, _ = env.step(a)
			rewards.append(r)

			opt_a = opt_env.get_optimal_action(opt_o) * 0.1
			# opt_a = np.clip(opt_a, env.action_space.low[0], env.action_space.high[0])
			opt_o, opt_r, opt_d, _ = opt_env.step(opt_a)
			opt_rewards.append(opt_r)

			fig.suptitle(f'Ep #{ep} - Step #{step} - Done {d}')
			fig2.suptitle(f'Ep #{ep} - Step #{step} - Done {d}')
			update_bars(o, a, opt_o, opt_a)

			rew_line.set_data(range(step + 1), rewards)
			opt_rew_line.set_data(range(step + 1), opt_rewards)
			axr.set_ylim((min(np.concatenate([rewards, opt_rewards])), 0))
			axr.set_xlim((0, step + 1))

			if plt.fignum_exists(1) and plt.fignum_exists(2):
				plt.draw()
				plt.pause(ANIM_DELAY)
			else:
				exit()

			if opt_d:
				break

def test_scale_actions():
	np.random.seed(123)
	rm = np.random.uniform(-1, 1, (5, 5))
	env = RandomEnv(5, 5, rm)
	o = env.reset()
	rdm_a = env.action_space.sample()
	opt_a = env.get_optimal_action(o)

	print(rdm_a)
	print(opt_a)

if __name__ == '__main__':
	main()
	# test_scale_actions()