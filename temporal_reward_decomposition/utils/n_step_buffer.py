from collections import deque

import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer


class NStepReplayBuffer:

    def __init__(self, buffer: ReplayBuffer, n_step: int, gamma: float):
        self.n_step = n_step
        self.gamma = gamma
        self.buffer = buffer

        self.n_step_gamma = np.power(gamma, np.arange(n_step))  # [1, gamma, gamma^2, ..., gamma^n_step
        # print(f'{self.gamma=}, {self.n_step=}, {self.n_step_gamma=}')

        self.pre_buffer_observation_buffer = deque(maxlen=n_step)
        self.pre_buffer_action_buffer = deque(maxlen=n_step)
        self.pre_buffer_reward_buffer = np.zeros(self.n_step)
        self.pre_buffer_next_observation_buffer = deque(maxlen=n_step)

    def add(self, observation, next_observation, action, reward, terminated, truncated):
        """Adds (observation, action, reward, terminated and next_observation) to the replay buffer

        :param observation: The observation that the agent acted on
        :param next_observation: The next observation resulting in the action being taken
        :param action: The action that the agent took
        :param reward: The reward for the action given the observation
        :param truncated: If the environment truncated after action
        :param terminated: If the action resulted in the environment terminating
        """
        assert isinstance(observation, np.ndarray)
        assert isinstance(next_observation, np.ndarray)
        assert isinstance(action, np.ndarray)
        assert isinstance(reward, np.ndarray) and reward.shape == (1,)
        assert isinstance(terminated, np.ndarray) and terminated.shape == (1,)
        assert isinstance(truncated, np.ndarray) and truncated.shape == (1,)

        index = len(self.pre_buffer_observation_buffer)
        # print(f'{observation=}, {index=}, {terminated=}, {truncated=}')
        if index == 0:
            self.pre_buffer_reward_buffer[0] = reward
        elif index == self.n_step:
            self.pre_buffer_reward_buffer = np.roll(self.pre_buffer_reward_buffer, shift=-1)
            self.pre_buffer_reward_buffer[-1] = reward
        else:
            self.pre_buffer_reward_buffer[index] = reward

        # print(f'{self.pre_buffer_reward_buffer=}, {index=}')

        self.pre_buffer_observation_buffer.append(observation)
        self.pre_buffer_action_buffer.append(action)
        self.pre_buffer_next_observation_buffer.append(next_observation)

        if terminated:
            index = len(self.pre_buffer_observation_buffer)
            # print(f'reward buffer: {self.pre_buffer_reward_buffer}')
            # print(f'obs buffer: {self.pre_buffer_observation_buffer}, '
            #       f'next obs: {self.pre_buffer_next_observation_buffer}')
            for i in range(index):
                n_step_reward = np.sum(self.pre_buffer_reward_buffer * self.n_step_gamma)
                # print(f'reward buffer: {self.pre_buffer_reward_buffer}, n-step: {n_step_reward}')
                self.pre_buffer_reward_buffer[0] = 0
                self.pre_buffer_reward_buffer = np.roll(self.pre_buffer_reward_buffer, shift=-1)
                # print(f'updated reward buffer: {self.pre_buffer_reward_buffer}')
                self.buffer.add(
                    self.pre_buffer_observation_buffer[i],
                    next_observation,
                    self.pre_buffer_action_buffer[i],
                    n_step_reward,
                    True,
                    [{}]
                )

            self.pre_buffer_observation_buffer.clear()
            self.pre_buffer_action_buffer.clear()
            self.pre_buffer_reward_buffer = np.zeros(self.n_step)
            self.pre_buffer_next_observation_buffer.clear()
        elif len(self.pre_buffer_observation_buffer) == self.n_step:
            n_step_reward = np.sum(self.pre_buffer_reward_buffer * self.n_step_gamma)
            # print(f'rewards: {self.pre_buffer_reward_buffer}, n-step rewards: {self.pre_buffer_reward_buffer * self.n_step_gamma}, sum: {n_step_reward}')
            self.buffer.add(
                self.pre_buffer_observation_buffer[0],
                next_observation,
                self.pre_buffer_action_buffer[0],
                n_step_reward,
                False,
                [{}]
            )

            if truncated:
                self.pre_buffer_observation_buffer.clear()
                self.pre_buffer_action_buffer.clear()
                self.pre_buffer_reward_buffer = np.zeros(self.n_step)
                self.pre_buffer_next_observation_buffer.clear()

    def sample(self, batch_size):
        return self.buffer.sample(batch_size)
