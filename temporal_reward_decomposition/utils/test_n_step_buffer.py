import numpy as np
from gymnasium.spaces import Discrete
from temporal_reward_decomposition.utils.n_step_buffer import NStepReplayBuffer
from stable_baselines3.common.buffers import ReplayBuffer
import pytest


@pytest.mark.parametrize("n_step", [1, 3, 5])
def test_non_terminating(n_step: int, timesteps: int = 10):
    buffer = ReplayBuffer(timesteps, Discrete(timesteps), Discrete(timesteps))
    n_step_buffer = NStepReplayBuffer(buffer, n_step, 0.9)

    for i in range(timesteps+n_step-1):
        n_step_buffer.add(
            observation=np.array([i]),
            action=np.array([i]),
            next_observation=np.array([i+1]),
            reward=i+1,
            terminated=False,
            truncated=False
        )

    print()
    for i in range(timesteps):
        print(f'o={buffer.observations[i]}, '
              f'a={buffer.actions[i]}, '
              f"o'={buffer.next_observations[i]}, "
              f'r={buffer.rewards[i]}, '
              f't={buffer.dones[i]}, ')

    # samples = buffer._get_samples(np.arange(timesteps))
    # assert np.all(samples.observations.numpy() == np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]))
    # assert np.all(samples.actions.numpy() == np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]))
    # assert samples.next_observations == np.arange(timesteps)
    # assert samples.rewards == np.arange(timesteps)
    # assert samples.dones == np.zeros(timesteps, dtype=np.bool_)


@pytest.mark.parametrize("n_step", [1, 3, 5])
def test_terminating(n_step: int, timesteps: int = 10):
    """MISSING OBS=9"""
    buffer = ReplayBuffer(timesteps, Discrete(timesteps), Discrete(timesteps))
    n_step_buffer = NStepReplayBuffer(buffer, n_step, 0.9)

    for i in range(timesteps):
        n_step_buffer.add(
            observation=np.array([i]),
            action=np.array([i]),
            next_observation=np.array([i+1]),
            reward=i+1,
            terminated=i == 6 or i == 9,
            truncated=False
        )

    print()
    for i in range(timesteps):
        print(f'o={buffer.observations[i]}, '
              f'a={buffer.actions[i]}, '
              f"o'={buffer.next_observations[i]}, "
              f'r={buffer.rewards[i]}, '
              f't={buffer.dones[i]}, ')

    # samples = buffer._get_samples(np.arange(timesteps))
    # assert samples.observations == np.arange(timesteps)
    # assert samples.actions == np.arange(timesteps)
    # assert samples.next_observations == np.arange(timesteps)
    # assert samples.rewards == np.arange(timesteps)
    # assert samples.dones == np.array([False, False, False, False, False, True, False, False, False])


@pytest.mark.parametrize("n_step", [1, 2, 3])
def test_truncating(n_step: int, timesteps: int = 10):
    buffer = ReplayBuffer(timesteps, Discrete(timesteps), Discrete(timesteps))
    n_step_buffer = NStepReplayBuffer(buffer, n_step, 0.9)

    for i in range(10):
        n_step_buffer.add(
            observation=np.array([i]),
            action=np.array([i]),
            next_observation=np.array([i+1]),
            reward=i+1,
            terminated=False,
            truncated=i == 6 or i == 9,
        )

    print()
    for i in range(timesteps):
        print(f'o={buffer.observations[i]}, '
              f'a={buffer.actions[i]}, '
              f"o'={buffer.next_observations[i]}, "
              f'r={buffer.rewards[i]}, '
              f't={buffer.dones[i]}, ')

    # samples = buffer._get_samples(np.arange(timesteps))
    # assert samples.observations == np.arange(timesteps)
    # assert samples.actions == np.arange(timesteps)
    # assert samples.next_observations == np.arange(timesteps)
    # assert samples.rewards == np.arange(timesteps)
    # assert samples.dones == np.zeros(timesteps, dtype=np.bool_)
