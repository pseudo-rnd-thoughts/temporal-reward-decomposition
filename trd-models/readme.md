This folder contains the trained dataset of dqn-atari-trd-qdagger for a range of hyperparameter sweeps.

The filename mean as followed: {env_id}-seed-{seed value}-n-{reward vector size}-w-{reward grouping size}.cleanrl_model

Models can be loaded through 

```python
from temporal_reward_decomposition.dqn_atari_trd_qdagger import QNetwork, make_env
import gymnasium as gym 
import jax
import flax 

envs = gym.vector.SyncVectorEnv(
    [make_env(f"{env_id}NoFrameskip-v4", 0, 0, True, "video-folder", disable_noop=True)]
)

rng = jax.random.PRNG(1)
network = QNetwork(action_dim=envs.single_action_space.n, num_bins=reward_vector_size)
with open(f"trd-models/{env_id}-seed-{seed}-n-{reward_vector_size}-w-{reward_grouping}.cleanrl_model", "rb") as file:
    params = flax.serialization.from_bytes(network.init(rng, envs.observation_space.sample()), file.read())
```