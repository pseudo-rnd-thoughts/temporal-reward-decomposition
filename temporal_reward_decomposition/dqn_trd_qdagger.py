"""This code is original based off https://raw.githubusercontent.com/vwxyzjn/cleanrl/master/cleanrl/qdagger_dqn_atari_jax_impalacnn.py
with the following changes made
1. Added temporal reward decomposition neural network
2. Added temporal reward decomposition loss functions
3. Loads a local teacher model rather than huggingface
4. Add periodic agent evaluation rather than just logging the episode rewards
"""

import argparse
import os
import random
import time
from collections import deque
from distutils.util import strtobool
from functools import partial

os.environ[
    "XLA_PYTHON_CLIENT_MEM_FRACTION"
] = "0.7"  # see https://github.com/google/jax/discussions/6332#discussioncomment-1279991

import chex
import flax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from rich.progress import track
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from cleanrl.dqn_jax import QNetwork as TeacherModel
from cleanrl_utils.evals.dqn_jax_eval import evaluate

from temporal_reward_decomposition.utils.n_step_buffer import NStepReplayBuffer


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to save model into the `runs/{run_name}` folder")
    parser.add_argument("--upload-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to upload the saved model to huggingface")
    parser.add_argument("--hf-entity", type=str, default="",
        help="the user or org name of the model repository from the Hugging Face Hub")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="CartPole-v1",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=250_000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--buffer-size", type=int, default=25_000,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=1.,
        help="the target network update rate")
    parser.add_argument("--target-network-frequency", type=int, default=500,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=32,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=1,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.05,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.5,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=10000,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=10,
        help="the frequency of training")

    # QDagger specific arguments
    # parser.add_argument("--teacher-policy-hf-repo", type=str, default=None,
    #     help="the huggingface repo of the teacher policy")
    parser.add_argument("--teacher-eval-episodes", type=int, default=10,
        help="the number of episodes to run the teacher policy evaluate")
    parser.add_argument("--teacher-steps", type=int, default=50_000,
        help="the number of steps to run the teacher policy to generate the replay buffer")
    parser.add_argument("--offline-steps", type=int, default=50_000,
        help="the number of steps to run the student policy with the teacher's replay buffer")
    parser.add_argument("--temperature", type=float, default=1.0,
        help="the temperature parameter for qdagger")
    parser.add_argument("--offline-eval-period", type=int, default=5_000,  # 10x
        help="how often the student will be evaluated within the offline training")
    parser.add_argument("--online-eval-period", type=int, default=25_000,  # 10x
        help="how often the student will be evaluated within the online training")

    # Temporal Reward Decomposition arguments
    parser.add_argument("--num-bins", type=int, required=True,
        help="the number of reward bins")
    parser.add_argument("--n-step", type=int, default=1,
        help="the width of reward bins")

    args = parser.parse_args()
    # fmt: on
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"

    # if args.teacher_policy_hf_repo is None:
    #     args.teacher_policy_hf_repo = f"cleanrl/{args.env_id}-dqn_atari_jax-seed1"

    assert args.num_bins > 1 and args.n_step >= 1

    return args


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    action_dim: int
    num_bins: int

    def __call__(self, x: jnp.ndarray):
        return jnp.sum(self.decomposed_q_value(x), axis=-1)

    @nn.compact
    def decomposed_q_value(self, x: jnp.ndarray):
        x = nn.Dense(120)(x)
        x = nn.relu(x)
        x = nn.Dense(84)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim * self.num_bins)(x)
        return jnp.reshape(x, (-1, self.action_dim, self.num_bins))


class TrainState(TrainState):
    target_params: flax.core.FrozenDict


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, q_key = jax.random.split(key, 2)

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(action_dim=envs.single_action_space.n, num_bins=args.num_bins)
    q_network.apply = jax.jit(q_network.apply, static_argnames=("method",))

    q_state = TrainState.create(
        apply_fn=q_network.apply,
        params=q_network.init(q_key, envs.observation_space.sample()),
        target_params=q_network.init(q_key, envs.observation_space.sample()),
        tx=optax.adam(learning_rate=args.learning_rate),
    )

    # TRD logic
    discount_factor = jnp.power(args.gamma, args.n_step)
    # Helper variables
    batch_size = args.batch_size
    num_actions = envs.single_action_space.n
    num_bins = args.num_bins

    # QDAGGER LOGIC:
    # teacher_model_path = hf_hub_download(repo_id=args.teacher_policy_hf_repo, filename="dqn_atari_jax.cleanrl_model")
    teacher_model_path = f"dqn-models/{args.env_id}-dqn_jax-seed-1/dqn_jax.cleanrl_model"
    teacher_model = TeacherModel(action_dim=envs.single_action_space.n)
    teacher_model_key = jax.random.PRNGKey(args.seed)
    teacher_params = teacher_model.init(teacher_model_key, envs.observation_space.sample())
    with open(teacher_model_path, "rb") as f:
        teacher_params = flax.serialization.from_bytes(teacher_params, f.read())
    teacher_model.apply = jax.jit(teacher_model.apply)

    # evaluate the teacher model
    teacher_episodic_returns = evaluate(
        teacher_model_path,
        make_env,
        args.env_id,
        eval_episodes=args.teacher_eval_episodes,
        run_name=f"{run_name}-teacher-eval",
        Model=TeacherModel,
        epsilon=0.05,
        capture_video=False,
    )
    for idx, episode_return in enumerate(teacher_episodic_returns):
        writer.add_scalar(f"charts/teacher/episodic_return", episode_return, idx)

    # collect teacher data for args.teacher_steps
    # we assume we don't have access to the teacher's replay buffer
    # see Fig. A.19 in Agarwal et al. 2022 for more detail
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        "cpu",
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )
    rb = NStepReplayBuffer(
        rb,
        args.n_step,
        args.gamma
    )

    obs, _ = envs.reset(seed=args.seed)
    start_time = time.time()
    print(f'Started filling: {start_time}')
    for global_step in track(range(args.teacher_steps), description="filling teacher's replay buffer"):
        epsilon = linear_schedule(args.start_e, args.end_e, args.teacher_steps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = teacher_model.apply(teacher_params, obs)
            actions = q_values.argmax(axis=-1)
            actions = jax.device_get(actions)
        next_obs, rewards, terminated, truncated, infos = envs.step(actions)
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(truncated):
            if d:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminated, infos)
        obs = next_obs
    end_time = time.time()
    print(f'Stopped filling : {end_time}, diff: {end_time - start_time:.2f} seconds')

    @jax.vmap
    def kl_divergence_with_logits(target_logits, prediction_logits):
        """Implementation of on-policy distillation loss."""
        out = -nn.softmax(target_logits) * (nn.log_softmax(prediction_logits) - nn.log_softmax(target_logits))
        return jnp.sum(out)

    @jax.jit
    def update(q_state, observations, actions, next_observations, rewards, terminated, distill_coeff):
        # Temporal reward decomposition loss function
        q_next_target = q_network.apply(q_state.target_params, next_observations, method=QNetwork.decomposed_q_value)
        chex.assert_shape(q_next_target, (batch_size, num_actions, num_bins))
        q_next_target_value = q_next_target[jnp.arange(args.batch_size), jnp.argmax(jnp.sum(q_next_target, axis=-1), axis=-1)]
        chex.assert_shape(q_next_target_value, (batch_size, num_bins))

        discounted_q_next_target = jnp.expand_dims(1 - terminated, axis=1) * discount_factor * q_next_target_value
        chex.assert_shape(discounted_q_next_target, (batch_size, num_bins))
        rolled_q_next_target = jnp.roll(discounted_q_next_target, shift=1, axis=1)
        chex.assert_shape(rolled_q_next_target, (batch_size, num_bins))
        next_q_value = rolled_q_next_target.at[:, -1].add(rolled_q_next_target[:, 0]).at[:, 0].set(rewards)
        chex.assert_shape(next_q_value, (batch_size, num_bins))

        teacher_q_values = teacher_model.apply(teacher_params, observations)
        chex.assert_shape(teacher_q_values, (batch_size, num_actions))

        def qdagger_trd_loss(params, td_target, teacher_q_values):
            student_q_values = q_network.apply(params, observations, method=QNetwork.decomposed_q_value)
            chex.assert_shape(student_q_values, (batch_size, num_actions, num_bins))

            # td loss
            q_pred = student_q_values[jnp.arange(batch_size), actions.squeeze()]
            chex.assert_shape(q_pred, (batch_size, num_bins))
            q_loss = jnp.mean(jnp.square(q_pred - td_target))
            chex.assert_shape(q_loss, ())

            # distil loss
            teacher_q_values = teacher_q_values / args.temperature
            student_q_values = jnp.sum(student_q_values, axis=-1) / args.temperature
            chex.assert_shape(teacher_q_values, (batch_size, num_actions))
            chex.assert_shape(student_q_values, (batch_size, num_actions))
            policy_divergence = kl_divergence_with_logits(teacher_q_values, student_q_values)
            chex.assert_shape(policy_divergence, (batch_size,))
            distill_loss = distill_coeff * jnp.mean(policy_divergence)
            chex.assert_shape(distill_loss, ())

            overall_loss = q_loss + distill_loss
            chex.assert_shape(overall_loss, ())
            return overall_loss, (q_loss, q_pred, distill_loss)

        (loss_value, (q_loss, q_pred, distill_loss)), grads = jax.value_and_grad(qdagger_trd_loss, has_aux=True)(
            q_state.params, next_q_value, teacher_q_values
        )
        q_state = q_state.apply_gradients(grads=grads)
        return loss_value, q_loss, q_pred, distill_loss, q_state

    # offline training phase: train the student model using the qdagger loss
    for global_step in track(range(args.offline_steps), description="offline student training"):
        data = rb.sample(args.batch_size)
        # perform a gradient-descent step
        loss, q_loss, old_val, distill_loss, q_state = update(
            q_state,
            data.observations.numpy(),
            data.actions.numpy(),
            data.next_observations.numpy(),
            data.rewards.flatten().numpy(),
            data.dones.flatten().numpy(),
            1.0,
        )

        # update the target network
        if global_step % args.target_network_frequency == 0:
            q_state = q_state.replace(target_params=optax.incremental_update(q_state.params, q_state.target_params, args.tau))

        if global_step % 100 == 0:
            writer.add_scalar("charts/offline/loss", jax.device_get(loss), global_step)
            writer.add_scalar("charts/offline/q_loss", jax.device_get(q_loss), global_step)
            writer.add_scalar("charts/offline/distill_loss", jax.device_get(distill_loss), global_step)

        if global_step % args.offline_eval_period == 0:
            # evaluate the student model
            model_path = f"runs/{run_name}/{args.exp_name}-offline-{global_step}.cleanrl_model"
            with open(model_path, "wb") as f:
                f.write(flax.serialization.to_bytes(q_state.params))
            print(f"model saved to {model_path}")

            episodic_returns = evaluate(
                model_path,
                make_env,
                args.env_id,
                eval_episodes=10,
                run_name=f"{run_name}-eval",
                Model=partial(QNetwork, num_bins=args.num_bins),
                epsilon=0.05,
                capture_video=False
            )
            for idx, returns in enumerate(episodic_returns):
                writer.add_scalar(f"charts/offline/episodic_return_{idx}", returns, global_step)

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        "cpu",
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )
    rb = NStepReplayBuffer(rb, args.n_step, args.gamma)
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    obs, _ = envs.reset(seed=args.seed)
    episodic_returns = deque(maxlen=10)
    # online training phase
    for global_step in track(range(args.total_timesteps), description="online student training"):
        global_step += args.offline_steps
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network.apply(q_state.params, obs)
            actions = q_values.argmax(axis=-1)
            actions = jax.device_get(actions)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminated, truncated, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                # Skip the envs that are not done
                if "episode" not in info:
                    continue
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                writer.add_scalar("charts/epsilon", epsilon, global_step)
                episodic_returns.append(info["episode"]["r"])
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(truncated):
            if d:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminated, truncated)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.offline_steps + args.batch_size:  # args.learning_starts
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                # perform a gradient-descent step
                if len(episodic_returns) < 10:
                    distill_coeff = 1.0
                else:
                    distill_coeff = max(1 - np.mean(episodic_returns) / np.mean(teacher_episodic_returns), 0)
                loss, q_loss, old_val, distill_loss, q_state = update(
                    q_state,
                    data.observations.numpy(),
                    data.actions.numpy(),
                    data.next_observations.numpy(),
                    data.rewards.flatten().numpy(),
                    data.dones.flatten().numpy(),
                    distill_coeff,
                )

                if global_step % 100 == 0:
                    writer.add_scalar("losses/loss", jax.device_get(loss), global_step)
                    writer.add_scalar("losses/td_loss", jax.device_get(q_loss), global_step)
                    writer.add_scalar("losses/distill_loss", jax.device_get(distill_loss), global_step)
                    writer.add_scalar("losses/q_values", jax.device_get(old_val).mean(), global_step)
                    writer.add_scalar("charts/distill_coeff", distill_coeff, global_step)
                    # print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            # update the target network
            if global_step % args.target_network_frequency == 0:
                q_state = q_state.replace(
                    target_params=optax.incremental_update(q_state.params, q_state.target_params, args.tau)
                )

            if global_step % args.online_eval_period == 0:
                # evaluate the student model
                model_path = f"runs/{run_name}/{args.exp_name}-online-{global_step}.cleanrl_model"
                with open(model_path, "wb") as f:
                    f.write(flax.serialization.to_bytes(q_state.params))
                print(f"model saved to {model_path}")

                episodic_returns = evaluate(
                    model_path,
                    make_env,
                    args.env_id,
                    eval_episodes=10,
                    run_name=f"{run_name}-eval",
                    Model=partial(QNetwork, num_bins=args.num_bins),
                    epsilon=0.05,
                    capture_video=False
                )
                for idx, returns in enumerate(episodic_returns):
                    writer.add_scalar(f"charts/online/episodic_return_{idx}", returns, global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        with open(model_path, "wb") as f:
            f.write(flax.serialization.to_bytes(q_state.params))
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.dqn_jax_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=partial(QNetwork, num_bins=args.num_bins),
            epsilon=args.end_e,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        # if args.upload_model:
        #     from cleanrl_utils.huggingface import push_to_hub
        #
        #     repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
        #     repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
        #     push_to_hub(args, episodic_returns, repo_id, "trd-Qdagger", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
