#! /usr/bin/env python
import dmcgym
import gym
import numpy as np
import tqdm
from absl import app, flags
from flax.core import FrozenDict
from ml_collections import config_flags

import wandb
from rlpd.agents import DrQLearner
from rlpd.data import MemoryEfficientReplayBuffer, ReplayBuffer
from rlpd.data.vd4rl_datasets import VD4RLDataset
from rlpd.evaluation import evaluate
from rlpd.wrappers import WANDBVideo, wrap_pixels

FLAGS = flags.FLAGS

flags.DEFINE_string("project_name", "rlpd_pixels", "wandb project name.")
flags.DEFINE_string("env_name", "cheetah-run-v0", "Environment name.")
flags.DEFINE_string(
    "dataset_level", "expert", "Dataset level (e.g., random, expert, etc.)."
)
flags.DEFINE_string("dataset_path", None, "Path to dataset. If None, uses '~/.vd4rl'.")
flags.DEFINE_integer("dataset_size", 500_000, "How many samples to load")
flags.DEFINE_float("offline_ratio", 0.5, "Offline ratio.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 5000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(5e5), "Number of training steps.")
flags.DEFINE_integer(
    "start_training", int(1e3), "Number of training steps to start training."
)
flags.DEFINE_integer("image_size", 64, "Image size.")
flags.DEFINE_integer("num_stack", 3, "Stack frames.")
flags.DEFINE_integer(
    "replay_buffer_size", None, "Number of training steps to start training."
)
flags.DEFINE_integer(
    "action_repeat", None, "Action repeat, if None, uses 2 or PlaNet default values."
)
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean(
    "memory_efficient_replay_buffer", True, "Use a memory efficient replay buffer."
)
flags.DEFINE_boolean("save_video", False, "Save videos during evaluation.")
flags.DEFINE_string("save_dir", None, "Directory to save checkpoints.")
flags.DEFINE_integer("utd_ratio", 1, "Update to data ratio.")
config_flags.DEFINE_config_file(
    "config",
    "configs/drq_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

PLANET_ACTION_REPEAT = {
    "cartpole-swingup-v0": 8,
    "reacher-easy-v0": 4,
    "cheetah-run-v0": 4,
    "finger-spin-v0": 2,
    "ball_in_cup-catch-v0": 4,
    "walker-walk-v0": 2,
}


def combine(one_dict, other_dict):
    combined = {}

    for k, v in one_dict.items():
        if isinstance(v, FrozenDict):
            if len(v) == 0:
                combined[k] = v
            else:
                combined[k] = combine(v, other_dict[k])
        else:
            tmp = np.empty(
                (v.shape[0] + other_dict[k].shape[0], *v.shape[1:]), dtype=v.dtype
            )
            tmp[0::2] = v
            tmp[1::2] = other_dict[k]
            combined[k] = tmp

    return FrozenDict(combined)


def main(_):
    wandb.init(project=FLAGS.project_name)
    wandb.config.update(FLAGS)

    action_repeat = FLAGS.action_repeat or PLANET_ACTION_REPEAT.get(FLAGS.env_name, 2)

    def wrap(env):
        if "quadruped" in FLAGS.env_name:
            camera_id = 2
        else:
            camera_id = 0
        return wrap_pixels(
            env,
            action_repeat=action_repeat,
            image_size=FLAGS.image_size,
            num_stack=FLAGS.num_stack,
            camera_id=camera_id,
        )

    env = gym.make(FLAGS.env_name)
    env, pixel_keys = wrap(env)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    if FLAGS.save_video:
        env = WANDBVideo(env)
    env.seed(FLAGS.seed)

    ds = VD4RLDataset(
        env,
        FLAGS.dataset_level,
        pixel_keys=pixel_keys,
        capacity=FLAGS.dataset_size,
        dataset_path=FLAGS.dataset_path,
    )
    ds_iterator = ds.get_iterator(
        sample_args={
            "batch_size": int(FLAGS.batch_size * FLAGS.utd_ratio * FLAGS.offline_ratio),
            "pack_obs_and_next_obs": True,
        }
    )

    eval_env = gym.make(FLAGS.env_name)
    eval_env, _ = wrap(eval_env)
    eval_env.seed(FLAGS.seed + 42)

    replay_buffer_size = FLAGS.replay_buffer_size or FLAGS.max_steps // action_repeat
    if FLAGS.memory_efficient_replay_buffer:
        replay_buffer = MemoryEfficientReplayBuffer(
            env.observation_space, env.action_space, replay_buffer_size
        )
        replay_buffer_iterator = replay_buffer.get_iterator(
            sample_args={
                "batch_size": int(
                    FLAGS.batch_size * FLAGS.utd_ratio * (1 - FLAGS.offline_ratio)
                ),
                "pack_obs_and_next_obs": True,
            }
        )
    else:
        replay_buffer = ReplayBuffer(
            env.observation_space, env.action_space, replay_buffer_size
        )
        replay_buffer_iterator = replay_buffer.get_iterator(
            sample_args={
                "batch_size": int(
                    FLAGS.batch_size * FLAGS.utd_ratio * (1 - FLAGS.offline_ratio)
                ),
            }
        )

    replay_buffer.seed(FLAGS.seed)

    # Crashes on some setups if agent is created before replay buffer.
    kwargs = dict(FLAGS.config)
    model_cls = kwargs.pop("model_cls")
    agent = globals()[model_cls].create(
        FLAGS.seed,
        env.observation_space,
        env.action_space,
        pixel_keys=pixel_keys,
        **kwargs,
    )

    observation, done = env.reset(), False
    for i in tqdm.tqdm(
        range(1, FLAGS.max_steps // action_repeat + 1),
        smoothing=0.1,
        disable=not FLAGS.tqdm,
    ):
        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action, agent = agent.sample_actions(observation)
        next_observation, reward, done, info = env.step(action)

        if not done or "TimeLimit.truncated" in info:
            mask = 1.0
        else:
            mask = 0.0

        replay_buffer.insert(
            dict(
                observations=observation,
                actions=action,
                rewards=reward,
                masks=mask,
                dones=done,
                next_observations=next_observation,
            )
        )
        observation = next_observation

        if done:
            observation, done = env.reset(), False
            for k, v in info["episode"].items():
                decode = {"r": "return", "l": "length", "t": "time"}
                wandb.log({f"training/{decode[k]}": v}, step=i * action_repeat)

        if i >= FLAGS.start_training:
            online_batch = next(replay_buffer_iterator)
            offline_batch = next(ds_iterator)
            batch = combine(offline_batch, online_batch)
            agent, update_info = agent.update(batch, FLAGS.utd_ratio)

            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    wandb.log({f"training/{k}": v}, step=i * action_repeat)

        if i % FLAGS.eval_interval == 0:
            eval_info = evaluate(
                agent,
                eval_env,
                num_episodes=FLAGS.eval_episodes,
                save_video=FLAGS.save_video,
            )
            for k, v in eval_info.items():
                wandb.log({f"evaluation/{k}": v}, step=i * action_repeat)

            if FLAGS.save_dir is not None:
                from flax.training import checkpoints

                checkpoints.save_checkpoint(
                    FLAGS.save_dir, target=agent, step=i * action_repeat, overwrite=True
                )


if __name__ == "__main__":
    app.run(main)
