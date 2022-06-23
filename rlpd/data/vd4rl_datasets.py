import os
import pathlib
import random
from collections import deque
from typing import Optional

import dmcgym
import gym
import numpy as np

from rlpd.data import MemoryEfficientReplayBuffer

VD4RL_DIR = "~/.vd4rl"


def get_dataset_dir(env, dataset_level, dataset_path):
    env_name = env.unwrapped.spec.id
    env_name = "_".join(env_name.split("-")[:-1])
    dataset_path = dataset_path if dataset_path else VD4RL_DIR
    dataset_dir = os.path.join(dataset_path, "main", f"{env_name}/{dataset_level}/64px")
    return pathlib.Path(os.path.expanduser(dataset_dir))


def load_episodes(directory, capacity=None, keep_temporal_order=False):
    # The returned directory from filenames to episodes is guaranteed to be in
    # temporally sorted order.
    filenames = sorted(directory.glob("*.npz"))
    if not keep_temporal_order:
        print("Shuffling order of offline trajectories!")
        random.Random(0).shuffle(filenames)
    if capacity:
        num_steps = 0
        num_episodes = 0
        for filename in reversed(filenames):
            length = int(str(filename).split("-")[-1][:-4])
            num_steps += length
            num_episodes += 1
            if num_steps >= capacity:
                break
        filenames = filenames[-num_episodes:]
    episodes = {}
    for filename in filenames:
        try:
            with filename.open("rb") as f:
                episode = np.load(f)
                episode = {k: episode[k] for k in episode.keys()}
                # Conversion for older versions of npz files.
                if "is_terminal" not in episode:
                    episode["is_terminal"] = episode["discount"] == 0.0
        except Exception as e:
            print(f"Could not load episode {str(filename)}: {e}")
            continue
        episodes[str(filename)] = episode
    return episodes


def convert(value):
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
        return value.astype(np.float32)
    elif np.issubdtype(value.dtype, np.signedinteger):
        return value.astype(np.int32)
    elif np.issubdtype(value.dtype, np.uint8):
        return value.astype(np.uint8)
    return value


def wrap(env):
    from rlpd.wrappers import wrap_pixels

    return wrap_pixels(
        env,
        action_repeat=2,
        image_size=64,
        num_stack=3,
        camera_id=0,
    )


class VD4RLDataset(MemoryEfficientReplayBuffer):
    def __init__(
        self,
        env: gym.Env,
        dataset_level: str,
        pixel_keys: tuple = ("pixels",),
        capacity: int = 500_000,
        dataset_path: Optional[str] = None,
    ):

        super().__init__(
            env.observation_space,
            env.action_space,
            capacity=capacity,
            pixel_keys=pixel_keys,
        )
        dataset_dir = get_dataset_dir(env, dataset_level, dataset_path)
        dataset_dict = load_episodes(dataset_dir, keep_temporal_order=True)
        framestack = env.observation_space[pixel_keys[0]].shape[-1]

        for episode in dataset_dict.values():
            for i in range(episode["image"].shape[0]):
                if not episode["is_first"][i]:
                    next_stacked_frames.append(episode["image"][i])
                    data_dict = dict(
                        observations={"pixels": np.stack(stacked_frames, axis=-1)},
                        actions=episode["action"][i],
                        rewards=episode["reward"][i],
                        masks=1 - np.float32(episode["is_terminal"][i]),
                        dones=np.float32(episode["is_terminal"][i]),
                        next_observations={
                            "pixels": np.stack(next_stacked_frames, axis=-1)
                        },
                    )
                    self.insert(data_dict)
                    stacked_frames.append(episode["image"][i])
                else:
                    stacked_frames = deque(maxlen=framestack)
                    next_stacked_frames = deque(maxlen=framestack)
                    while len(stacked_frames) < framestack:
                        stacked_frames.append(episode["image"][i])
                        next_stacked_frames.append(episode["image"][i])


if __name__ == "__main__":
    env = gym.make("cheetah-run-v0")
    env, pixel_keys = wrap(env)
    dataset = VD4RLDataset(env, "expert", pixel_keys=pixel_keys)
