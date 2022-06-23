import copy

import gym
import numpy as np
from gym.spaces import Box, Dict


def _convert_space(obs_space):
    if isinstance(obs_space, Box):
        obs_space = Box(obs_space.low, obs_space.high, obs_space.shape)
    elif isinstance(obs_space, Dict):
        for k, v in obs_space.spaces.items():
            obs_space.spaces[k] = _convert_space(v)
        obs_space = Dict(obs_space.spaces)
    else:
        raise NotImplementedError
    return obs_space


def _convert_obs(obs):
    if isinstance(obs, np.ndarray):
        if obs.dtype == np.float64:
            return obs.astype(np.float32)
        else:
            return obs
    elif isinstance(obs, dict):
        obs = copy.copy(obs)
        for k, v in obs.items():
            obs[k] = _convert_obs(v)
        return obs


class SinglePrecision(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        obs_space = copy.deepcopy(self.env.observation_space)
        self.observation_space = _convert_space(obs_space)

    def observation(self, observation):
        return _convert_obs(observation)
