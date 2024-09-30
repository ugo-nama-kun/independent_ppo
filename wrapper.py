"""Wrapper that tracks the cumulative rewards and episode lengths."""
import time
from collections import deque
from typing import Optional

import numpy as np

import gymnasium as gym
from pandas.core.missing import clean_interp_method
from pettingzoo.utils.env import AgentID, ObsType, ActionType, ParallelEnv

from pettingzoo.utils.wrappers import BaseParallelWrapper


class RecordParallelEpisodeStatistics(BaseParallelWrapper[AgentID, ObsType, ActionType], gym.utils.RecordConstructorArgs):
    """This wrapper will keep track of cumulative rewards and episode lengths.

    At the end of an episode, the statistics of the episode will be added to ``info``
    using the key ``episode``. If using a vectorized environment also the key
    ``_episode`` is used which indicates whether the env at the respective index has
    the episode statistics.

    After the completion of an episode, ``info`` will look like this::

        >>> info = {
        ...     "episode": {
        ...         "r": "<cumulative reward>",
        ...         "l": "<episode length>",
        ...         "t": "<elapsed time since beginning of episode>"
        ...     },
        ... }

    For a vectorized environments the output will be in the form of::

        >>> infos = {
        ...     "final_observation": "<array of length num-envs>",
        ...     "_final_observation": "<boolean array of length num-envs>",
        ...     "final_info": "<array of length num-envs>",
        ...     "_final_info": "<boolean array of length num-envs>",
        ...     "episode": {
        ...         "r": "<array of cumulative reward>",
        ...         "l": "<array of episode length>",
        ...         "t": "<array of elapsed time since beginning of episode>"
        ...     },
        ...     "_episode": "<boolean array of length num-envs>"
        ... }

    Moreover, the most recent rewards and episode lengths are stored in buffers that can be accessed via
    :attr:`wrapped_env.return_queue` and :attr:`wrapped_env.length_queue` respectively.

    Attributes:
        return_queue: The cumulative rewards of the last ``deque_size``-many episodes
        length_queue: The lengths of the last ``deque_size``-many episodes
    """

    def __init__(self, env: ParallelEnv[AgentID, ObsType, ActionType], deque_size: int = 100):
        """This wrapper will keep track of cumulative rewards and episode lengths.

        Args:
            env (Env): The environment to apply the wrapper
            deque_size: The size of the buffers :attr:`return_queue` and :attr:`length_queue`
        """
        gym.utils.RecordConstructorArgs.__init__(self, deque_size=deque_size)
        BaseParallelWrapper.__init__(self, env)

        self.possible_agents = getattr(env, "possible_agents", 1)
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_count = 0
        self.episode_start_times: np.ndarray = None
        self.episode_returns: Optional[dict] = None
        self.episode_lengths: Optional[dict] = None
        self.return_queue = {agent_id: deque(maxlen=deque_size) for agent_id in self.possible_agents}
        self.length_queue = {agent_id: deque(maxlen=deque_size) for agent_id in self.possible_agents}
        self.is_vector_env = getattr(env, "is_vector_env", False)

    def reset(self, **kwargs):
        """Resets the environment using kwargs and resets the episode returns and lengths."""
        obss, infos = super().reset(**kwargs)
        self.episode_start_times = {agent_id: np.full(self.num_envs, time.perf_counter(), dtype=np.float32) for agent_id in self.possible_agents}
        self.episode_returns = {agent_id : np.zeros(self.num_envs, dtype=np.float32) for agent_id in self.possible_agents}
        self.episode_lengths = {agent_id: np.zeros(self.num_envs, dtype=np.int32) for agent_id in self.possible_agents}
        return obss, infos

    def step(self, actions):
        """Steps through the environment, recording the episode statistics."""
        (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        ) = self.env.step(actions)
        assert isinstance(
            infos, dict
        ), f"`info` dtype is {type(infos)} while supported dtype is `dict`. This may be due to usage of other wrappers in the wrong order."

        # multi-agent data structure: [agent_id, num_envs]
        for i, agent_id in enumerate(self.possible_agents):
            self.episode_returns[agent_id] += rewards[agent_id]
            dones = np.logical_or(terminations[agent_id], truncations[agent_id])

            self.episode_lengths[agent_id] += 1

            # calculate done for every agents
            num_dones = np.sum((dones, ))
            if num_dones:
                if "episode" in infos[agent_id] or "_episode" in infos[agent_id]:
                    raise ValueError(
                        "Attempted to add episode stats when they already exist"
                    )
                else:
                    infos[agent_id]["episode"] = {}
                    infos[agent_id]["episode"] = {
                        "r": np.where(dones, self.episode_returns[agent_id], 0.0),
                        "l": np.where(dones, self.episode_lengths[agent_id], 0),
                        "t": np.where(
                            dones,
                            np.round(time.perf_counter() - self.episode_start_times[agent_id], 6),
                            0.0,
                        ),
                    }
                    if self.is_vector_env:
                        infos[agent_id]["_episode"] = {}
                        infos[agent_id]["_episode"] = np.where(dones, True, False)

                self.return_queue[agent_id].extend(self.episode_returns[agent_id][dones])
                self.length_queue[agent_id].extend(self.episode_lengths[agent_id][dones])
                self.episode_count += num_dones
                self.episode_lengths[agent_id][dones] = 0
                self.episode_returns[agent_id][dones] = 0
                self.episode_start_times[agent_id][dones] = time.perf_counter()
        return (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        )
