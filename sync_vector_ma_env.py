"""A synchronous vector environment."""
from copy import deepcopy
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union, Dict

import numpy as np
from numpy.typing import NDArray

from gymnasium.spaces import Space
from gymnasium.vector.utils import concatenate, create_empty_array, iterate
from pettingzoo import ParallelEnv

from vector_ma_env import VectorMAEnv

__all__ = ["SyncVectorMAEnv"]


class SyncVectorMAEnv(VectorMAEnv):

    def __init__(
        self,
        env_fns: Iterable[Callable[[], ParallelEnv]],
        joint_observation_space: Optional[Dict[str, Space]] = None,
        joint_action_spaces: Optional[Dict[str, Space]] = None,
        copy: bool = True,
    ):
        self.env_fns = env_fns
        self.ma_envs = [env_fn() for env_fn in env_fns]
        self.copy = copy
        self.metadata = self.ma_envs[0].metadata
        self.possible_agents = self.ma_envs[0].possible_agents

        if (joint_observation_space is None) or (joint_action_spaces is None):
            joint_observation_space = joint_observation_space or self.ma_envs[0].observation_spaces
            joint_action_spaces = joint_action_spaces or {agent: self.ma_envs[0].action_space(agent) for agent in self.ma_envs[0].possible_agents}

        super().__init__(
            agent_ids=self.ma_envs[0].possible_agents,
            num_envs=len(self.ma_envs),
            joint_observation_spaces=joint_observation_space,
            joint_action_spaces=joint_action_spaces,
        )

        self.observations = {
            agent: create_empty_array(self.single_joint_observation_space[agent], n=self.num_envs, fn=np.zeros)
            for agent in self.ma_envs[0].possible_agents
        }
        self._rewards = {
            agent: np.zeros((self.num_envs,), dtype=np.float64)
            for agent in self.ma_envs[0].possible_agents
        }
        self._terminateds = {
            agent: np.zeros((self.num_envs,), dtype=np.bool_)
            for agent in self.ma_envs[0].possible_agents
        }
        self._truncateds = {
            agent: np.zeros((self.num_envs,), dtype=np.bool_)
            for agent in self.ma_envs[0].possible_agents
        }
        self._actions = None

    def seed(self, seed: Optional[Union[int, Sequence[int]]] = None):
        super().seed(seed=seed)
        if seed is None:
            seed = [None for _ in range(self.num_envs)]
        if isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]
        assert len(seed) == self.num_envs

        for env, single_seed in zip(self.ma_envs, seed):
            env.seed(single_seed)

    def reset_wait(
        self,
        seed: Optional[Union[int, List[int]]] = None,
        options: Optional[dict] = None,
    ):
        if seed is None:
            seed = [None for _ in range(self.num_envs)]
        if isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]
        assert len(seed) == self.num_envs

        for agent in self.ma_envs[0].possible_agents:
            self._terminateds[agent][:] = False
            self._truncateds[agent][:] = False

        joint_observations = []
        infos = {}
        for i, (env, single_seed) in enumerate(zip(self.ma_envs, seed)):
            kwargs = {}
            if single_seed is not None:
                kwargs["seed"] = single_seed
            if options is not None:
                kwargs["options"] = options

            joint_observation, info = env.reset(**kwargs)
            joint_observations.append(joint_observation)
            infos = self._add_info(infos, info, i)

        self.observations = {
            agent: concatenate(
                self.single_joint_observation_space[agent],
                [joint_observations[i][agent] for i in range(self.num_envs)],
                self.observations[agent]
            )
            for agent in self.ma_envs[0].possible_agents
        }
        return (deepcopy(self.observations) if self.copy else self.observations), infos

    def step_async(self, actions):
        """Sets :attr:`_actions` for use by the :meth:`step_wait` by converting the ``actions`` to an iterable version."""
        self._actions = []
        for i in range(len(self.ma_envs)):
            self._actions.append({agent_id: actions[agent_id][i] for agent_id in self.ma_envs[i].possible_agents})

    def step_wait(self) -> Tuple[Dict, Dict, Dict, Dict, dict]:
        """Steps through each of the environments returning the batched results.

        Returns:
            The batched environment step results
        """
        observations, infos = [], {}

        for i, (ma_env, joint_action) in enumerate(zip(self.ma_envs, self._actions)):
            (
                observation,
                rewards,
                terminates,
                truncateds,
                info,
            ) = ma_env.step(joint_action)
            for agent_id in self.possible_agents:
                self._rewards[agent_id][i] = rewards[agent_id]
                self._terminateds[agent_id][i] = terminates[agent_id]
                self._truncateds[agent_id][i] = truncateds[agent_id]

            # reset all agents simultaneously
            if any(terminates.values()) or any(truncateds.values()):
                old_observation, old_info = observation, info
                observation, info = ma_env.reset()
                for agent_id in self.possible_agents:
                    info[agent_id].update({"final_observation": old_observation[agent_id],})
                    info[agent_id].update({"final_info": old_info[agent_id],})

            observations.append(observation)
            infos = self._add_info(infos, info, i)

        self.observations = {
            agent: concatenate(
                self.single_joint_observation_space[agent],
                [observations[j][agent] for j in range(self.num_envs)],
                self.observations[agent]
            )
            for agent in self.ma_envs[0].possible_agents
        }

        return (
            deepcopy(self.observations) if self.copy else self.observations,
            deepcopy(self._rewards),
            deepcopy(self._terminateds),
            deepcopy(self._truncateds),
            infos,
        )

    def call(self, name, *args, **kwargs) -> tuple:
        results = []
        for env in self.ma_envs:
            function = getattr(env, name)
            if callable(function):
                results.append(function(*args, **kwargs))
            else:
                results.append(function)

        return tuple(results)

    def set_attr(self, name: str, values: Union[list, tuple, Any]):
        if not isinstance(values, (list, tuple)):
            values = [values for _ in range(self.num_envs)]
        if len(values) != self.num_envs:
            raise ValueError(
                "Values must be a list or tuple with length equal to the "
                f"number of environments. Got `{len(values)}` values for "
                f"{self.num_envs} environments."
            )

        for env, value in zip(self.ma_envs, values):
            setattr(env, name, value)

    def close_extras(self, **kwargs):
        [env.close() for env in self.ma_envs]
