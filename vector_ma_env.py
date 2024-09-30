"""Base class for vectorized environments."""
from typing import Any, List, Optional, Tuple, Union, Dict

import numpy as np
from numpy.typing import NDArray

import gymnasium as gym
from gymnasium.vector.utils.spaces import batch_space


__all__ = ["VectorMAEnv"]

from pettingzoo import ParallelEnv


class VectorMAEnv(ParallelEnv):

    def __init__(
        self,
        agent_ids: List[str],
        num_envs: int,
        joint_observation_spaces: Dict[str, gym.Space],
        joint_action_spaces: Dict[str, gym.Space],
    ):
        self.num_envs = num_envs
        self.is_vector_env = True
        self.parallel_joint_observation_space = {agent: batch_space(joint_observation_spaces[agent], n=num_envs) for agent in agent_ids}
        self.parallel_joint_action_space = {agent: batch_space(joint_action_spaces[agent], n=num_envs) for agent in agent_ids}

        self.closed = False
        self.viewer = None

        # The observation and action spaces of a single environment are
        # kept in separate properties
        self.single_joint_observation_space = joint_observation_spaces
        self.single_joint_action_space = joint_action_spaces

    def reset_async(
        self,
        seed: Optional[Union[int, List[int]]] = None,
        options: Optional[dict] = None,
    ):
        pass

    def reset_wait(
        self,
        seed: Optional[Union[int, List[int]]] = None,
        options: Optional[dict] = None,
    ):
        raise NotImplementedError("VectorEnv does not implement function")

    def reset(
        self,
        *,
        seed: Optional[Union[int, List[int]]] = None,
        options: Optional[dict] = None,
    ):
        self.reset_async(seed=seed, options=options)
        return self.reset_wait(seed=seed, options=options)

    def step_async(self, actions):
        """Asynchronously performs steps in the sub-environments.

        The results can be retrieved via a call to :meth:`step_wait`.

        Args:
            actions: The actions to take asynchronously
        """

    def step_wait(
        self, **kwargs
    ) -> Tuple[Any, NDArray[Any], NDArray[Any], NDArray[Any], dict]:
        raise NotImplementedError()

    def step(
        self, actions
    ) -> Tuple[
        Dict[str, Any],
        Dict[str, NDArray[Any]],
        Dict[str, NDArray[Any]],
        Dict[str, NDArray[Any]],
        Dict[str, dict]]:

        self.step_async(actions)
        return self.step_wait()

    def call_async(self, name, *args, **kwargs):
        """Calls a method name for each parallel environment asynchronously."""

    def call_wait(self, **kwargs) -> List[Any]:  # type: ignore
        """After calling a method in :meth:`call_async`, this function collects the results."""

    def call(self, name: str, *args, **kwargs) -> List[Any]:
        self.call_async(name, *args, **kwargs)
        return self.call_wait()

    def get_attr(self, name: str):
        return self.call(name)

    def set_attr(self, name: str, values: Union[list, tuple, object]):
        """Set a property in each sub-environment.

        Args:
            name (str): Name of the property to be set in each individual environment.
            values (list, tuple, or object): Values of the property to be set to. If `values` is a list or
                tuple, then it corresponds to the values for each individual environment, otherwise a single value
                is set for all environments.
        """

    def close_extras(self, **kwargs):
        """Clean up the extra resources e.g. beyond what's in this base class."""
        pass

    def close(self, **kwargs):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras(**kwargs)
        self.closed = True

    def _add_info(self, infos: dict, info: dict, env_num: int) -> dict:
        # info の中身が ndarrayを仮定？
        for k in info.keys():
            if k not in infos:
                info_array, array_mask = self._init_info_arrays(type(info[k]))
            else:
                info_array, array_mask = infos[k], infos[f"_{k}"]

            info_array[env_num], array_mask[env_num] = info[k], True
            infos[k], infos[f"_{k}"] = info_array, array_mask
        return infos

    def _init_info_arrays(self, dtype: type) -> Tuple[np.ndarray, np.ndarray]:
        if dtype in [int, float, bool] or issubclass(dtype, np.number):
            array = np.zeros(self.num_envs, dtype=dtype)
        else:
            array = np.zeros(self.num_envs, dtype=object)
            array[:] = None
        array_mask = np.zeros(self.num_envs, dtype=bool)
        return array, array_mask

    def __del__(self):
        if not getattr(self, "closed", True):
            self.close()

    def __repr__(self) -> str:
        if self.spec is None:
            return f"{self.__class__.__name__}({self.num_envs})"
        else:
            return f"{self.__class__.__name__}({self.spec.id}, {self.num_envs})"

