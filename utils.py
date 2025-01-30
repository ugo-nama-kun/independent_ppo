from typing import Dict

import numpy as np
import torch

from sync_vector_ma_env import SyncVectorMAEnv


def dict_detach(data_dict: Dict[str, torch.Tensor]):
    return {id_: data_dict[id_].detach() for id_ in data_dict.keys()}


def dict_cpu_numpy(data_dict: Dict[str, torch.Tensor]):
    return {id_: data_dict[id_].cpu().numpy() for id_ in data_dict.keys()}


def dict_tensor(data_dict: Dict[str, torch.Tensor], device: torch.device):
    return {id_: torch.Tensor(data_dict[id_]).to(device) for id_ in data_dict.keys()}


def test_env_single(agent, test_envs, device, render=False):
    agent.eval()

    episode_reward = 0.0
    episode_length = 0.0
    ave_reward = 0.0

    n_runs = len(test_envs.envs)

    not_done_flags = {i: True for i in range(n_runs)}

    obs, info = test_envs.reset()
    obs = torch.Tensor(obs).to(device)
    done = torch.Tensor([False,]*n_runs).to(device)

    while np.any(list(not_done_flags.values())):

        with torch.no_grad():
            action, _, _ = agent.get_action_and_value(obs, done)

        obs, reward, done, truncated, info = test_envs.step(action.cpu().numpy())
        done = done | truncated

        if render:
            test_envs.envs[0].render()

        obs = torch.Tensor(obs).to(device)
        done = torch.Tensor(done).to(device)

        if np.any(done.cpu().numpy()):
            for i in np.where(info["final_info"])[0]:
                if not_done_flags[i] is True:
                    not_done_flags[i] = False
                    info_ = info["final_info"][i]
                    print(
                        f"TEST: episodic_return={info_['episode']['r']}, episodic_length={info_['episode']['l']}")

                    episode_reward += info_['episode']['r']
                    episode_length += info_['episode']['l']
                    ave_reward += info_['episode']['r'] / info_['episode']['l']

                if np.any(list(not_done_flags.values())) is False:
                    break

    episode_reward /= n_runs
    episode_length /= n_runs
    ave_reward /= n_runs

    agent.train()

    return episode_reward, episode_length, ave_reward


def test_env_multi(agent, test_envs: SyncVectorMAEnv, device, render=False):
    agent.eval()
    possible_agents = test_envs.possible_agents

    episode_reward = {agent_id: 0.0 for agent_id in possible_agents}
    episode_length = {agent_id: 0.0 for agent_id in possible_agents}
    ave_reward = {agent_id: 0.0 for agent_id in possible_agents}

    n_runs = len(test_envs.ma_envs)
    not_done_flags = {i: True for i in range(n_runs)}

    obs, info = test_envs.reset()
    obs = {agent: torch.Tensor(obs[agent]).to(device) for agent in obs.keys()}
    done = {agent: torch.zeros(n_runs).to(device) for agent in possible_agents}

    while np.any(list(not_done_flags.values())):

        with torch.no_grad():
            action, _, _ = agent.get_action_and_value(obs, done)

        obs, reward, done, truncated, infos = test_envs.step(dict_cpu_numpy(action))
        done = done | truncated

        if render:
            test_envs.ma_envs[0].render()

        done = np.logical_or(done, truncated)
        obs, done = dict_tensor(obs, device), dict_tensor(done, device)

        for env_id in range(n_runs):
            if not_done_flags[env_id] is True:
                for agent_id in possible_agents:
                    if "final_info" in infos[agent_id][env_id]:
                        not_done_flags[env_id] = False
                        info = infos[agent_id][env_id]["final_info"]
                        print(
                            f"TEST-{agent_id}: episodic_return={info['episode']['r']}, episodic_length={info['episode']['l']}")
                        episode_reward[agent_id] += info['episode']['r']
                        episode_length[agent_id] += info['episode']['l']
                        ave_reward[agent_id] += info['episode']['r'] / info['episode']['l']

    for agent_id in possible_agents:
        episode_reward[agent_id] /= n_runs
        episode_length[agent_id] /= n_runs
        ave_reward[agent_id] /= n_runs

    agent.train()

    return episode_reward, episode_length, ave_reward