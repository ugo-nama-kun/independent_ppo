from typing import Dict

import numpy as np
import torch

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