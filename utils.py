from typing import Dict

import torch

def dict_detach(data_dict: Dict[str, torch.Tensor]):
    return {id_: data_dict[id_].detach() for id_ in data_dict.keys()}


def dict_cpu_numpy(data_dict: Dict[str, torch.Tensor]):
    return {id_: data_dict[id_].cpu().numpy() for id_ in data_dict.keys()}


def dict_tensor(data_dict: Dict[str, torch.Tensor], device: torch.device):
    return {id_: torch.Tensor(data_dict[id_]).to(device) for id_ in data_dict.keys()}


# TODO: use this for evaluation ここから
def test_env_single(agent, test_envs, device, render=False):
    agent.eval()

    episode_reward = 0
    episode_length = 0
    episode_error = 0
    ave_reward = 0

    n_runs = len(test_envs.envs)

    not_done_flags = {i: True for i in range(n_runs)}

    intero_errors = np.zeros(n_runs)
    obs, info = test_envs.reset()
    obs = torch.Tensor(obs).to(device)

    while np.any(list(not_done_flags.values())):

        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(obs)

        obs, reward, done, truncated, info = test_envs.step(action.cpu().numpy())
        done = done | truncated

        if render:
            test_envs.envs[0].render()

        obs = torch.Tensor(obs).to(device)

        if "error" in info.keys():
            try:
                intero_errors += info["error"]
            except TypeError:
                print(info["error"])

        if np.any(done):
            for i in np.where(info["_episode"])[0]:
                if not_done_flags[i] is True:
                    not_done_flags[i] = False
                    print(
                        f"TEST: episodic_return={info['episode']['r'][i]}, episodic_length={info['episode']['l'][i]}")

                    episode_reward += info['episode']['r'][i]
                    episode_length += info['episode']['l'][i]
                    episode_error += intero_errors[i] / info['episode']['l'][i]
                    ave_reward += info['episode']['r'][i] / info['episode']['l'][i]
                    intero_errors[i] = 0

                if np.any(list(not_done_flags.values())) is False:
                    break

    episode_reward /= n_runs
    episode_length /= n_runs
    episode_error /= n_runs
    ave_reward /= n_runs

    agent.train()

    return episode_reward, episode_length, episode_error, ave_reward