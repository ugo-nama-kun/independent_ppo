
import random
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import tyro
from pettingzoo.mpe import simple_speaker_listener_v4

from ippo_lstm import IPPO_LSTM
from sync_vector_ma_env import SyncVectorMAEnv
from utils import dict_detach, dict_cpu_numpy, dict_tensor
from wrapper import RecordParallelEpisodeStatistics


@dataclass
class Args:
    # Algorithm specific arguments
    capture_video: bool = False
    # remove agent name and .pth to identify model path
    file_path: str = "saved_models/ippo-2024-10-21_16-33-06-466824/20/ippo_SimpleSpeakerListener-vv"
    seed: int = 42
    torch_deterministic: bool = True
    num_envs = 1


def make_env():
    def thunk():
        env = simple_speaker_listener_v4.parallel_env(continuous_actions=False, render_mode="human")
        env = RecordParallelEpisodeStatistics(env)
        return env

    return thunk


if __name__ == "__main__":
    args = tyro.cli(Args)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    ma_envs = SyncVectorMAEnv(
        [make_env()],
    )
    for action_space in ma_envs.single_joint_action_space.values():
        assert isinstance(action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # agent = Agent(envs).to(device)
    ippo_agent = IPPO_LSTM(ma_envs, device, args, test=True)
    ippo_agent.load_model(args.file_path)
    ippo_agent.eval()

    next_obs, _ = ma_envs.reset(seed=args.seed)
    next_obs = {agent: torch.Tensor(next_obs[agent]).to(device) for agent in next_obs.keys()}
    next_done = {agent: torch.zeros(args.num_envs).to(device) for agent in ma_envs.ma_envs[0].possible_agents}
    ippo_agent.reset_lstm_state()

    done = False
    step = 0
    while not done:
        # ALGO LOGIC: action logic
        action, logprob, values = ippo_agent.get_action_and_value(next_obs, next_done)

        # copy previous obs and done
        next_done_prev, next_obs_prev = dict_detach(next_done), dict_detach(next_obs)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, terminations, truncations, infos = ma_envs.step(dict_cpu_numpy(action))
        # ma_envs.render()

        next_done = np.logical_or(terminations, truncations)
        next_obs, next_done = dict_tensor(next_obs, device), dict_tensor(next_done, device)

        step += 1

    ma_envs.close()
