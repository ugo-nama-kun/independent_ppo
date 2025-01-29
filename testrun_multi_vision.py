
import random
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import tyro
from gridhunt.envs import MultiHunt2APZ, MultiHunt1APZ

from ippo_lstm_vision import IPPO_LSTM_VISION
from sync_vector_ma_env import SyncVectorMAEnv
from utils import dict_detach, dict_cpu_numpy, dict_tensor
from wrapper import RecordParallelEpisodeStatistics


@dataclass
class Args:
    env_id: str = "MultiHunt2APZ"
    # Algorithm specific arguments
    capture_video: bool = False
    # remove agent name and .pth to identify model path
    file_path: str = "saved_models/MultiHunt2APZ-v0__running_multi_vision_example__1__1738137439/final"
    seed: int = 42
    torch_deterministic: bool = True
    num_envs = 1


def make_env(env_id):
    def thunk():
        env = MultiHunt2APZ(render_mode="human") if "MultiHunt2APZ" in env_id else MultiHunt1APZ(render_mode="human")
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
        [make_env(args.env_id)],
    )
    for action_space in ma_envs.single_joint_action_space.values():
        assert isinstance(action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # agent = Agent(envs).to(device)
    ippo_agent = IPPO_LSTM_VISION(ma_envs, device, args, run_name="test", test=True)
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
