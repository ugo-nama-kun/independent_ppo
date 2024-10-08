
import random
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import tyro
from stable_baselines3.common.atari_wrappers import NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, FireResetEnv, \
    ClipRewardEnv

from ppo_lstm import PPO_LSTM
from ppo_lstm_vision import PPO_LSTM_V


@dataclass
class Args:
    # Algorithm specific arguments
    env_id: str = "BreakoutNoFrameskip-v4"
    capture_video: bool = False
    file_path: str = "saved_models/ppo-vision_2024-10-08_17-59-35-075728/final/ppo-vision_BreakoutNoFrameskip-v4.pth"
    seed: int = 42
    torch_deterministic: bool = True
    num_envs = 1


def make_env(env_id):
    def thunk():
        env = gym.make(env_id, render_mode="human")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 1)
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
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # agent = Agent(envs).to(device)
    ppo_agent = PPO_LSTM_V(envs, device, args, test=True)
    ppo_agent.load_model(args.file_path)
    ppo_agent.eval()
    ppo_agent.reset_lstm_state()

    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    done = False
    step = 0
    while not done:
        # ALGO LOGIC: action logic
        action, logprob, values = ppo_agent.get_action_and_value(next_obs, next_done)

        # copy previous obs and done
        next_done_prev, next_obs_prev = next_done.detach(), next_obs.detach()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
        envs.envs[0].render()

        done = np.logical_or(terminations, truncations)
        next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

        step += 1

    envs.close()
