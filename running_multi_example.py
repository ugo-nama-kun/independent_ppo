# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_lstmpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import tyro

from torch.utils.tensorboard import SummaryWriter
from pettingzoo.mpe import simple_reference_v3

from ippo_lstm import IPPO_LSTM
from sync_vector_ma_env import SyncVectorMAEnv
from utils import dict_detach, dict_cpu_numpy, dict_tensor
from wrapper import RecordParallelEpisodeStatistics


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "SimpleReference"
    """the id of the environment"""
    total_timesteps: int = 1_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 0.003
    """the learning rate of the optimizer"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env():
    def thunk():
        env = simple_reference_v3.parallel_env(continuous_actions=False)
        # env = gym.wrappers.RecordEpisodeStatistics(env)
        env = RecordParallelEpisodeStatistics(env)
        return env

    return thunk


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    ma_envs = SyncVectorMAEnv(
        [make_env() for i in range(args.num_envs)],
    )
    for action_space in ma_envs.single_joint_action_space.values():
        assert isinstance(action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # agent = Agent(envs).to(device)
    ippo_agent = IPPO_LSTM(ma_envs, device, args)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = ma_envs.reset(seed=args.seed)
    next_obs = {agent : torch.Tensor(next_obs[agent]).to(device) for agent in next_obs.keys()}
    next_done = {agent: torch.zeros(args.num_envs).to(device) for agent in ma_envs.ma_envs[0].possible_agents}

    # reset ppo lstm state
    ippo_agent.reset_lstm_state()

    for iteration in range(1, args.num_iterations + 1):
        # saving initial lstm state of the rollout
        ippo_agent.save_initial_lstm_state()

        # Annealing the rate if instructed to do so.
        ippo_agent.update_learning_rate(iteration)

        for step in range(0, args.num_steps):
            global_step += args.num_envs

            # ALGO LOGIC: action logic
            action, logprob, values = ippo_agent.get_action_and_value(next_obs, next_done)

            # copy previous obs and done
            next_done_prev, next_obs_prev = dict_detach(next_done), dict_detach(next_obs)

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = ma_envs.step(dict_cpu_numpy(action))

            # store experience here
            ippo_agent.collect(step, next_done_prev, next_obs_prev, action, logprob, values, reward)

            next_done = np.logical_or(terminations, truncations)
            next_obs, next_done = dict_tensor(next_obs, device), dict_tensor(next_done, device)

            for env_id in range(args.num_envs):
                for agent_id in ma_envs.possible_agents:
                    if "final_info" in infos[agent_id][env_id]:
                        info = infos[agent_id][env_id]["final_info"]
                        if "episode" in info:
                            print(f"global_step={global_step}, episodic_return={info['episode']['r']}, episodic_length={info['episode']['l']}")
                            writer.add_scalar(f"charts/episodic_return/{agent_id}", info["episode"]["r"], global_step)
                            writer.add_scalar(f"charts/episodic_length/{agent_id}", info["episode"]["l"], global_step)

        # optimize
        metrics = ippo_agent.update(next_obs, next_done)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        for agent_id in metrics.keys():
            writer.add_scalar(f"charts/learning_rate/{agent_id}", metrics[agent_id][0], global_step)
            writer.add_scalar(f"losses/value_loss/{agent_id}", metrics[agent_id][1], global_step)
            writer.add_scalar(f"losses/policy_loss/{agent_id}", metrics[agent_id][2], global_step)
            writer.add_scalar(f"losses/entropy/{agent_id}", metrics[agent_id][3], global_step)
            writer.add_scalar(f"losses/old_approx_kl/{agent_id}", metrics[agent_id][4], global_step)
            writer.add_scalar(f"losses/approx_kl/{agent_id}", metrics[agent_id][5], global_step)
            writer.add_scalar(f"losses/clipfrac/{agent_id}", metrics[agent_id][6], global_step)
            writer.add_scalar(f"losses/explained_variance/{agent_id}", metrics[agent_id][7], global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    ma_envs.close()
    writer.close()