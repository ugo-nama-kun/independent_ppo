import datetime
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.distributions.categorical import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Model(nn.Module):
    def __init__(self, action_space, observation_space):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(np.prod(observation_space.shape), 256)),
            nn.ELU(),
            layer_init(nn.Linear(256, 128)),
            nn.ELU(),
        )
        self.lstm = nn.LSTM(128, 64)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        self.actor = layer_init(nn.Linear(64, action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(64, 1), std=1)

    def get_states(self, x, lstm_state, done):
        hidden = self.network(x)

        # LSTM logic
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state

    def get_value(self, x, lstm_state, done):
        hidden, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden)

    def get_action_and_value(self, x, lstm_state, done, action=None):
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), lstm_state


class PPO_LSTM:
    def __init__(self, agent_id, env, device, args):
        self.agent_id = agent_id
        self.args = args
        self.device = device
        self.single_action_space = env.action_spaces[agent_id]
        self.single_observation_space = env.observation_spaces[agent_id]

        self.model = Model(self.single_action_space, self.single_observation_space).to(device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate, eps=1e-5)

        # ALGO Logic: Storage setup
        self.obs = torch.zeros((args.num_steps, args.num_envs) + self.single_observation_space.shape).to(device)
        self.actions = torch.zeros((args.num_steps, args.num_envs) + self.single_action_space.shape).to(device)
        self.logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
        self.rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
        self.dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
        self.values = torch.zeros((args.num_steps, args.num_envs)).to(device)

        self.lstm_state = (
            torch.zeros(self.model.lstm.num_layers, args.num_envs, self.model.lstm.hidden_size).to(device),
            torch.zeros(self.model.lstm.num_layers, args.num_envs, self.model.lstm.hidden_size).to(device),
        )  # hidden and cell states (see https://youtu.be/8HyCNIVRbSU)

        # saving lstm state at the start of rollouts
        self.initial_lstm_state = (
            self.lstm_state[0].clone(),
            self.lstm_state[1].clone()
        )

    def reset_lstm_state(self):
        self.lstm_state = (
            torch.zeros(self.model.lstm.num_layers, self.args.num_envs, self.model.lstm.hidden_size).to(self.device),
            torch.zeros(self.model.lstm.num_layers, self.args.num_envs, self.model.lstm.hidden_size).to(self.device),
        )  # hidden and cell states (see https://youtu.be/8HyCNIVRbSU)

    def save_initial_lstm_state(self):
        # saving initial lstm state of the rollout
        self.initial_lstm_state = (
            self.lstm_state[0].clone(),
            self.lstm_state[1].clone()
        )

    def update_learning_rate(self, iteration: int):
        if self.args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / self.args.num_iterations
            lrnow = frac * self.args.learning_rate
            self.optimizer.param_groups[0]["lr"] = lrnow

    def get_action_and_value(self, obs, done):
        with torch.no_grad():
            action, logprob, _, value, self.lstm_state = self.model.get_action_and_value(
                obs, self.lstm_state, done
            )
            value = value.flatten()
        return action, logprob, value

    def collect(self, step: int, done, obs, action, logprob, value, reward):
        self.obs[step] = obs
        self.dones[step] = done
        self.actions[step] = action
        self.logprobs[step] = logprob
        self.values[step] = value
        self.rewards[step] = torch.tensor(reward).to(self.device).view(-1)

    def update(self, next_obs, next_done):
        # bootstrap value if not done
        with torch.no_grad():
            next_value = self.model.get_value(
                next_obs,
                self.lstm_state,
                next_done,
            ).reshape(1, -1)
            advantages = torch.zeros_like(self.rewards).to(self.device)
            lastgaelam = 0
            for t in reversed(range(self.args.num_steps)):
                if t == self.args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.dones[t + 1]
                    nextvalues = self.values[t + 1]
                delta = self.rewards[t] + self.args.gamma * nextvalues * nextnonterminal - self.values[t]
                advantages[
                    t] = lastgaelam = delta + self.args.gamma * self.args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + self.values

        # flatten the batch
        b_obs = self.obs.reshape((-1,) + self.single_observation_space.shape)
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape((-1,) + self.single_action_space.shape)
        b_dones = self.dones.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.values.reshape(-1)

        # Optimizing the policy and value network
        assert self.args.num_envs % self.args.num_minibatches == 0
        envsperbatch = self.args.num_envs // self.args.num_minibatches
        envinds = np.arange(self.args.num_envs)
        flatinds = np.arange(self.args.batch_size).reshape(self.args.num_steps, self.args.num_envs)
        clipfracs = []
        for epoch in range(self.args.update_epochs):
            np.random.shuffle(envinds)
            for start in range(0, self.args.num_envs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mb_inds = flatinds[:, mbenvinds].ravel()  # be really careful about the index

                _, newlogprob, entropy, newvalue, _ = self.model.get_action_and_value(
                    b_obs[mb_inds],
                    (self.initial_lstm_state[0][:, mbenvinds], self.initial_lstm_state[1][:, mbenvinds]),
                    b_dones[mb_inds],
                    b_actions.long()[mb_inds],
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if self.args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if self.args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.args.clip_coef,
                        self.args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.args.ent_coef * entropy_loss + v_loss * self.args.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()

            if self.args.target_kl is not None and approx_kl > self.args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        return (
            self.optimizer.param_groups[0]["lr"],
            v_loss.item(),
            pg_loss.item(),
            entropy_loss.item(),
            old_approx_kl.item(),
            approx_kl.item(),
            approx_kl.item(),
            np.mean(clipfracs),
            explained_var,
        )


class IPPO_LSTM:
    def __init__(self, ma_envs, device, args):
        self.args = args
        self.device = device

        self.possible_agents = ma_envs.ma_envs[0].possible_agents
        self.agents = {
            agent: PPO_LSTM(agent, ma_envs.ma_envs[0], device, args)
            for agent in ma_envs.ma_envs[0].possible_agents
        }

        self.create_at = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")

    def reset_lstm_state(self):
        for agent_ in self.agents.values():
            agent_.reset_lstm_state()

    def save_initial_lstm_state(self):
        for agent_ in self.agents.values():
            agent_.save_initial_lstm_state()

    def update_learning_rate(self, iteration):
        for agent_ in self.agents.values():
            agent_.update_learning_rate(iteration)

    def get_action_and_value(self, obss, dones):
        actions = {}
        logprobs = {}
        values = {}

        for agent_id in self.possible_agents:
            actions[agent_id], logprobs[agent_id], values[agent_id] = self.agents[agent_id].get_action_and_value(
                obss[agent_id], dones[agent_id])

        return actions, logprobs, values

    def collect(self, step: int, dones, obss, actions, logprobs, values, rewards):
        for agent_id in self.possible_agents:
            self.agents[agent_id].collect(
                step,
                dones[agent_id], obss[agent_id], actions[agent_id], logprobs[agent_id], values[agent_id],
                rewards[agent_id]
            )

    def update(self, next_obss, next_dones):
        metrics = {}
        for agent_id in self.possible_agents:
            metrics[agent_id] = self.agents[agent_id].update(next_obss[agent_id], next_dones[agent_id])
        return metrics

    def save_model(self, dir_name=None):
        path = f"saved_models/ippo-{self.create_at}"
        if dir_name is not None:
            path = os.path.join(path, dir_name)
        os.makedirs(path, exist_ok=True)

        for agent_id, agent in self.agents.items():
            filepath = os.path.join(path, f"ippo_{self.args.env_id}_{agent_id}.pth")
            torch.save(agent.model.state_dict(), filepath)
