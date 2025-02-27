# PPO/IPPO implementations

## Contents
Basically code implementations are extracted from [CleanRL reference code](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari.py).

Single-agent settings are [Gymnasium](https://gymnasium.farama.org/) compatible.

Multi-agent settings are [PettingZoo](https://pettingzoo.farama.org/) compatible.

### PPO Single Agent + LSTM

- `ppo_lstm` (usage: [running_single_example.py](running_ppo_lstm_example.py))

PPO agent with LSTM memory cells. Discrete action.

- `ppo_lstm_vision.py` (usage: [running_single_vision_example.py](running_ppo_lstm_vision_example.py))

PPO agent with LSTM memory cells. Discrete action. Vision inputs.

### Independent PPO Multi Agent

- `ippo` (usage: [running_ippo_example.py](running_ippo_example.py))

Multi-agent setting. 
[Independent PPO](https://arxiv.org/abs/2011.09533) agent. Discrete action.

- `ippo_lstm` (usage: [running_ippo_lstm_example.py](running_ippo_lstm_example.py))

Multi-agent setting.
[Independent PPO](https://arxiv.org/abs/2011.09533) agent with LSTM memory cells. Discrete action.

- `ippo_lstm_vision.py` (usage: [running_ippo_lstm_vision_example.py](running_ippo_lstm_vision_example.py))

Multi-agent setting.
[Independent PPO](https://arxiv.org/abs/2011.09533) agent with LSTM memory cells. Discrete action. Vision inputs.

# Mimimum codes for running multi-agent custom setting
```text
ippo_lstm.py/ ippo_lstm_vision.py
utils.py
vector_ma_env.py
sync_vector_ma_env.py
```

# Requirements
```text
gym==0.23.1
gymnasium==0.28.1
numpy==1.26.4
opencv-python==4.11.0.86
pettingzoo==1.24.3
stable_baselines3==2.4.1
tensorboard==2.18.0
torch==2.5.1
tyro==0.9.11

# To run running_multi_vision_example.py, you need to pip-install the below package
https://github.com/ugo-nama-kun/gridhunt
```

# Reference
[ClearnRL] https://github.com/vwxyzjn/cleanrl

[PettingZoo] https://pettingzoo.farama.org/

[Independent PPO] https://arxiv.org/abs/2011.09533