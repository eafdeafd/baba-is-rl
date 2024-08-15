# baba-is-rl

# baba-is-ai
The baba environment is a heavily modified version of this [baba-is-you simulator](https://github.com/nacloos/baba-is-ai). Changes include: removing the "idle" action, modifying the environments to follow the [gymnasium](https://gymnasium.farama.org/) API instead of gym, heavily standardizing level generation and adding new levels, adding in optional randomness for baba and object placement, fixing video saving, allowing step by step play for a model, and simplifying output (currently, status and color are not useful). \
after cloning this repo, run \
`cd baba-is-ai`\
`pip install -e .`
`pip install cleanrl`

# PPO and DQN
PPO and DQN follow the cleanrl arguments. See their [docs](https://docs.cleanrl.dev/get-started/basic-usage/#two-ways-to-run) for more arguments. To train on a single environment run and in the config/ yaml file for each respective model, set cumulative_train=False: \
`python agents.py --name dqn`\
`python agents.py --name ppo`\
To train using cumulative learning set it to true in the config file and run: \
`python agents.py`  \
The config file and editing the training loop is useful for narrowing down which environments should be included in the cumulative trainer \
A list of full environments can be found in the baba-is-ai package under evs.py

# Collecting trajectories
DQN can take in expert trajectories using the seed_traj_dirs arugment in the config file. To collect run: \
`python collect_traj.py --env xxxx` \
It will automatically dump these trajectories into trajectories/

# Testing models
If you want to test a model, and visualize it, run: \
`python play_model.py --env xxx --name run_name --model ppo/dqn/gpt` \
The script attempts to load your model from local storage with the same run name under models/env folder. \
Credit to Cloos et al. and Huang et al. for the base Baba simulator and CleanRL implementations.\
Andrew (andrew.m.wu@gmail.com)
