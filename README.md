# baba-is-rl

# baba-is-ai
The baba environment is a heavily modified version of this [baba-is-you simulator](https://github.com/nacloos/baba-is-ai). Changes include: removing the "idle" action, modifying the environments to follow the [gymnasium](https://gymnasium.farama.org/) API instead of gym, heavily standardizing level generation and adding new levels, adding in optional randomness for baba and object placement, fixing video saving, allowing step by step play for a model, and simplifying output (currently, status and color are not useful). \
after cloning this repo, run \
`cd baba-is-ai`\
`pip install -e .`

# PPO and DQN
PPO and DQN follow the cleanrl arguments. See their [docs](https://docs.cleanrl.dev/get-started/basic-usage/#two-ways-to-run) for more arguments. \
`python ppo.py --seed 1 --total-timesteps 1000000 --env_id env/make_win`\
`python dqn.py --seed 1 --total-timesteps 1000000 --env_id env/make_win`

# LLM
Using gpt4o-mini requires an API key, so you'll need one to execute it. If you do not have one and want to see this functionality in use, you can message me! \
`python llm.py`

Andrew (andrew.m.wu@gmail.com)
