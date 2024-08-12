import gymnasium as gym
import baba
import yaml
from dataclasses import make_dataclass

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = baba.make(f"{env_id}", render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = baba.make(f"{env_id}")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

        return env

    return thunk

def load_config(path):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
        cfg = make_dataclass(
            "exp_args", ((k, type(v)) for k, v in config.items())
        )(**config)
        return cfg