import pygame
from gymnasium.utils.play import display_arr
from pygame import VIDEORESIZE
import json
import time
import os
import atexit
import glob
import numpy as np
from dataclasses import dataclass

# Collects player trajectories
# yes, this is very jank.

trajectories = {}
current_index = 0


@dataclass
class Traj_instance:
    prev_obs: np.ndarray
    obs: np.ndarray
    rew: int
    done: bool
    action: int
    name: str

def load_trajectories(directory):
    trajectories = []
    json_files = glob.glob(f"/home/andrew/baba-is-rl/trajectories/{directory}/*.json")    
    for file in json_files:
        with open(file, 'r') as f:
            data = json.load(f)
            for traj in data:
                for t in data[traj]:
                    trajectories.append(Traj_instance(prev_obs=np.array(t["prev_obs"]), obs=np.array(t["obs"]), rew=t["reward"], done=t["done"], action=t["action"], name=str(file)))
    return trajectories

def save_traj(prev_obs, obs, action, rew, env_done, info, env_name):
    global trajectories, current_index
    if prev_obs is not None and obs is not None and action is not None:
        if current_index not in trajectories:
            trajectories[current_index] = []
        trajectories[current_index].append({
            "prev_obs": prev_obs.tolist(),
            "obs": obs.tolist(),
            "action": action,
            "reward": rew,
            "done": env_done
        })

    if env_done:
        print(f"Trajectory {current_index} completed")
        current_index += 1

def save_all_trajectories(env_name):
    filename = f"{env_name}__all_trajectories.json"
    filepath = os.path.join("trajectories", filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(trajectories, f, indent=4)
    print(f"All trajectories saved to {filepath}")


def play(env, transpose=True, fps=30, zoom=None, callback=None, keys_to_action=None, env_name=None):
    if keys_to_action is None:
        keys_to_action = {
            (pygame.K_UP,): env.actions.up,
            (pygame.K_DOWN,): env.actions.down,
            (pygame.K_LEFT,): env.actions.left,
            (pygame.K_RIGHT,): env.actions.right,
        }

    env.reset()
    rendered = env.render(mode="rgb_array")

    if keys_to_action is None:
        if hasattr(env, "get_keys_to_action"):
            keys_to_action = env.get_keys_to_action()
        elif hasattr(env.unwrapped, "get_keys_to_action"):
            keys_to_action = env.unwrapped.get_keys_to_action()
        else:
            assert False, (
                env.spec.id
                + " does not have explicit key to action mapping, "
                + "please specify one manually"
            )
    relevant_keys = set(sum(map(list, keys_to_action.keys()), []))

    video_size = [rendered.shape[1], rendered.shape[0]]
    if zoom is not None:
        video_size = int(video_size[0] * zoom), int(video_size[1] * zoom)

    pressed_keys = []
    running = True
    env_done = True

    screen = pygame.display.set_mode(video_size)
    clock = pygame.time.Clock()

    while running:
        if env_done:
            env_done = False
            obs, _ = env.reset()
        else:
            action = keys_to_action.get(tuple(sorted(pressed_keys)), None) 
            pressed_keys = []
            prev_obs = obs
            if action is not None:
                obs, rew, env_done, truncation, info = env.step(action)
                assert obs.shape == prev_obs.shape
                print("Reward:", rew) if rew != 0 else None
            if callback is not None and action is not None:
                callback(prev_obs, obs, action, rew, env_done, info, env_name)
        if obs is not None:
            rendered = env.render(mode="rgb_array")
            display_arr(screen, rendered, transpose=transpose, video_size=video_size)

        # process pygame events
        for event in pygame.event.get():
            # test events, set key states
            if event.type == pygame.KEYDOWN:
                if event.key in relevant_keys:
                    pressed_keys.append(event.key)
                elif event.key == 27:
                    env_done = True
            elif event.type == pygame.QUIT:
                running = False
            elif event.type == VIDEORESIZE:
                video_size = event.size
                screen = pygame.display.set_mode(video_size)
                print(video_size)

        pygame.display.flip()
        clock.tick(fps)
    pygame.quit()


if __name__ == "__main__":
    import argparse
    import baba
    parser = argparse.ArgumentParser(description="Play Baba Is You")
    parser.add_argument("--env", type=str, default="two_room-break_stop-make_win-distr_obj_rule", help="Environment id")
    args = parser.parse_args()

    env = baba.make(f"env/{args.env}")
    atexit.register(save_all_trajectories, args.env)
    play(env, env_name=args.env, callback=save_traj)