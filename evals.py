import baba
from all_envs import init_goto_win_envs, init_make_win, init_two_room, init_two_room_envs_break_stop, init_two_room_envs_break_stop_make_win, init_two_room_anywhere, init_two_room_make_you_make_win
import numpy as np
import wandb
from gym.wrappers import RecordVideo
# Given a method, evaluate it using a test set of all the environments.
# 25 times per env, random everything, and evaluate it

def evaluate_model(model, n=25, start_size=7, end_size=7):
    # 8x8 -> 12x12
    env_constructors = [init_two_room]#[init_goto_win_envs, init_two_room, init_two_room_envs_break_stop, init_two_room_anywhere]
                        #, init_make_win, init_two_room]#, init_two_room_envs_break_stop,
                        #init_two_room_envs_break_stop_make_win, init_two_room_anywhere, init_two_room_make_you_make_win]
    descriptions = ["two_room"]#["goto_win", "two_room", "two_room_break_stop", "two_room_anywhere"] 
                    #"two_room_anywhere"]#, "make_win", "two_room"]#, "two_room_break_stop", "two_room_break_stop_make_win",] 
                    #"two_room_anywhere", "two_room_make_you_make_win"]
    for size in range(start_size, end_size + 1):
        for constructor, description in zip(env_constructors, descriptions):
            envs_dict = constructor(width=size, height=size)
            for difficulty, environments in envs_dict.items():
                for env_index, env in enumerate(environments):
                    env_key = f"{description}_{difficulty}_{size}x{size}"
                    #env = RecordVideo(
                    #env,
                    #video_folder=f"videos/eval/{env_key}",
                    #episode_trigger=lambda x: True
                    #)
                    scores = []
                    passes = 0
                    total_steps = 0
                    for _ in range(n):
                        obs, _ = env.reset()
                        steps = 0
                        done = False
                        total_reward = 0
                        while not done:
                            action = model.get_action(obs)
                            obs, reward, done, _, _ = env.step(action)
                            total_reward += reward
                            steps += 1
                        scores.append(total_reward)
                        passes += (total_reward > 0)
                        total_steps += steps
                    
                    # Calculate statistics
                    mean_score = np.mean(scores)
                    iqr = np.subtract(*np.percentile(scores, [75, 25]))
                    average_steps = total_steps / n
                    print("evaluating:", env_key)
                    # Log detailed statistics
                    wandb.log({
                        f"{env_key}_mean_score": mean_score,
                        f"{env_key}_pass_rate": passes / 25,
                        f"{env_key}_average_steps": average_steps,
                        f"{env_key}_iqr": iqr
                    })

if __name__ == '__main__':
    env_constructors = [init_goto_win_envs, init_make_win, init_two_room, init_two_room_envs_break_stop,
                        init_two_room_envs_break_stop_make_win, init_two_room_anywhere, init_two_room_make_you_make_win]
    descriptions = ["goto_win", "make_win", "two_room", "two_room_break_stop", "two_room_break_stop_make_win", 
                    "two_room_anywhere", "two_room_make_you_make_win"]
    aggregate_scores = {}
    evit = 0
    for size in range(5, 12 + 1):
        for constructor, description in zip(env_constructors, descriptions):
            envs_dict = constructor(width=size, height=size)
            for difficulty, environments in envs_dict.items():
                for env_index, env in enumerate(environments):
                    env_key = f"{description}_{difficulty}_{size}x{size}"
                    env = RecordVideo(
                    env,
                    video_folder=f"videos/eval/{env_key}",
                    episode_trigger=lambda x: True
                    )
                    scores = []
                    passes = 0
                    total_steps = 0
                    for _ in range(n):
                        obs, _ = env.reset()
                        steps = 0
                        done = False
                        total_reward = 0
                        while not done:
                            action = model.get_action(obs)
                            obs, reward, done, _, _ = env.step(action)
                            total_reward += reward
                            steps += 1
                        scores.append(total_reward)
                        passes += (total_reward > 0)
                        total_steps += steps
                    
                    # Calculate statistics
                    mean_score = np.mean(scores)
                    iqr = np.subtract(*np.percentile(scores, [75, 25]))
                    average_steps = total_steps / n
                    print("evaluating:", env_key)
                    # Log detailed statistics
                    wandb.log({
                        f"{env_key}_mean_score": mean_score,
                        f"{env_key}_passes": passes,
                        f"{env_key}_average_steps": average_steps,
                        f"{env_key}_iqr": iqr
                    })

                    # Aggregate statistics across all sizes and difficulties
                    aggregate_scores.setdefault(f"{description}_{difficulty}", []).append(mean_score)
                    aggregate_scores.setdefault(f"{description}_all", []).append(mean_score)

    # Log aggregated statistics
    for key, values in aggregate_scores.items():
        wandb.log({
            f"aggregate_{key}_mean_score": np.mean(values),
            f"aggregate_{key}_iqr": np.subtract(*np.percentile(values, [75, 25]))
        })
