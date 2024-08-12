import pygame
from gymnasium.utils.play import display_arr
from pygame import VIDEORESIZE
from agents import DQNTrainer, PPOTrainer
from util import load_config
import torch


def play(env, model, transpose=True, fps=30, zoom=None, callback=None):
    # Set up the action keys and environment
    keys_to_action = {
        (pygame.K_UP,): env.actions.up,
        (pygame.K_DOWN,): env.actions.down,
        (pygame.K_LEFT,): env.actions.left,
        (pygame.K_RIGHT,): env.actions.right,
        pygame.K_SPACE: 'step',  # one step at a time
        pygame.K_f: 'fast'       # fast forward to level end
    }

    env.reset()
    rendered = env.render(mode="rgb_array")
    video_size = [rendered.shape[1], rendered.shape[0]]
    if zoom is not None:
        video_size = int(video_size[0] * zoom), int(video_size[1] * zoom)

    screen = pygame.display.set_mode(video_size)
    clock = pygame.time.Clock()
    running = True
    env_done = True
    step = 0
    always_step = False
    while running:
        if env_done:
            env_done = False
            obs, _ = env.reset()
            step = 0
            always_step = False
        # Use model to predict action based on current observation
        action = model.get_action(obs[None, :])
        prev_obs = obs

        # Render and display the current state
        rendered = env.render(mode="rgb_array")
        display_arr(screen, rendered, transpose=transpose, video_size=video_size)

        # Process pygame events
        if always_step:
            # Perform one step using the model's action
            obs, rew, env_done, truncation, info = env.step(action)
            step += 1
            if callback:
                callback(prev_obs, obs, action, rew, env_done, info)
            print(f"Step:{step}, Reward:{rew}")
        else:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == 27:  # ESC key to quit
                        running = False
                    elif event.key in keys_to_action:
                        user_action = keys_to_action[event.key]
                        if user_action == 'step':
                            # Perform one step using the model's action
                            obs, rew, env_done, truncation, info = env.step(action)
                            step += 1
                            if callback:
                                callback(prev_obs, obs, action, rew, env_done, info)
                            print(f"Step:{step}, Reward:{rew}")

                        elif user_action == 'fast':
                            # Fast forward using the model's actions until the end
                            always_step = True
        pygame.display.flip()
        clock.tick(fps)
    pygame.quit()

if __name__ == "__main__":
    import argparse
    import baba
    parser = argparse.ArgumentParser(description="Play Baba Is You")
    parser.add_argument("--env", type=str, default="two_room-break_stop-make_win-distr_obj_rule", help="Environment id")
    parser.add_argument("--name", type=str, help="run name", required=True)
    parser.add_argument("--model", type=str, help="model", required=True)
    args = parser.parse_args()

    config = load_config(f"config/{args.model}.yaml")
    config.track = False
    env = baba.make(f"env/{args.env}")
    if args.model == "ppo":
        model = PPOTrainer(config).load_model(local=True, name=args.name)
    elif args.model == "dqn":
        model = DQNTrainer(config).load_model(local=True, name=args.name)
    elif args.model == "gpt":
        pass
    else:
        raise NotImplementedError
    play(env, model)