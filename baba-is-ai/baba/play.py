import pygame
from gymnasium.utils.play import display_arr
from pygame import VIDEORESIZE


def play(env, transpose=True, fps=30, zoom=None, callback=None, keys_to_action=None):
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
            if callback is not None:
                callback(prev_obs, obs, action, rew, env_done, info)
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
    from baba import make

    parser = argparse.ArgumentParser(description="Play Baba Is You")
    parser.add_argument("--env", type=str, default="two_room-break_stop-make_win-distr_obj_rule", help="Environment id")
    parser.add_argument("--size", type=int, default=3)
    args = parser.parse_args()

    env = make(f"env/{args.env}", width=args.size, height=args.size)
    play(env)
