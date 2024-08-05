from collect_traj import load_trajectories

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="trajectories/goto_win")
    args = parser.parse_args()
    load_trajectories(args.dir)