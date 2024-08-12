import baba


# Given a method, evaluate it using a test set of all the environments.
# 25 times per env, random everything, and evaluate it
def evaluate_model(model, get_action, random_baba=True, random_obj=True, n=25):
    
    for env in envs:
        
        info = [] # (win, total_steps)
        for _ in n:
            steps = 0

            if done and reward == 1:
                info.append((1, steps))
        