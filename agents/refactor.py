import argparse
import random
import time
from typing import Dict, Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.buffers import ReplayBuffer
import baba
import einops
import yaml
from util import make_env
from dataclasses import make_dataclass
import wandb

class Agent(nn.Module):
    def __init__(self, envs, device=None):
        super().__init__()
        self.action_space = envs.single_action_space.n
        self.obs_space = envs.single_observation_space.shape
        self.envs = envs
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def preprocess(self, obs):
        obs = torch.Tensor(obs).to(self.device)
        NUM_OBJECTS = 20.0
        obs = obs.float() / NUM_OBJECTS # normalize by number of objects (this is jank - do not hardcode)
        obs = einops.rearrange(obs, 'b w h c -> b c h w')  # Change shape to (batch_size, channels, height, width)
        obs = obs[:, 0, :, :].unsqueeze(1) # ignore channels besides objects (color, status doesn't matter for now)
        return obs

    def get_action(self, obs):
        raise NotImplementedError

class RandomAgent(Agent):
    def __init__(self, envs, device=None, **kwargs):
        super().__init__(envs, device)

    def get_action(self, obs):
        return np.array([self.envs.single_action_space.sample()])

class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.single_action_space.n),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)
    
class QCNNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        input_channels = 1 #env.single_observation_space.shape[2]
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=8, kernel_size=3, stride=1)
        self.final_conv = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, stride=1)
        #self.conv3 = nn.Conv2d(in_channels=32, out_channels=, kernel_size=3, stride=1)
        # try and ignore channels 

        # Calculate the size of the flattened features after convolutions
        def conv2d_size_out(size, kernel_size=3, stride=1, n=1):
            if n == 0: 
                return size
            return conv2d_size_out((size - (kernel_size - 1) - 1) // stride + 1, n=n-1)

        convw = conv2d_size_out(env.single_observation_space.shape[0], n=2)
        convh = conv2d_size_out(env.single_observation_space.shape[1], n=2)
        linear_input_size = convw * convh * self.final_conv.out_channels

        self.fc1 = nn.Linear(linear_input_size, 128)
        self.fc2 = nn.Linear(128, env.single_action_space.n)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.final_conv(x))
        #x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class DQNAgent(Agent):
    def __init__(self, envs, device=None, **kwargs):
        super().__init__(envs, device)
        self.q_network = QCNNetwork(envs)
        self.envs = envs
        self.epsilon = 0

    def update_epsilon(self, epsilon):
        self.epsilon = epsilon

    def forward(self, obs):
        return self.q_network(self.preprocess(obs))

    def get_action(self, obs):
        if random.random() < self.epsilon:
            #actions = np.array([self.env.single_action_space.sample() for _ in range(self.config.num_envs)])
            actions = np.array([self.envs.single_action_space.sample()])
        else:
            q_values = self.q_network(self.preprocess(obs))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()
        return actions


class PPOAgent(Agent):
    def __init__(self, envs, device=None, **kwargs):
        super().__init__(envs, device)
        input_channels = 1
        obs_shape = np.array(self.envs.single_observation_space.shape)
        obs_shape[2] = input_channels # we get rid of channels

        def conv2d_size_out(size, kernel_size=[3], stride=[1], n=1):
            if n == 0: 
                return size
            return conv2d_size_out((size - (kernel_size[-n] - 1) - 1) // stride[-n] + 1, n=n-1)

        convw = conv2d_size_out(obs_shape[0], [4,3], [2,1], n=2)
        convh = conv2d_size_out(obs_shape[1], [4,3],[2,1], n=2)
        linear_input_size = convw * convh * 64 # final channel out
        self.actor = nn.Sequential(
            self.layer_init(nn.Conv2d(input_channels, 32, 4, stride=2)),
            nn.ReLU(),
            self.layer_init(nn.Conv2d(32, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            self.layer_init(nn.Linear(linear_input_size, 512)),
            nn.ReLU(),
            self.layer_init(nn.Linear(512, self.envs.single_action_space.n), std=0.01)
        )
        self.critic = nn.Sequential(
            self.layer_init(nn.Conv2d(input_channels, 32, 4, stride=2)),
            nn.ReLU(),
            self.layer_init(nn.Conv2d(32, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            self.layer_init(nn.Linear(linear_input_size, 512)),
            nn.ReLU(),
            self.layer_init(nn.Linear(512, 1), std=1)
        )
    def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer
    
    def forward(self, obs):
        return self.actor(self.preprocess(obs))

    def get_action(self, obs):
        return self.get_action_and_value(obs)[0]
    
    def get_value(self, x):
        return self.critic(self.preprocess(x))
    
    def get_action_and_value(self, x, action=None):
        logits = self.actor(self.preprocess(x))
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(self.preprocess(x))


class Trainer:
    def __init__(self, config):
        self.config = config
        self.run_name = f"{config.env_id}__{config.exp_name}__{config.seed}__{int(time.time())}"
        if config.track:
            wandb.init(
                project=config.wandb_project_name,
                entity=config.wandb_entity,
                sync_tensorboard=True,
                config=vars(config),
                name=self.run_name,
                monitor_gym=True,
                save_code=True,
            )
        self.writer = SummaryWriter(f"runs/{self.run_name}")
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(config).items()])),
        )

        # TRY NOT TO MODIFY: seeding
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.backends.cudnn.deterministic = config.torch_deterministic
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.cuda else "cpu")
        #assert config.num_envs == 1, "vectorized env not supported"
        self.envs = gym.vector.SyncVectorEnv([make_env(config.env_id, config.seed + i, i, config.capture_video, self.run_name) for i in range(config.num_envs)])

    def __del__(self):
        self.envs.close()
        self.writer.close()

class DQNTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)
        self.q_network = DQNAgent(self.envs, self.device).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
        self.rb = ReplayBuffer(
            config.buffer_size,
            self.envs.single_observation_space,
            self.envs.single_action_space,
            self.device,
            handle_timeout_termination=False,
        )
        # init of misc stuff

    def linear_schedule(self, start_e: float, end_e: float, duration: int, t: int):
        slope = (end_e - start_e) / duration
        return max(slope * t + start_e, end_e)

    def train(self):
        #TODO: seed DQN with imitation learning from human.

        start_time = time.time()        
        target_network = QCNNetwork(self.envs)
        target_network.load_state_dict(self.q_network.q_network.state_dict())
        # TRY NOT TO MODIFY: start the game
        obs, _ = self.envs.reset(seed=self.config.seed)
        #print("Initial observation shape:", obs.shape)
        for global_step in range(self.config.total_timesteps):
            # ALGO LOGIC: put action logic here
            self.q_network.update_epsilon(self.linear_schedule(self.config.start_e, self.config.end_e, self.config.exploration_fraction * self.config.total_timesteps, global_step))
            actions = self.q_network.get_action(obs)

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, terminations, truncations, infos = self.envs.step(actions)
            # TRY NOT TO MODIFY: record rewards for plotting purposes
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        self.writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        self.writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            real_next_obs = next_obs.copy()
            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next_obs[idx] = infos["final_observation"][idx]
            self.rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

            # ALGO LOGIC: training.
            if global_step > self.config.learning_starts:
                if global_step % self.config.train_frequency == 0:
                    data = self.rb.sample(self.config.batch_size)
                    with torch.no_grad():
                        target_max, _ = target_network(data.next_observations).max(dim=1)
                        td_target = data.rewards.flatten() + self.config.gamma * target_max * (1 - data.dones.flatten())
                    old_val = self.q_network.q_network(data.observations).gather(1, data.actions).squeeze()
                    loss = F.mse_loss(td_target, old_val)

                    if global_step % 100 == 0:
                        self.writer.add_scalar("losses/td_loss", loss, global_step)
                        self.writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                        print("SPS:", int(global_step / (time.time() - start_time)))
                        self.writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                    # optimize the model
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                # update target network
                if global_step % self.config.target_network_frequency == 0:
                    for target_network_param, q_network_param in zip(target_network.parameters(), self.q_network.q_network.parameters()):
                        target_network_param.data.copy_(
                            self.config.tau * q_network_param.data + (1.0 - self.config.tau) * target_network_param.data
                        )
        return self.q_network
    
    def save_model(self):
        model_path = f"runs/{self.run_name}/{self.config.exp_name}.cleanrl_model"
        torch.save(self.q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.dqn_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            self.config.env_id,
            eval_episodes=10,
            run_name=f"{self.run_name}-eval",
            Model=DQNAgent,
            device=self.device,
            epsilon=0.05,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            self.writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if self.config.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{self.config.env_id}-{self.config.exp_name}-seed{self.config.seed}"
            repo_id = f"{self.config.hf_entity}/{repo_name}" if self.config.hf_entity else repo_name
            push_to_hub(self.config, episodic_returns, repo_id, "DQN", f"runs/{self.run_name}", f"videos/{self.run_name}-eval")

class PPOTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)
        self.agent = PPOAgent(self.envs, self.device).to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=config.learning_rate, eps=1e-5)

    def train(self):
        # ALGO Logic: Storage setup
        obs = torch.zeros((self.config.num_steps, self.config.num_envs) + self.envs.single_observation_space.shape).to(self.device)
        actions = torch.zeros((self.config.num_steps, self.config.num_envs) + self.envs.single_action_space.shape).to(self.device)
        logprobs = torch.zeros((self.config.num_steps, self.config.num_envs)).to(self.device)
        rewards = torch.zeros((self.config.num_steps, self.config.num_envs)).to(self.device)
        dones = torch.zeros((self.config.num_steps, self.config.num_envs)).to(self.device)
        values = torch.zeros((self.config.num_steps, self.config.num_envs)).to(self.device)

        # TRY NOT TO MODIFY: start the game
        global_step = 0
        start_time = time.time()
        next_obs, _ = self.envs.reset(seed=self.config.seed)
        next_obs = torch.Tensor(next_obs).to(self.device)
        next_done = torch.zeros(self.config.num_envs).to(self.device)
        self.config.batch_size = int(self.config.num_envs * self.config.num_steps)
        self.config.minibatch_size = int(self.config.batch_size // self.config.num_minibatches)
        self.config.num_iterations = self.config.total_timesteps // self.config.batch_size

        for iteration in range(1, self.config.num_iterations + 1):
            # Annealing the rate if instructed to do so.
            if self.config.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / self.config.num_iterations
                lrnow = frac * self.config.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, self.config.num_steps):
                global_step += self.config.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminations, truncations, infos = self.envs.step(action.cpu().numpy())
                next_done = np.logical_or(terminations, truncations)
                rewards[step] = torch.tensor(reward).to(self.device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(next_done).to(self.device)

                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                            self.writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                            self.writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

            # bootstrap value if not done
            with torch.no_grad():
                next_value = self.agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(self.device)
                lastgaelam = 0
                for t in reversed(range(self.config.num_steps)):
                    if t == self.config.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + self.config.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + self.config.gamma * self.config.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            # flatten the batch
            b_obs = obs.reshape((-1,) + self.envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + self.envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(self.config.batch_size)
            clipfracs = []
            for epoch in range(self.config.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.config.batch_size, self.config.minibatch_size):
                    end = start + self.config.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > self.config.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if self.config.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.config.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.config.clip_coef,
                            self.config.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.config.ent_coef * entropy_loss + v_loss * self.config.vf_coef

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()

                if self.config.target_kl is not None and approx_kl > self.config.target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            self.writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], global_step)
            self.writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            self.writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            self.writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            self.writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            self.writer.add_scalar("losses/explained_variance", explained_var, global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            self.writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        return self.agent
def main():
    def load_config(path):
        with open(path, 'r') as file:
            return yaml.safe_load(file)
    parse = argparse.ArgumentParser()
    # get ppo or dqn
    parse.add_argument('--name', type=str, required=True, default='ppo', help='agent name')
    args = parse.parse_args()
    config = load_config(f"config/{args.name}.yaml")
    # makes a dataclass out of a dict
    cfg = make_dataclass(
        "exp_args", ((k, type(v)) for k, v in config.items())
    )(**config)

    # now we can do things the cleanrl way
    if cfg.exp_name == 'dqn':
        trainer = DQNTrainer(cfg)
        trainer.train()
        trainer.save_model()
    elif cfg.exp_name == 'ppo':
        trainer = PPOTrainer(cfg)
        trainer.train()
    else:
        raise ValueError("Unknown algorithm specified in config")


if __name__ == "__main__":
    main()