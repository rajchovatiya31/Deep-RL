import torch
import numpy as np

import pybullet_envs
import wandb
import gym
from gym.wrappers import Monitor
from collections import deque

from model import SACValueNet, SACGaussianPolicy
from strategies import GreedyStrategy, NormalNoiseStrategy
from replay_buffer import ReplayBufferEff


class SAC:
    def __init__(self, env, episodes=2000, t_max=2000, buffer_len=100000, batch_size=5):
        """SAC agent

        Args:
            env (gym.env): gym environment
            episodes (int, optional): number of episodes to train. Defaults to 2000.
            t_max (int, optional): maxmimum time step to run agent in each episode. Defaults to 2000.
            buffer_len (int, optional): maximum number of elements to store in replay buffer . Defaults to 100000.
            batch_size (int, optional): batch size for training. Defaults to 128.
        """
        self.env = env
        self.env_name = env.unwrapped.spec.id
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.shape[0]
        self.action_bounds = env.action_space.low, env.action_space.high

        self.policy_network = SACGaussianPolicy(self.state_space, self.action_bounds, hidden_dim=(256, 256))
        self.policy_optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=0.0003)
        self.policy_grad_max_norm = float('inf')

        self.target_value_network_a = SACValueNet(self.state_space, self.action_space, hidden_dim=(256, 256))
        self.local_value_network_a = SACValueNet(self.state_space, self.action_space, hidden_dim=(256, 256))
        self.target_value_network_b = SACValueNet(self.state_space, self.action_space, hidden_dim=(256, 256))
        self.local_value_network_b = SACValueNet(self.state_space, self.action_space, hidden_dim=(256, 256))

        self.soft_update(self.local_value_network_a, self.target_value_network_a, 1)
        self.soft_update(self.local_value_network_b, self.target_value_network_b, 1)

        self.value_optimizer_a = torch.optim.Adam(self.local_value_network_a.parameters(), lr=0.0003)
        self.value_optimizer_b = torch.optim.Adam(self.local_value_network_b.parameters(), lr=0.0003)
        self.value_grad_max_norm = float('inf')

        self.episodes = episodes
        self.batch_size = batch_size
        self.memory = ReplayBufferEff(buffer_len, batch_size)

        self.gamma = 0.99
        self.update_every = 2
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        if len(self.memory) > self.memory.batch_size:
            experiences = self.memory.sample()
            self.optimize(experiences)

        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            self.soft_update(self.local_value_network_a,
                             self.target_value_network_a, 0.005)
            self.soft_update(self.local_value_network_b,
                             self.target_value_network_b, 0.005)

    def optimize(self, experiences):
        """Function to calculate losses and do back propogation

        Args:
            experiences (tuple): tuple of (s, a, r, s', d)
        """
        torch.autograd.set_detect_anomaly(True)
        states = torch.from_numpy(experiences[0]).float()
        actions = torch.from_numpy(experiences[1]).float()
        rewards = torch.from_numpy(experiences[2]).float()
        next_states = torch.from_numpy(experiences[3]).float()
        dones = torch.from_numpy(experiences[4]).float()

        current_actions, logpi_s, _ = self.policy_network.full_pass(states)
        target_alpha = (logpi_s + self.policy_network.target_entropy).detach()
        alpha_loss = -(self.policy_network.log_alpha * target_alpha).mean()
 
        self.policy_network.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.policy_network.alpha_optimizer.step()
        alpha = self.policy_network.log_alpha.exp()

        current_q_sa_a = self.local_value_network_a(states, current_actions)
        current_q_sa_b = self.local_value_network_b(states, current_actions)
        current_q_sa = torch.min(current_q_sa_a, current_q_sa_b)
        
        # Q loss
        ap, logpi_sp, _ = self.policy_network.full_pass(next_states)
        q_spap_a = self.target_value_network_a(next_states, ap)
        q_spap_b = self.target_value_network_b(next_states, ap)
        q_spap = torch.min(q_spap_a, q_spap_b) - alpha * logpi_sp
        target_q_sa = (rewards + self.gamma * q_spap * (1 - dones)).detach()

        q_sa_a = self.local_value_network_a(states, actions)
        q_sa_b = self.local_value_network_b(states, actions)
        qa_loss = (q_sa_a - target_q_sa).pow(2).mul(0.5).mean()
        qb_loss = (q_sa_b - target_q_sa).pow(2).mul(0.5).mean()

        self.value_optimizer_a.zero_grad()
        qa_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.local_value_network_a.parameters(), 
                                       self.value_grad_max_norm)
        self.value_optimizer_a.step()

        self.value_optimizer_b.zero_grad()
        qb_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.local_value_network_b.parameters(),
                                       self.value_grad_max_norm)
        self.value_optimizer_b.step()

        self.policy_optimizer.zero_grad()
        policy_loss = (alpha * logpi_s - current_q_sa).mean()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 
                                       self.policy_grad_max_norm)        
        self.policy_optimizer.step()

    def soft_update(self, local_model, target_model, tau):
        """function to update parameter from local to target model

        Args:
            local_model (torch network): local network
            target_model (torch network): target network
            tau (float): parameter for soft update to fade weights
        """
        for local_parameter, target_parameter in zip(local_model.parameters(), target_model.parameters()):
            target_parameter.data.copy_(
                tau * local_parameter.data + (1.0 - tau) * target_parameter.data)
    
    def run(self, save_every=100, project_name=None):
        project_name = f"{self.env_name}-TD3" if project_name is None else project_name
        # wandb.init(project="ActorCritic-Pytorch", name=f"{project_name}", group="TD3")
        scores_window = deque(maxlen=100)
        try:
            for i_episode in range(1, self.episodes + 1):
                score, done = 0, False
                state = self.env.reset()

                while not done:
                    with torch.no_grad():
                        if len(self.memory) < self.batch_size:
                            action = self.policy_network.select_random_action(state)
                        else:
                            action = self.policy_network.select_action(state)
                        next_state, reward, done, _ = self.env.step(action)
                    self.step(state, action, reward, next_state, done)

                    state = next_state
                    score += reward
                    if done:
                        break
                scores_window.append(score)
                # wandb.log({'Scores': score, 'Scores_avg': np.mean(scores_window)})
                print("\r Episode {}/{} Average Score:{}".format(i_episode,
                                                                self.episodes, np.mean(scores_window)), end="")
                if i_episode % 100 == 0:
                    print("\r Episode {}/{} Average Score:{}".format(i_episode,
                                                                self.episodes, np.mean(scores_window)))
                if i_episode % save_every == 0:
                    torch.save(self.local_policy_network.state_dict(),
                                f"./checkpoint/SAC/{self.env_name}-{i_episode}.pth")
            torch.save(self.local_policy_network.state_dict(),
                        f"./checkpoint/SAC/{self.env_name}-{i_episode}.pth")
        except KeyboardInterrupt:
            torch.save(self.local_policy_network.state_dict(),
                       f"./checkpoint/SAC/{self.env_name}-{i_episode}.pth")

if __name__ == "__main__":
    env = gym.make('HalfCheetahBulletEnv-v0')
    agent = SAC(env)
    agent.run()