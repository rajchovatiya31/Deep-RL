import torch
import numpy as np

import pybullet_envs
import wandb
import gym
from gym.wrappers import Monitor

from collections import deque
from model import TwinDDPG, DDPGPolicyNet
from replay_buffer import ReplayBufferEff
from strategies import GreedyStrategy, NormalDecayStrategy


class TD3:
    def __init__(self, env, episodes=2000, t_max=1000, buffer_len=100000, batch_size=256):
        """TD3 agent

        Args:
            env (gym.env): gym environment
            episodes (int, optional): number of episodes to train. Defaults to 2000.
            t_max (int, optional): maxmimum time step to run agent in each episode. Defaults to 2000.
            buffer_len (int, optional): maximum number of elements to store in replay buffer . Defaults to 100000.
            batch_size (int, optional): batch size for training. Defaults to 256.
        """
        self.env = env
        self.env_name = env.unwrapped.spec.id
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.shape[0]
        self.action_bounds = env.action_space.low, env.action_space.high

        # initiate actor / policy networks
        self.target_policy_network = DDPGPolicyNet(
            self.state_space, self.action_bounds, hidden_dim=(256, 256))
        self.local_policy_network = DDPGPolicyNet(
            self.state_space, self.action_bounds, hidden_dim=(256, 256))
        self.policy_optimizer = torch.optim.Adam(
            self.local_policy_network.parameters(), lr=0.0003)
        self.policy_max_grad_norm = float('inf')

        # initiate critic / value networks
        self.target_value_network = TwinDDPG(
            self.state_space, self.action_space, hidden_dim=(256, 256))
        self.local_value_network = TwinDDPG(
            self.state_space, self.action_space, hidden_dim=(256, 256))
        self.value_optimizer = torch.optim.Adam(
            self.local_value_network.parameters(), lr=0.0003)
        self.value_max_grad_norm = float('inf')

        self.training_strategy = NormalDecayStrategy(self.action_bounds)
        self.testing_strategy = GreedyStrategy(self.action_bounds)

        self.memory = ReplayBufferEff(buffer_len, batch_size)

        self.episodes = episodes
        self.t_max = t_max

        self.policy_noise_clip_ratio = 0.5
        self.policy_noise_ratio = 0.1
        self.gamma = 0.99
        self.t_policy_optimize = 0
        self.t_step = 0
        self.train_policy_every_steps = 2
        self.update_every = 2

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        if len(self.memory) > self.memory.batch_size:
            experiences = self.memory.sample()
            self.optimize(experiences)

        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            self.soft_update(self.local_policy_network,
                             self.target_policy_network, 0.005)
            self.soft_update(self.local_value_network,
                             self.target_value_network, 0.005)

    def optimize(self, experiences):
        """Function to calculate losses and do back propogation

        Args:
            experiences (tuple): tuple of (s, a, r, s', d)
        """
        # convert to tensor
        states = torch.from_numpy(experiences[0]).float()
        actions = torch.from_numpy(experiences[1]).float()
        rewards = torch.from_numpy(experiences[2]).float()
        next_states = torch.from_numpy(experiences[3]).float()
        dones = torch.from_numpy(experiences[4]).float()

        # Choose action, add noise and scale it. Find target value
        with torch.no_grad():
            a_ran = self.target_policy_network.action_max - self.target_policy_network.action_min
            a_noise = torch.randn_like(actions) * self.policy_noise_ratio * a_ran  # scaled noise 
            n_min = self.target_policy_network.action_min * self.policy_noise_clip_ratio
            n_max = self.target_policy_network.action_max * self.policy_noise_clip_ratio
            a_noise = torch.max(torch.min(a_noise, n_max), n_min)  # action noise

            argmax_a_q_sp = self.target_policy_network(next_states)
            noisy_argmax_a_q_sp = argmax_a_q_sp + a_noise  # add action noise
            noisy_argmax_a_q_sp = torch.max(torch.min(noisy_argmax_a_q_sp,
                                                      self.target_policy_network.action_max),
                                            self.target_policy_network.action_min)  # clamp argma action

            max_a_q_sp_a, max_a_q_sp_b = self.target_value_network(
                next_states, noisy_argmax_a_q_sp)  
            max_a_q_sp = torch.min(max_a_q_sp_a, max_a_q_sp_b) 

            target_q_sa = rewards + self.gamma * max_a_q_sp * (1 - dones)   # target value

        q_sa_a, q_sa_b = self.local_value_network(states, actions) # expected value
        td_error_a = q_sa_a - target_q_sa
        td_error_b = q_sa_b - target_q_sa

        # critic / value loss and update weights
        value_loss = td_error_a.pow(2).mul(0.5).mean() + td_error_b.pow(2).mul(0.5).mean()
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.local_value_network.parameters(),
                                       self.value_max_grad_norm)
        self.value_optimizer.step()

        # actor / policy loss and update weights
        self.t_policy_optimize = (self.t_policy_optimize + 1) % self.train_policy_every_steps
        if self.t_policy_optimize == 0:
            argmax_a_q_s = self.local_policy_network(states)
            max_a_q_s = self.local_value_network.forward_Q(states, argmax_a_q_s)

            policy_loss = -max_a_q_s.mean()
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.local_policy_network.parameters(),
                                           self.policy_max_grad_norm)
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
        wandb.init(project="ActorCritic-Pytorch", name=f"{project_name}", group="TD3")
        scores_window = deque(maxlen=100)
        try:
            for i_episode in range(1, self.episodes + 1):
                score, done = 0, False
                state = self.env.reset()

                while not done:
                    action = self.training_strategy.select_action(
                        self.local_policy_network, state, len(self.memory) < self.memory.batch_size)
                    next_state, reward, done, _ = self.env.step(action)
                    self.step(state, action, reward, next_state, done)

                    state = next_state
                    score += reward
                    if done:
                        break
                scores_window.append(score)
                wandb.log({'Scores': score, 'Scores_avg': np.mean(scores_window)})
                print("\r Episode {}/{} Average Score:{}".format(i_episode,
                                                                self.episodes, np.mean(scores_window)), end="")
                if i_episode % 100 == 0:
                    print("\r Episode {}/{} Average Score:{}".format(i_episode,
                                                                self.episodes, np.mean(scores_window)))
                if i_episode % save_every == 0:
                    torch.save(self.local_policy_network.state_dict(),
                                f"./checkpoint/TD3/{self.env_name}-{i_episode}.pth")
            torch.save(self.local_policy_network.state_dict(),
                        f"./checkpoint/TD3/{self.env_name}-{i_episode}.pth")
        except KeyboardInterrupt:
            torch.save(self.local_policy_network.state_dict(),
                       f"./checkpoint/TD3/{self.env_name}-{i_episode}.pth")

    def test(self, file, record=False):
        """
        Descreption: Function to test pertrained model
        parameter:
            file(string): path of the pretrained weights file
            record(bool): true if want to record video of environment run
        """
        self.local_policy_network.load_state_dict(torch.load(file))
        if record:
            wandb.init(project="ActorCritic-pytorch", name=f"Trial-Video-TD3",
                       group="Trial Videos", monitor_gym=True)
            self.env = Monitor(
                self.env, f'./video/{self.env_name}/', resume=True, force=True)
        for i_episode in range(5):
            score = 0
            self.env.render()
            state = self.env.reset()

            for t in range(self.t_max):
                # self.env.render()
                action = self.testing_strategy.select_action(
                    self.local_policy_network, state)
                state, reward, done, _ = self.env.step(action)
                score += reward

                if done:
                    break
            print(f"Run:{i_episode+1} -->  Score:{score}")

if __name__ == "__main__":
    env = gym.make('HopperBulletEnv-v0')
    agent = TD3(env)
    # agent.run()  # uncomment to train the agent
    agent.test("checkpoint/TD3/HopperBulletEnv-v0.pth", record=False)