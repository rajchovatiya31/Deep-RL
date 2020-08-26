import torch
import numpy as np
import gym
import wandb

from gym.wrappers import Monitor
from model import DDPGPolicyNet, DDPGValueNet
from replay_buffer import ReplayBufferEff
from collections import deque


class GreedyStrategy:
    def __init__(self, bounds):
        """Greedy strategy

        Args:
            bounds (tuple): bound of action
        """
        self.low, self.high = bounds

    def select_action(self, model, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            greedy_action = model(state).cpu().detach().data.numpy().squeeze()
        action = np.clip(greedy_action, self.low, self.high)
        return np.reshape(action, self.high.shape)


class NormalNoiseStrategy:
    def __init__(self, bounds, exploration_noise_rate=0.1):
        self.low, self.high = bounds
        self.noise_rate = exploration_noise_rate

    def select_action(self, model, state, max_exploration):
        state = torch.from_numpy(state).float().unsqueeze(0)

        if max_exploration:
            noise_scale = self.high
        else:
            noise_scale = self.noise_rate * self.high

        with torch.no_grad():
            greedy_action = model(state).cpu().detach().data.numpy().squeeze()

        noise = np.random.normal(loc=0, scale=noise_scale, size=len(self.high))
        action_noisy = greedy_action + noise
        action = np.clip(action_noisy, self.low, self.high)

        return action


class DDPG:
    def __init__(self, env, episodes=2000, t_max=2000, buffer_len=100000, batch_size=256):
        """DDPG agent

        Args:
            env (gym.env): gym environment
            episodes (int, optional): number of episodes to train. Defaults to 2000.
            t_max (int, optional): maxmimum time step to run agent in each episode. Defaults to 2000.
            buffer_len (int, optional): maximum number of elements to store in replay buffer . Defaults to 100000.
            batch_size (int, optional): batch size for training. Defaults to 256.
        """
        self.env = env
        self.env_name = self.env.unwrapped.spec.id
        self.state_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.shape[0]
        self.action_bounds = self.env.action_space.low, self.env.action_space.high

        self.local_policy_network = DDPGPolicyNet(
            self.state_space, self.action_bounds, hidden_dim=(256, 256))
        self.target_policy_network = DDPGPolicyNet(
            self.state_space, self.action_bounds, hidden_dim=(256, 256))
        self.policy_optimizer = torch.optim.Adam(
            self.local_policy_network.parameters(), lr=0.0003)
        self.policy_max_grad_norm = float('inf')

        self.local_value_network = DDPGValueNet(
            self.state_space, self.action_space, hidden_dim=(256, 256))
        self.target_value_network = DDPGValueNet(
            self.state_space, self.action_space, hidden_dim=(256, 256))
        self.value_optimizer = torch.optim.Adam(
            self.local_value_network.parameters(), lr=0.0003)
        self.value_max_grad_norm = float('inf')

        self.soft_update(self.local_policy_network,
                         self.target_policy_network, tau=1.0)
        self.soft_update(self.local_value_network,
                         self.target_value_network, tau=1.0)

        self.training_strategy = NormalNoiseStrategy(self.action_bounds)
        self.testing_strategy = GreedyStrategy(self.action_bounds)

        self.buffer_len = buffer_len
        self.batch_size = batch_size
        self.memory = ReplayBufferEff(self.buffer_len, self.batch_size)

        self.episodes = episodes
        self.t_max = t_max
        self.gamma = 0.99

        self.t_step = 0
        self.update_every = 1

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        if len(self.memory) > self.batch_size:
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

        # calculate critic loss
        argmax_action_ns = self.target_policy_network(next_states)
        max_action_q_ns = self.target_value_network(
            next_states, argmax_action_ns)
        q_target = rewards + (self.gamma * max_action_q_ns * (1-dones))
        q_expected = self.local_value_network(states, actions)
        value_loss = torch.nn.functional.mse_loss(q_expected, q_target)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.local_value_network.parameters(), self.value_max_grad_norm)
        self.value_optimizer.step()

        # calculate policy loss
        argmax_action_s = self.local_policy_network(states)
        max_action_q_s = self.local_value_network(
            states, argmax_action_s)
        policy_loss = -max_action_q_s.mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.local_policy_network.parameters(), self.policy_max_grad_norm)
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
        project_name = f"{self.env_name}-DDPG" if project_name is None else project_name
        wandb.init(project="ActorCritic-Pytorch",
                   name=f"{project_name}", group="DDPG")
        scores_window = deque(maxlen=100)
        try:
            for i_episode in range(1, self.episodes+1):
                state = self.env.reset()
                done = False
                score = 0
                for _ in range(self.t_max):
                    action = self.training_strategy.select_action(
                        self.local_policy_network, state, len(self.memory) < self.batch_size)
                    next_state, reward, done, _ = self.env.step(action)
                    self.step(state, action, reward, next_state, done)

                    state = next_state
                    score += reward
                    if done:
                        break
                scores_window.append(score)
                wandb.log(
                    {'Scores': score, 'Scores_avg': np.mean(scores_window)})
                print("\r Episode {}/{} Average Score:{}".format(i_episode,
                                                                 self.episodes, np.mean(scores_window)), end="")
                if i_episode % 100 == 0:
                    print("\r Episode {}/{} Average Score:{}".format(i_episode,
                                                                     self.episodes, np.mean(scores_window)))
                if i_episode % save_every == 0:
                    torch.save(self.local_policy_network.state_dict(),
                               f"./checkpoint/DDPG/{self.env_name}-{i_episode}.pth")
            torch.save(self.local_policy_network.state_dict(),
                       f"./checkpoint/DDPG/{self.env_name}-{i_episode}.pth")
        except KeyboardInterrupt:
            torch.save(self.local_policy_network.state_dict(),
                       f"./checkpoint/DDPG/{self.env_name}-{i_episode}.pth")

    def test(self, file, record=False):
        """
        Descreption: Function to test pertrained model
        parameter:
            file(string): path of the pretrained weights file
            record(bool): true if want to record video of environment run
        """
        self.local_policy_network.load_state_dict(torch.load(file))
        if record:
            wandb.init(project="ActorCritic-pytorch", name=f"Trial-Video-DDPG",
                       group="Trial Videos", monitor_gym=True)
            self.env = Monitor(
                self.env, f'./video/{self.env_name}/', resume=True, force=True)
        for i_episode in range(5):
            score = 0
            state = self.env.reset()

            for t in range(self.t_max):
                self.env.render()
                action = self.testing_strategy.select_action(
                    self.local_policy_network, state)
                state, reward, done, _ = self.env.step(action)
                score += reward

                if done:
                    break
            print(f"Run:{i_episode+1} -->  Score:{score}")


if __name__ == "__main__":
    env = gym.make('Pendulum-v0')
    agent = DDPG(env)
    # agent.run()  # uncomment to train agent
    agent.test("checkpoint/DDPG/Pendulum-v0-DDPG.pth", record=False)
