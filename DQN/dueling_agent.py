import torch
import gym
import wandb
import random
import numpy as np

from replay_buffer import ReplayBufferEff
from model import DuellingDQN

from collections import deque
import matplotlib.pyplot as plt
from gym.wrappers import Monitor

class DQNagent:
    """
    description: DQN agent
    parameters:
        env(gym env): openai gym environment
        episodes(int): number of episodes to train
        t_max(int): maximum time to run environment in each episode
        buffer_size(int): number of elements to store in replay buffer
        device(torch.device): device on which to train an agent as torch.device
    """
    def __init__(self, env, max_gradient_norm=None, episodes=1000, t_max=1000, buffer_size=50000, batch_size=128, device='cpu'):
        self.env = env
        self.env_name = self.env.unwrapped.spec.id
        self.state_space = self.env.observation_space.shape[0]                                     # shape of state
        self.action_space = self.env.action_space.n                                             # number of actions
        self.buffer_size = buffer_size                                                          
        self.batch_size = batch_size                                                            # batch size

        self.episodes = episodes
        self.t_max = t_max

        self.max_gradient_norm = float('inf') if max_gradient_norm is None else max_gradient_norm

        self.memory = ReplayBufferEff(self.buffer_size, self.batch_size)

        self.device = device

        self.network_local = DuellingDQN(self.state_space, self.action_space).to(self.device)     # local network
        self.network_target = DuellingDQN(self.state_space, self.action_space).to(self.device)     # target network

        self.optimzer = torch.optim.RMSprop(self.network_local.parameters(), lr=0.0007)

        self.epsilon = 1.0
        self.epsilon_decay = 0.998
        self.epsilon_min = 0.001

        self.t_step = 0
        self.update_every = 15

        self.gamma = 1.0
 
    def act(self, state, test=False):
        """
        Descreption: Return action for given state according to current policy
        Parameters:
            state(numpy array): current state
            test(bool): turn test mode on or off 
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        self.network_local.eval()
        with torch.no_grad():
            action_prob = self.network_local(state)
        self.network_local.train()
        if test:
            return torch.argmax(action_prob).cpu().item()
        else:
            if random.random() < self.epsilon:
                return random.choice(range(self.action_space))
            else:
                return torch.argmax(action_prob).cpu().item()
    
    def step(self, state, action, reward, next_state, done):
        """function to step optimizer

        Args:
            state (ndarray): array of state
            action (ndarray): array of action
            reward (ndarray): array of rewards
            next_state (ndarray): array of next_states
            done (ndarray): array of dones
        """
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % self.update_every

        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)
    
    def learn(self, experiences, soft_update=True):
        """
        Descreption: update network parameters using given experience
        Parameters:
            experiences(named tuple): namedtuple of (s, a, r, s', done)
            soft_update(bool): bool to choose soft update vs hard update
        """
        states = torch.from_numpy(experiences[0]).float()
        actions = torch.from_numpy(experiences[1]).long()
        rewards = torch.from_numpy(experiences[2]).float()
        next_states = torch.from_numpy(experiences[3]).float()
        dones = torch.from_numpy(experiences[4]).float()

        q_target_next_argmax = self.network_local(next_states).max(axis=1)[1]
        q_target_next = self.network_target(next_states).detach()
        q_max_target_next = q_target_next[np.arange(self.batch_size), q_target_next_argmax].unsqueeze_(1)
        q_target = rewards + (self.gamma * q_max_target_next * (1-dones))

        q_expacted = self.network_local(states).gather(1, actions)
        
        loss = torch.nn.functional.mse_loss(q_expacted, q_target)
        self.optimzer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network_local.parameters(), self.max_gradient_norm)
        self.optimzer.step()

        if soft_update:
            self.soft_update(self.network_local, self.network_target, 1e-3)
        self.hard_update(self.network_local, self.network_target)

    def soft_update(self, local_model, target_model, tau):
        """function to update parameter from local to target model

        Args:
            local_model (torch network): local network
            target_model (torch network): target network
            tau (float): parameter for soft update to fade weights
        """
        for local_parameter, target_parameter in zip(local_model.parameters(), target_model.parameters()):
            target_parameter.data.copy_(tau * local_parameter.data + (1.0 - tau) * target_parameter.data)
    
    def hard_update(self, local_model, target_model):
        """function to copy local model parameter to target model

        Args:
            local_model (torch network): local network
            target_model (torch network): target network
        """
        for local_parameter, target_parameter in zip(local_model.parameters(), target_model.parameters()):
            target_parameter.data.copy_(local_parameter)

    def run(self, save_every=100, render=False, project_name=None):
        """
        Descreption: function to train the agent
        Parameters:
            save_every(int): save agent weights after every n episodes
            render(bool): render environment while training
            project_name(string): name of the run of the project
        """
        project_name = self.env_name if project_name is None else project_name
        wandb.init(project="DQN-Pytorch", name=f"{project_name}-Dueling DQN")
        scores_window = deque(maxlen=100)
        try:
            for i_episode in range(1, self.episodes+1):
                state = self.env.reset()
                score = 0
                for t in range(self.t_max):
                    if render:
                        self.env.render()
                    action = self.act(state)
                    next_state, reward, done, _ = self.env.step(action)
                    self.step(state, action, reward, next_state, done)

                    state = next_state
                    score += reward
                    if done:
                        break
                wandb.log({'Epsilon': self.epsilon, 'Scores': score})
                self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
                scores_window.append(score)
                print("\r Episode {}/{} Average Score:{}".format(i_episode, self.episodes, np.mean(scores_window)), end="")
                if i_episode % 100 == 0:
                    print("\r Episode {}/{} Average Score:{}".format(i_episode, self.episodes, np.mean(scores_window)))
                if i_episode % save_every == 0:
                    torch.save(agent.network_local.state_dict(), f"./my-dqn/pytorch/{self.env_name}-pytorch-{i_episode}.pth")
            torch.save(agent.network_local.state_dict(), f"./my-dqn/pytorch/{self.env_name}-pytorch-{i_episode}.pth")
        except KeyboardInterrupt:
            torch.save(agent.network_local.state_dict(), f"./my-dqn/pytorch/{self.env_name}-pytorch-{i_episode}.pth")
    
    def test(self, file, record=False):
        """
        Descreption: Function to test pertrained model
        parameter:
            file(string): path of the pretrained weights file
            record(bool): true if want to record video of environment run
        """
        self.network_local.load_state_dict(torch.load(file))
        if record:
            wandb.init(project="DQN-Pytorch", name="Test Video-Dueling DQN", monitor_gym=True)
            self.env = Monitor(self.env, f'./video/{self.env_name}/', resume=True, force=True)
        for i_episode in range(5):
            score = 0
            state = self.env.reset()

            for t in range(self.t_max):
                self.env.render()
                action = self.act(state, test=True)
                state, reward, done, _ = self.env.step(action)
                score += reward

                if done:
                    break
            print(f"Run:{i_episode+1} -->  Score:{score}")

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    agent = DQNagent(env, episodes=2000)
    # agent.run() # uncomment to train model
    agent.test('checkpoints/CartPole-v1/dueling-dqn/CartPole-v1-pytorch-duelingDDQN.pth', record=False)
