"""
This algorithm does not work properly. It has some error in run loop.
"""
import os
import torch
import torch.nn as nn

import wandb
import gym
import numpy as np
from gym.wrappers import Monitor
from collections import deque

from model import DenseActorCritic
from multienv import MultiEnv

os.environ['OMP_NUM_THREADS'] = '1'

class A2C:
    def __init__(self, env, episodes=1000, t_max=1000, no_envs=2, seed=0):
        assert no_envs > 1
        self.env = env
        self.env_name = self.env.unwrapped.spec.id
        self.state_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n
        self.no_envs = no_envs
        self.episodes = episodes
        self.t_max = t_max

        self.envs = MultiEnv(self.env, self.no_envs, seed)

        self.gamma = 0.99
        self.lamda = 0.95
        self.policy_loss_weight = 1.0
        self.value_loss_weight = 0.6
        self.entropy_loss_weight = 0.001

        self.ac_network = DenseActorCritic(self.state_space, self.action_space, hidden_dim=(128, 64))
        self.ac_optimizer = torch.optim.RMSprop(self.ac_network.parameters(), lr=0.001)
        self.ac_max_grad_norm = 1

        self.log_probs = []
        self.entropies = []
        self.values = []
        self.rewards = []

    def optimize(self):
        no_traj = len(self.rewards)
        rewards = np.array(self.rewards).squeeze()
        discounts = np.logspace(0, no_traj, num=no_traj, endpoint=False, base=self.gamma)
        returns = np.array([[np.sum(discounts[:no_traj-t] * rewards[t:, w]) for t in range(no_traj)] for w in range(self.no_envs)])
        
        log_probs_tensor = torch.stack(self.log_probs).squeeze()
        values_tensor = torch.stack(self.values).squeeze()
        entropies_tensor = torch.stack(self.entropies).squeeze()
        
        values_array = values_tensor.data.numpy()
        gae_discounts = np.logspace(0, no_traj-1, num=no_traj-1, base=self.gamma*self.lamda, endpoint=False)
        deltas = rewards[:-1] + self.gamma * values_array[1:] - values_array[:-1]
        advantages = np.array([[np.sum(gae_discounts[:no_traj-1-t] * deltas[t:, w]) for t in range(no_traj-1)]
                                for w in range(self.no_envs)])
        discounted_advantages = discounts[:-1] * advantages
        values_tensor = values_tensor[:-1,...].view(-1).unsqueeze(1)
        log_probs_tensor = log_probs_tensor.view(-1).unsqueeze(1)
        entropies_tensor = entropies_tensor.view(-1).unsqueeze(1)
        returns = torch.FloatTensor(returns.T[:-1]).view(-1).unsqueeze(1)
        discounted_advantages = torch.from_numpy(discounted_advantages.T).reshape((-1, 1))
	
        no_traj = (no_traj - 1) * self.no_envs
        assert returns.size() == (no_traj, 1)
        assert values_tensor.size() == (no_traj, 1)
        assert log_probs_tensor.size() == (no_traj, 1)
        assert entropies_tensor.size() == (no_traj, 1)

        value_error = returns.detach() - values_tensor
        value_loss = value_error.pow(2).mul(0.5).mean()
        policy_loss = -(discounted_advantages.detach() * log_probs_tensor).mean()
        entropy_loss = -entropies_tensor.mean()

        loss = self.policy_loss_weight * policy_loss + \
               self.value_loss_weight * value_loss + \
               self.entropy_loss_weight * entropy_loss
        
        self.ac_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.ac_network.parameters(), self.ac_max_grad_norm)
        self.ac_optimizer.step()

    def run(self, save_every=100):
        project_name = f"{self.env_name}-A2C"
        # wandb.init(project="ActorCritic-pytorch", name=project_name)
        episode = 0
        scores_avg = deque(maxlen=100)
        states = self.envs.reset()
        try:
            while episode <= self.episodes:
                score = np.zeros(self.no_envs)
                for _ in range(self.t_max):
                    actions, log_probs, entropies, values = self.ac_network.act(states)
                    new_states, rewards, dones, _ = self.envs.step(actions)

                    self.log_probs.append(log_probs)
                    self.entropies.append(entropies)
                    self.values.append(values)
                    self.rewards.append(rewards)

                    for i in range(self.no_envs):
                        score[i] += rewards[i].sum()

                    if dones.sum():
                        past_limits = self.envs._past_limit()
                        is_failures = np.logical_and(dones, np.logical_not(past_limits))
                        next_values = self.ac_network.evaluate_state(states).detach().numpy() * (1 - is_failures)
                        self.rewards.append(next_values) 
                        self.values.append(torch.Tensor(next_values))
                        self.optimize()
                        for i_rank in range(self.no_envs):
                            if dones[i_rank]:
                                del self.log_probs[i_rank]
                                del self.rewards[i_rank]
                                del self.values[i_rank]                                
                                del self.entropies[i_rank]
                                
                                states[i_rank] = self.envs.reset(rank=i_rank)
                                episode += 1
                                scores_avg.append(score[i])
                                print("\r Episode {}/{} Average Score:{}".format(episode, self.episodes, np.mean(scores_avg)), end="")
                                # wandb.log({'Scores': score, 'Scores_avg': np.mean(scores_avg)})
                                if episode % 100 == 0:
                                    print("\r Episode {}/{} Average Score:{}".format(episode, self.episodes, np.mean(scores_avg)))
            torch.save(self.ac_network.state_dict(), f"./checkpoint/A2C/{self.env_name}-A2C-pytorch-{episode}.pth")
        except KeyboardInterrupt:
            torch.save(self.ac_network.state_dict(), f"./checkpoint/A2C/{self.env_name}-A2C-pytorch-{episode}.pth")

    def test(self, file, trial=5, greedy=False, record=False):
        """Function to run agent with given traied weights

        Args:
            file (string): file path for trained checkpoint
            trial (int, optional): number of trials to run. Defaults to 5.
            greedy (bool, optional): if true, choose action greedily else samples from distribution. 
                                      Defaults to False.
            record (bool, optional): if true, record a video of running agent. Defaults to False.
        """
        
        self.ac_network.load_state_dict(torch.load(file))
        if record:
            wandb.init(project="ActorCritic-pytorch", name=f"Trial-Video-A2C", group="Trial Videos", monitor_gym=True)
            self.env = Monitor(self.env, f'./video/{self.env_name}/', resume=True, force=True)

        for i in range(trial):
            score = 0
            state = self.env.reset()

            for _ in range(self.t_max):
                self.env.render()
                if greedy:
                    action = self.ac_network.act_greedily(state)
                else:
                    action, _, _, _ = self.ac_network.act(state)

                next_state, reward, done, _ = self.env.step(action)
                state = next_state
                score += reward

                if done:
                    break
            print(f"Trial: {i+1}  --->  Score: {score}")

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    agent = A2C(env, episodes=5000, no_envs=10)
    agent.run()
    # agent.test("checkpoint/A2C/CartPole-A2C-pytorch2.pth", greedy=True)
