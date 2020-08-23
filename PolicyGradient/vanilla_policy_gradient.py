import numpy as np
import torch
from model import DiscretePolicy, DenseValueNetwork

from collections import deque
import gym
from gym.wrappers import Monitor
import wandb

class Reinforce:
    def __init__(self, env, entropy_weight, episodes=1000, max_t=1000):
        self.env = env
        self.env_name = self.env.unwrapped.spec.id
        self.state_space = self.env.observation_space.shape[0]
        self.actio_space = self.env.action_space.n
        
        self.episodes = episodes
        self.max_t = max_t
        self.gamma = 1.0
        self.entropy_weight = entropy_weight

        self.policy = DiscretePolicy(self.state_space, self.actio_space)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=0.0005)
        self.policy_max_grad_norm = 1

        self.value_network = DenseValueNetwork(self.state_space, hidden_dim=(256,128))
        self.value_optimizer = torch.optim.RMSprop(self.value_network.parameters(), lr=0.0007)
        self.value_max_grad_norm = float('inf')

        self.log_probs = []
        self.rewards = []
        self.entropies =[]
        self.values = []
    
    def optimize(self):
        """Function to calculate loss and backpropogate
        """
        no_traj = len(self.rewards)
        discounts = np.logspace(0, no_traj, num=no_traj, endpoint=False, base=self.gamma)
        returns = np.array([np.sum(discounts[:no_traj-t] * self.rewards[t:]) for t in range(no_traj)])

        discounts = torch.from_numpy(discounts).float().unsqueeze(1)
        returns = torch.from_numpy(returns).float().unsqueeze(1)
        log_probs_tensor = torch.cat(self.log_probs)
        entropies_tensor = torch.cat(self.entropies)
        values_tensor = torch.cat(self.values)

        value_error = returns - values_tensor
        policy_loss = -(discounts * value_error.detach() * log_probs_tensor).mean()
        entropy_loss = -entropies_tensor.mean()

        loss = policy_loss + self.entropy_weight * entropy_loss
        self.policy_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.policy.parameters(), self.policy_max_grad_norm)
        self.policy_optimizer.step()

        value_loss = value_error.pow(2).mul(0.5).mean()
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm(self.value_network.parameters(), self.value_max_grad_norm)
        self.value_optimizer.step()

    def run(self, project_name=None, save_every=100):
        """Function to train the agent for given environment

        Args:
            project_name (string, optional): name for project for wandb log. Defaults to None.
            save_every (int, optional): save checkpoint every n episode. Defaults to 100.
        """
        project_name = f"{self.env_name}-VPG" if project_name is None else project_name
        scores_avg = deque(maxlen=100)
        wandb.init(project="PolicyGradient-pytorch", name=f"{project_name}")
        try:
            for i_episode in range(1, self.episodes+1):
                state = self.env.reset()
                self.log_probs.clear()
                self.rewards.clear()
                self.entropies.clear()
                self.values.clear()
                score = 0
                for t in range(self.max_t):
                    action, log_prob, entropy = self.policy.act(state)
                    next_state, reward, done, _ = self.env.step(action)
                    
                    self.log_probs.append(log_prob)
                    self.rewards.append(reward)
                    self.entropies.append(entropy)
                    self.values.append(self.value_network(state))
                    
                    state = next_state
                    score += reward
                    if done:
                        break
                self.optimize()
                scores_avg.append(score)
                wandb.log({'Scores': score, 'Scores_avg': np.mean(scores_avg)})
                print("\r Episode {}/{} Average Score:{}".format(i_episode, self.episodes, np.mean(scores_avg)), end="")
                if i_episode % 100 == 0:
                    print("\r Episode {}/{} Average Score:{}".format(i_episode, self.episodes, np.mean(scores_avg)))
                if i_episode % save_every == 0:
                    torch.save(self.policy.state_dict(), f"./PolicyGradient/checkpoint/{self.env_name}-pytorch-{i_episode}.pth")
            torch.save(self.policy.state_dict(), f"./PolicyGradient/checkpoint/{self.env_name}-pytorch-{i_episode}.pth")
        except KeyboardInterrupt:
            torch.save(self.policy.state_dict(), f"./PolicyGradient/checkpoint/{self.env_name}-pytorch-{i_episode}.pth")
        

    def test(self, file, trial=5, greedy=False, record=False):
        """Function to run agent with given traied weights

        Args:
            file (string): file path for trained checkpoint
            trial (int, optional): number of trials to run. Defaults to 5.
            greedy (bool, optional): if true, choose action greedily else samples from distribution. 
                                      Defaults to False.
            record (bool, optional): if true, record a video of running agent. Defaults to False.
        """
        self.policy.load_state_dict(torch.load(file))
        if record:
            wandb.init(project="PolicyGradient-pytorch", name=f"Trial-Video-VPG", monitor_gym=True)
            self.env = Monitor(self.env, f'./video/{self.env_name}/', resume=True, force=True)
        
        for i in range(trial):
            score = 0
            state = self.env.reset()

            for t in range(self.max_t):
                self.env.render()
                if greedy:
                    action = self.policy.act_greedily(state)
                action, _, _ = self.policy.act(state)

                next_state, reward, done, _ = self.env.step(action)
                state = next_state
                score += reward

                if done:
                    break
            print(f"Trial: {i+1}  --->  Score: {score}")


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    agent = Reinforce(env, entropy_weight=0.001)
    # agent.run()
    agent.test('checkpoint/CartPole-v1-pytorch-VPG.pth', greedy=False, record=False)
