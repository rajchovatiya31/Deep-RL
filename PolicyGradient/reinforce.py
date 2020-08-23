import numpy as np
import torch
from model import DiscretePolicy

from collections import deque
import gym
from gym.wrappers import Monitor
import wandb

class Reinforce:
    def __init__(self, env, episodes=1000, max_t=1000):
        self.env = env
        self.env_name = self.env.unwrapped.spec.id
        self.state_space = self.env.observation_space.shape[0]
        self.actio_space = self.env.action_space.n
        
        self.episodes = episodes
        self.max_t = max_t
        self.gamma = 1.0

        self.policy = DiscretePolicy(self.state_space, self.actio_space)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=0.0005)

        self.log_probs = []
        self.rewards = []
    
    
    def optimize(self):
        """Function to calculate loss and backpropogate
        """
        no_traj = len(self.rewards)
        discounts = np.logspace(0, no_traj, num=no_traj, endpoint=False, base=self.gamma)
        returns = np.array([np.sum(discounts[:no_traj-t] * self.rewards[t:]) for t in range(no_traj)])

        discounts = torch.from_numpy(discounts).float().unsqueeze(1)
        returns = torch.from_numpy(returns).float().unsqueeze(1)
        log_probs_tensor = torch.cat(self.log_probs)

        loss = -(discounts * returns * log_probs_tensor).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    

    def optimize_var(self):
        no_traj = len(self.rewards)
        discounts = [self.gamma**i for i in range(no_traj+1)]
        returns = np.sum([a*b for a,b in zip(discounts, self.rewards)])

        loss = []
        for log_prob in self.log_probs:
            loss.append(-log_prob * returns)
        loss = torch.cat(loss).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    

    def run(self, project_name=None, save_every=100):
        """Function to train the agent for given environment

        Args:
            project_name (string, optional): name for project for wandb log. Defaults to None.
            save_every (int, optional): save checkpoint every n episode. Defaults to 100.
        """
        project_name = f"{self.env_name}-Reinforce" if project_name is None else project_name
        scores_avg = deque(maxlen=100)
        wandb.init(project="PolicyGradient-pytorch", name=f"{project_name}")
        try:
            for i_episode in range(1, self.episodes+1):
                state = self.env.reset()
                self.log_probs.clear()
                self.rewards.clear()
                score = 0
                for t in range(self.max_t):
                    action, log_prob, _ = self.policy.act(state)
                    next_state, reward, done, _ = self.env.step(action)
                    self.log_probs.append(log_prob)
                    self.rewards.append(reward)
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
                    torch.save(agent.policy.state_dict(), f"./reinforce/checkpoint/{self.env_name}-pytorch-{i_episode}.pth")
            torch.save(agent.policy.state_dict(), f"./reinforce/checkpoint/{self.env_name}-pytorch-{i_episode}.pth")
        except KeyboardInterrupt:
            torch.save(agent.policy.state_dict(), f"./reinforce/checkpoint/{self.env_name}-pytorch-{i_episode}.pth")
        

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
            wandb.init(project="PolicyGradient-pytorch", name=f"Trial-Video-REINFORCE", monitor_gym=True)
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
    agent = Reinforce(env)
    # agent.run()
    agent.test('D:/Deep-RL/PolicyGradient/checkpoint/CartPole-v1-pytorch-reinforce.pth', greedy=False, record=False)
