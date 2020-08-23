import sys
import torch
import torch.multiprocessing as mp

import numpy as np
import wandb

import gym
from gym.wrappers import Monitor
from collections import deque
from shared_optimizer import *
from model import *

class GAE:
    def __init__(self, env, entropy_weight, episodes=500, t_max=1000, no_worker=2, seed=0):
        """GAE Agent

        Args:
            env (gym env): OpenAI gym environment
            entropy_weight (float): weight of entopy error
            episodes (int, optional): Number of episodes to train. Defaults to 500.
            t_max (int, optional): Maximum time limit for each episode. Defaults to 1000.
            no_worker (int, optional): Number of workers to train agent parallel. Defaults to 1.
            seed (int, optional): seed for numpy and pytorch random. Defaults to 0.
        """
        self.env = env
        self.env_name = self.env.unwrapped.spec.id
        self.state_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n
        self.entropy_weight = entropy_weight

        self.episodes = episodes
        self.t_max = t_max
        self.seed = seed
        self.gamma = 0.99
        self.lamda = 0.95
        self.no_worker = no_worker

        self.local_policy_network = DiscretePolicy(self.state_space, self.action_space, hidden_dim=(128, 64))
        self.local_policy_optimizer = SharedAdam(self.local_policy_network.parameters(), lr=0.0005)
        self.policy_max_grad_norm = 1

        self.local_value_network = DenseValueNetwork(self.state_space, hidden_dim=(256,128))
        self.local_value_optimizer = SharedRMSprop(self.local_value_network.parameters(), lr=0.0007)
        self.value_max_grad_norm = float('inf')

        self.shared_policy_network = DiscretePolicy(self.state_space, self.action_space).share_memory()
        self.shared_policy_optimizer = SharedAdam(self.shared_policy_network.parameters(), lr=0.0005)

        self.shared_value_network = DenseValueNetwork(self.state_space, hidden_dim=(256,128))
        self.shared_value_optimizer = SharedRMSprop(self.shared_value_network.parameters(), lr=.0007)

        self.log_probs = []
        self.rewards = []
        self.values = []
        self. entropies = []

    def optimize(self, local_policy_network, local_value_network):
        """function to calculte losses and do backpropogation for actor and critic networks

        Args:
            local_policy_network (nn.Module): actor network
            local_value_network (nn.Module): critic network
        """
        # Calculating discounted return up to n+1 step
        no_traj = len(self.rewards)
        discounts = np.logspace(0, no_traj, num=no_traj, endpoint=False, base=self.gamma)
        returns = np.array([np.sum(discounts[:no_traj-t] * self.rewards[t:]) for t in range(no_traj)])

        log_probs_tensor = torch.cat(self.log_probs[:-1])
        values_tensor = torch.cat(self.values)
        entropies_tensor = torch.cat(self.entropies)

        # calculate GAE targets
        values_array = values_tensor.view(-1).data.numpy()
        gae_discounts = np.logspace(0, no_traj-1, num=no_traj-1, base=self.gamma*self.lamda, endpoint=False)
        deltas = self.rewards[:-1] + self.gamma * values_array[1:] - values_array[:-1]
        advantages = np.array([np.sum(gae_discounts[:no_traj-1-t] * deltas[t:]) for t in range(no_traj-1)])
        
        # convert numpy arrays to torch tensors
        values_tensor = values_tensor[:-1,...]
        discounts = torch.from_numpy(discounts[:-1]).float().unsqueeze_(1)
        returns = torch.from_numpy(returns[:-1]).float().unsqueeze_(1)
        advantages = torch.from_numpy(advantages).float().unsqueeze_(1)

        # calcuate actor/policy and entropy loss and do back propogation
        policy_loss = -(discounts * advantages.detach() * log_probs_tensor).mean()
        entropy_loss = entropies_tensor.mean()
        loss = policy_loss + entropy_loss * self.entropy_weight
        self.shared_policy_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(local_policy_network.parameters(), self.policy_max_grad_norm)  # clip gradient norm
        for param, shared_param in zip(local_policy_network.parameters(),  # Here copy local network gradients to shared network gradients
                                       self.shared_policy_network.parameters()):
            if shared_param.grad is None:
                shared_param.grad = param.grad
        self.shared_policy_optimizer.step()
        local_policy_network.load_state_dict(self.shared_policy_network.state_dict())  # load newly calculated parameters again to local network
        
        # calculate loss for critic/value network
        value_error = returns - values_tensor
        value_loss = value_error.pow(2).mul(0.5).mean()
        self.shared_value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm(local_value_network.parameters(), self.value_max_grad_norm)
        for param, shared_param in zip(local_value_network.parameters(),  # Here copy local network gradients to shared network gradients
                                       self.shared_value_network.parameters()):
            if shared_param.grad is None:
                shared_param.grad = param.grad
        self.shared_value_optimizer.step()
        local_value_network.load_state_dict(self.shared_value_network.state_dict())  # load newly calculated parameters again to local network
    
    def one_run(self, rank, save_every=100):
        """function to run one worker

        Args:
            rank (int): worker instatnce number
            save_every (int, optional): save chekpoint every n episode. Defaults to 100.
        """
        project_name = f"{self.env_name}-GAE"
        wandb.init(project="ActorCritic-pytorch", name=f"{project_name}-worker-{rank+1}", group="GAE-Workers")
        scores_avg = deque(maxlen=100)
        local_policy_network = self.local_policy_network
        local_policy_network.load_state_dict(self.shared_policy_network.state_dict())
        local_value_network = self.local_value_network
        local_value_network.load_state_dict(self.shared_value_network.state_dict())

        local_seed = self.seed + rank
        torch.manual_seed(local_seed); np.random.seed(local_seed)
        try:
            for i_episode in range(1, self.episodes+1):
                self.log_probs.clear()
                self.values.clear()
                self.entropies.clear()
                self.rewards.clear()
                score = 0

                state = self.env.reset()

                for t in range(self.t_max):
                    action, log_prob, entropy = local_policy_network.act(state)
                    next_state, reward, done, _ = self.env.step(action)

                    self.log_probs.append(log_prob)
                    self.entropies.append(entropy)
                    self.rewards.append(reward)
                    self.values.append(local_value_network(state))

                    state = next_state
                    score += reward

                    if done:
                        break
                self.optimize(local_policy_network, local_value_network)
                scores_avg.append(score)
                wandb.log({'Scores': score, 'Scores_avg': np.mean(scores_avg)})
                if rank==0:
                    print("\r Episode {}/{} Average Score:{}".format(i_episode, self.episodes, np.mean(scores_avg)), end="")
                    if i_episode % 100 == 0:
                        print("\r Episode {}/{} Average Score:{}".format(i_episode, self.episodes, np.mean(scores_avg)))
                if i_episode % save_every == 0:
                    torch.save(local_policy_network.state_dict(), f"./PolicyGradient/checkpoint/GAE/{self.env_name}-pytorch-{i_episode}-{rank+1}.pth")
            torch.save(local_policy_network.state_dict(), f"./PolicyGradient/checkpoint/GAE/{self.env_name}-pytorch-{i_episode}-{rank+1}.pth")
            torch.save(self.shared_policy_network.state_dict(), f"./PolicyGradient/checkpoint/GAE/shared-{i_episode}.pth")
        except KeyboardInterrupt:
            torch.save(local_policy_network.state_dict(), f"./PolicyGradient/checkpoint/GAE/{self.env_name}-pytorch-{i_episode}-{rank+1}.pth")
            torch.save(self.shared_policy_network.state_dict(), f"./PolicyGradient/checkpoint/GAE/shared-{i_episode}.pth")    


    def run(self):
        """Run parallel instaces of workers
        """
        workers = [mp.Process(target=self.one_run, args=(rank,)) for rank in range(self.no_worker)]
        [worker.start() for worker in workers]; [worker.join() for worker in workers]
        [worker.close() for worker in workers]
    
    def test(self, file, trial=5, greedy=False, record=False):
        """Function to run agent with given traied weights

        Args:
            file (string): file path for trained checkpoint
            trial (int, optional): number of trials to run. Defaults to 5.
            greedy (bool, optional): if true, choose action greedily else samples from distribution. 
                                      Defaults to False.
            record (bool, optional): if true, record a video of running agent. Defaults to False.
        """
        
        self.local_policy_network.load_state_dict(torch.load(file))
        if record:
            wandb.init(project="ActorCritic-pytorch", name=f"Trial-Video-GAE", group="Trial Videos", monitor_gym=True)
            self.env = Monitor(self.env, f'./video/{self.env_name}/', resume=True, force=True)

        for i in range(trial):
            score = 0
            state = self.env.reset()

            for t in range(self.t_max):
                self.env.render()
                if greedy:
                    action = self.local_policy_network.act_greedily(state)
                action, _, _ = self.local_policy_network.act(state)

                next_state, reward, done, _ = self.env.step(action)
                state = next_state
                score += reward

                if done:
                    break
            print(f"Trial: {i+1}  --->  Score: {score}")
        

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    agent = GAE(env, 0.001)
    # agent.run()
    agent.test("checkpoint/GAE/shared-500.pth", record=False)