import torch
import numpy as np

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


class NormalDecayStrategy:
    def __init__(self, bounds, init_noise_ratio=0.5, min_noise_ratio=0.1, decay_steps=200000):
        """Strategy to add gaussian noise

        Args:
            bounds (tuple): action bounds
            init_noise_ratio (float, optional): Initial noise ratio. Defaults to 0.5.
            min_noise_ratio (float, optional): Minimum noise ratio. Defaults to 0.1.
            decay_steps (int, optional): over n steps to decay noise ratio. Defaults to 200000.
        """
        self.low, self.high = bounds
        self.init_noise_ratio = init_noise_ratio
        self.min_noise_ratio = min_noise_ratio
        self.noise_rate = init_noise_ratio

        self.decay_steps = decay_steps
        self.t = 0

    def noise_rate_update(self):
        noise_rate = 1 - self.t / self.decay_steps
        noise_rate = (self.init_noise_ratio - self.min_noise_ratio) * \
            noise_rate + self.min_noise_ratio
        noise_rate = np.clip(
            noise_rate, self.min_noise_ratio, self.init_noise_ratio)
        self.t += 1
        return noise_rate

    def select_action(self, model, state, max_exploration=False):
        """function to select action

        Args:
            model (model): pytorch model
            state (tensor): state tensor
            max_exploration (bool, optional): if true, do exploration. Defaults to False.

        Returns:
            float: action value
        """
        state = torch.from_numpy(state).float().unsqueeze(0)
        if max_exploration:
            noise_scale = self.high
        else:
            noise_scale = self.noise_rate * self.high

        with torch.no_grad():
            greedy_action = model(state).cpu().detach().data.numpy().squeeze()

        noise = np.random.normal(loc=0, scale=noise_scale, size=len(self.high))
        action = np.clip(greedy_action + noise, self.low, self.high)

        self.noise_rate = self.noise_rate_update()
        return action
