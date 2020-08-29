import torch
import torch.nn as nn
import numpy as np


class DiscretePolicy(nn.Module):
    """Actor / Policy nework 
    """

    def __init__(self, input_dim, output_dim, hidden_dim=(128, 64)):
        """Initalize nework

        Args:
            input_dim (int): number of inputs
            output_dim (int): number of outputs
            hidden_dim (tuple, optional): tuple of ints for hidden layer dimension. Defaults to (128,64).
        """
        super().__init__()
        self.in_layer = nn.Linear(input_dim, hidden_dim[0])

        self.hidden_list = nn.ModuleList()
        for i in range(len(hidden_dim) - 1):
            hidden_layer = nn.Linear(hidden_dim[i], hidden_dim[i+1])
            self.hidden_list.append(hidden_layer)

        self.out_layer = nn.Linear(hidden_dim[-1], output_dim)

    def forward(self, X):
        X = nn.functional.relu(self.in_layer(X))
        for layer in self.hidden_list:
            X = nn.functional.relu(layer(X))
        X = self.out_layer(X)
        return X

    def act(self, state):
        """choose action according to distribution

        Args:
            state (ndarray): state of the enviornment

        Returns:
            int:     action
            tensor: log probabilities for the actions
            tensor: entropies for the actions
        """
        state = torch.from_numpy(state).float().unsqueeze(0)
        logits = self.forward(state)
        distribution = torch.distributions.Categorical(logits=logits)
        action = distribution.sample()
        log_prob = distribution.log_prob(action).unsqueeze(-1)
        entropy = distribution.entropy().unsqueeze(-1)
        return action.item(), log_prob, entropy

    def act_greedily(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(state)
        return torch.argmax(probs).item()


class DenseValueNetwork(nn.Module):
    """Critic / Value network
    """

    def __init__(self, input_dim, hidden_dim=(128, 64)):
        """Initalize nework

        Args:
            input_dim (int): number of inputs
            output_dim (int): number of outputs
            hidden_dim (tuple, optional): tuple of ints for hidden layer dimension. Defaults to (128,64).
        """
        super().__init__()
        self.in_layer = nn.Linear(input_dim, hidden_dim[0])

        self.hidden_list = nn.ModuleList()
        for i in range(len(hidden_dim) - 1):
            hidden_layer = nn.Linear(hidden_dim[i], hidden_dim[i+1])
            self.hidden_list.append(hidden_layer)

        self.out_layer = nn.Linear(hidden_dim[-1], 1)

    def forward(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(X).float().unsqueeze(0)
        X = nn.functional.relu(self.in_layer(X))
        for layer in self.hidden_list:
            X = nn.functional.relu(layer(X))
        X = self.out_layer(X)
        return X


class DenseActorCritic(nn.Module):
    """Combined Actor-Critic Network
    """

    def __init__(self, input_dim, output_dim, hidden_dim=(128, 64)):
        """Initalize nework

        Args:
            input_dim (int): number of inputs
            output_dim (int): number of outputs
            hidden_dim (tuple, optional): tuple of ints for hidden layer dimension. Defaults to (128,64).
        """
        super().__init__()
        self.in_layer = nn.Linear(input_dim, hidden_dim[0])
        self.hidden_list = nn.ModuleList()

        for i in range(len(hidden_dim) - 1):
            hidden_layer = nn.Linear(hidden_dim[i], hidden_dim[i-1])
            self.hidden_list.append(hidden_layer)

        self.out_policy = nn.Linear(hidden_dim[-1], output_dim)
        self.out_value = nn.Linear(hidden_dim[-1], 1)

    def forward(self, X):
        X = nn.functional.relu(self.in_layer(X))

        for layer in self.hidden_list:
            X = nn.functional.relu(layer(X))
        X_policy = self.out_policy(X)
        X_value = self.out_value(X)
        return X_policy, X_value

    def act(self, state):
        """choose action according to distribution

        Args:
            state (ndarray): state of the enviornment

        Returns:
            int:     action
            tensor: log probabilities for the actions
            tensor: entropies for the actions
            tensor: q-value for the predicted actions
        """
        state = torch.from_numpy(state).float()
        logits, values = self.forward(state)
        distribution = torch.distributions.Categorical(logits=logits)
        action = distribution.sample()
        log_prob = distribution.log_prob(action).unsqueeze(-1)
        entropy = distribution.entropy().unsqueeze(-1)
        action = action.item() if len(action) == 1 else action.data.numpy()
        return action, log_prob, entropy, values

    def act_greedily(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        logits, _ = self.forward(state)
        return torch.argmax(logits).item()

    def evaluate_state(self, state):
        state = torch.from_numpy(state).float()
        _, value = self.forward(state)
        return value


class DDPGPolicyNet(nn.Module):
    def __init__(self, input_dim, action_bounds, hidden_dim=(128, 64)):
        super().__init__()
        self.action_min, self.action_max = action_bounds

        self.in_layer = nn.Linear(input_dim, hidden_dim[0])
        self.hidden_list = nn.ModuleList()

        for i in range(len(hidden_dim) - 1):
            layer = nn.Linear(hidden_dim[i], hidden_dim[i+1])
            self.hidden_list.append(layer)

        self.out_layer = nn.Linear(hidden_dim[-1], len(self.action_max))

        self.action_max = torch.tensor(self.action_max, dtype=torch.float32)
        self.action_min = torch.tensor(self.action_min, dtype=torch.float32)
        self.max_val = nn.functional.tanh(torch.Tensor([float('inf')]))
        self.min_val = nn.functional.tanh(torch.Tensor([float('-inf')]))

        self.rescale_fun = lambda x: (x - self.min_val) * (self.action_max - self.action_min) / \
            (self.max_val - self.min_val) + self.action_min

    def forward(self, X):
        X = nn.functional.relu(self.in_layer(X))
        for layer in self.hidden_list:
            X = nn.functional.relu(layer(X))
        X = self.out_layer(X)
        X = nn.functional.tanh(X)
        return self.rescale_fun(X)


class DDPGValueNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=(128, 64)):
        super().__init__()
        self.in_layer = nn.Linear(input_dim, hidden_dim[0])
        self.hidden_list = nn.ModuleList()

        for i in range(len(hidden_dim) - 1):
            in_dim = hidden_dim[i]
            if i == 0:
                in_dim += output_dim
            layer = nn.Linear(in_dim, hidden_dim[i+1])
            self.hidden_list.append(layer)
        self.out_layer = nn.Linear(hidden_dim[-1], 1)

    def forward(self, X, a):
        X = nn.functional.relu(self.in_layer(X))

        for i, layer in enumerate(self.hidden_list):
            if i == 0:
                X = torch.cat((X, a), dim=1)
            X = nn.functional.relu(layer(X))
        return self.out_layer(X)


class TwinDDPG(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=(128, 64)):
        super().__init__()
        self.in_layer_a = nn.Linear(input_dim + output_dim, hidden_dim[0])
        self.in_layer_b = nn.Linear(input_dim + output_dim, hidden_dim[0])

        self.hidden_list_a = nn.ModuleList()
        self.hidden_list_b = nn.ModuleList()

        for i in range(len(hidden_dim) - 1):
            layer_a = nn.Linear(hidden_dim[i], hidden_dim[i+1])
            self.hidden_list_a.append(layer_a)
            layer_b = nn.Linear(hidden_dim[i], hidden_dim[i+1])
            self.hidden_list_b.append(layer_b)

        self.out_layer_a = nn.Linear(hidden_dim[-1], output_dim)
        self.out_layer_b = nn.Linear(hidden_dim[-1], output_dim)

    def forward(self, X, a):
        X = torch.cat((X, a), dim=1)

        Xa = nn.functional.relu(self.in_layer_a(X))
        Xb = nn.functional.relu(self.in_layer_b(X))

        for layer_a, layer_b in zip(self.hidden_list_a, self.hidden_list_b):
            Xa = nn.functional.relu(layer_a(Xa))
            Xb = nn.functional.relu(layer_b(Xb))

        return self.out_layer_a(Xa), self.out_layer_b(Xb)

    def forward_Q(self, X, a):
        X = torch.cat((X, a), dim=1)
        Xa = nn.functional.relu(self.in_layer_a(X))
        for layer in self.hidden_list_a:
            Xa = nn.functional.relu(layer(Xa))
        return self.out_layer_a(Xa)


class SACValueNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=(128, 64)):
        super().__init__()
        self.in_layer = nn.Linear(input_dim + output_dim, hidden_dim[0])
        
        self.hidden_list = nn.ModuleList()
        for i in range(len(hidden_dim) - 1):
            layer = nn.Linear(hidden_dim[i], hidden_dim[i+1])
            self.hidden_list.append(layer)
        
        self.out_layer = nn.Linear(hidden_dim[-1], 1)

    def forward(self, X, a):
        X = torch.cat((X, a), dim=1)
        Xa = nn.functional.relu(self.in_layer(X))
        for layer in self.hidden_list:
            Xa = nn.functional.relu(layer(Xa))
        return self.out_layer(Xa)


class SACGaussianPolicy(nn.Module):
    def __init__(self, input_dim, action_bounds, log_std_min=-20, log_std_max=2, entropy_lr=0.001, hidden_dim=(128, 64)):
        super().__init__()
        self.action_min, self.action_max = action_bounds
        self.log_std_max = log_std_max
        self.log_std_min = log_std_min

        self.in_layer = nn.Linear(input_dim, hidden_dim[0])
        self.hidden_list = nn.ModuleList()
        for i in range(len(hidden_dim) - 1):
            layer = nn.Linear(hidden_dim[i], hidden_dim[i+1])
            self.hidden_list.append(layer)
        self.out_layer_mean = nn.Linear(hidden_dim[-1], len(self.action_max))
        self.out_layer_std = nn.Linear(hidden_dim[-1], len(self.action_max))

        self.action_max = torch.tensor(self.action_max, dtype=torch.float32)
        self.action_min = torch.tensor(self.action_min, dtype=torch.float32)
        self.max_val = nn.functional.tanh(torch.Tensor([float('inf')]))
        self.min_val = nn.functional.tanh(torch.Tensor([float('-inf')]))
        self.rescale_fun = lambda x: (x - self.min_val) * (self.action_max - self.action_min) / \
                                     (self.max_val - self.min_val) + self.action_min

        self.target_entropy = -np.prod(self.action_max.shape)
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=entropy_lr)

    def forward(self, X):
        X = nn.functional.relu(self.in_layer(X))
        for layer in self.hidden_list:
            X = nn.functional.relu(layer(X))
        X_mean = self.out_layer_mean(X)
        X_std = self.out_layer_std(X)
        X_std = torch.clamp(X_std, self.log_std_min, self.log_std_max)
        return X_mean, X_std

    def full_pass(self, state, epsilon=1e-6):
        mean, std = self.forward(state)

        pi_s = torch.distributions.Normal(mean, std.exp())
        pre_tanh_action = pi_s.rsample()
        tanh_action = torch.tanh(pre_tanh_action)
        action = self.rescale_fun(tanh_action)

        log_prob = pi_s.log_prob(
            pre_tanh_action) - torch.log((1 - tanh_action.pow(2)).clamp(0, 1) + epsilon)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob, self.rescale_fun(torch.tanh(mean))

    def _update_exploration_ratio(self, greedy_action, action_taken):
        action_min, action_max = self.action_min.cpu().numpy(), self.action_max.cpu().numpy()
        self.exploration_ratio = np.mean(
            abs((greedy_action - action_taken)/(action_max - action_min)))

    def _get_actions(self, state):
        state = torch.from_numpy(state).unsqueeze(0)
        mean, log_std = self.forward(state)

        action = self.rescale_fun(torch.tanh(
            torch.distributions.Normal(mean, log_std.exp()).sample()))
        greedy_action = self.rescale_fun(torch.tanh(mean))
        random_action = np.random.uniform(low=self.action_min.cpu().numpy(),
                                          high=self.action_max.cpu().numpy())

        action_shape = self.action_max.cpu().numpy().shape
        action = action.detach().cpu().numpy().reshape(action_shape)
        greedy_action = greedy_action.detach().cpu().numpy().reshape(action_shape)
        random_action = random_action.reshape(action_shape)

        return action, greedy_action, random_action

    def select_random_action(self, state):
        action, greedy_action, random_action = self._get_actions(state)
        self._update_exploration_ratio(greedy_action, random_action)
        return random_action

    def select_greedy_action(self, state):
        action, greedy_action, random_action = self._get_actions(state)
        self._update_exploration_ratio(greedy_action, greedy_action)
        return greedy_action

    def select_action(self, state):
        action, greedy_action, random_action = self._get_actions(state)
        self._update_exploration_ratio(greedy_action, action)
        return action
        