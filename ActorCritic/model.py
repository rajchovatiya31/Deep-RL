import torch
import torch.nn as nn

class DiscretePolicy(nn.Module):
    """Actor / Policy nework 
    """
    def __init__(self, input_dim, output_dim, hidden_dim=(128,64)):
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
    def __init__(self, input_dim, hidden_dim=(128,64)):
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
    def __init__(self, input_dim, action_bounds, hidden_dim=(128,64)):
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
    def __init__(self, input_dim, output_dim, hidden_dim=(128,64)):
        super().__init__()
        self.in_layer = nn.Linear(input_dim, hidden_dim[0])
        self.hidden_list = nn.ModuleList()

        for i in range(len(hidden_dim) - 1):
            in_dim = hidden_dim[i]
            if i==0:
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
        X = torch.cat((X,a), dim=1)
        Xa = nn.functional.relu(self.in_layer_a(X))
        for layer in self.hidden_list_a:
            Xa = nn.functional.relu(layer(Xa))
        return self.out_layer_a(Xa)       
