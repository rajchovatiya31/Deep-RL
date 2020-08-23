import torch
import torch.nn as nn

class DiscretePolicy(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=(128,64)):
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

    def act(self, state, device='cpu'):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        logits = self.forward(state)
        distribution = torch.distributions.Categorical(logits=logits)
        action = distribution.sample()
        log_prob = distribution.log_prob(action).unsqueeze(-1)
        entropy = distribution.entropy().unsqueeze(-1)
        return action.item(), log_prob, entropy
    
    def act_greedily(self, state, device='cpu'):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state)
        return torch.argmax(probs).item()

class DenseValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=(128,64)):
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