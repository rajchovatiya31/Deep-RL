import torch
import torch.nn as nn

class DenseDQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=(512,256)):
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
        return self.out_layer(X)

class DuellingDQN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=(512,256)):
        super().__init__()
        self.in_layer = nn.Linear(input_dim, hidden_dim[0])
        self.hidden_list = nn.ModuleList()

        for i in range(len(hidden_dim) - 1):
            hidden_layer = nn.Linear(hidden_dim[i], hidden_dim[i+1])
            self.hidden_list.append(hidden_layer)
        
        self.value_fun = nn.Linear(hidden_dim[-1], 1)
        self.advantage_fun = nn.Linear(hidden_dim[-1], output_dim)
    
    def forward(self, X):
        X = nn.functional.relu(self.in_layer(X))
        for layer in self.hidden_list:
            X = nn.functional.relu(layer(X))
        
        a = self.advantage_fun(X)
        v = self.value_fun(X)
        v = v.expand_as(a)

        q = v + a - a.mean(1, keepdim=True).expand_as(a)

        return q
              
