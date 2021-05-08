import torch
import torch.nn as nn
import torch.nn.functional as F

class WeavedMLP(nn.Module):
    def __init__(self, dim_in, dim_out, hidden_dims, n_groups=2, dropout=0.):
        super(WeavedMLP, self).__init__()
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        self.depth = len(hidden_dims)-1
        self.n_groups = n_groups

        self.layers = nn.ModuleList([])
        self.layers.append(
            GroupedLinear(dim_in, 
            hidden_dims[0], 
            n_groups=n_groups)
            )
        for i in range(self.depth):
            self.layers.append(
                GroupedLinear(hidden_dims[i], 
                              hidden_dims[i+1], 
                              n_groups=n_groups)
                )
        self.outlayer = nn.Linear(hidden_dims[-1], dim_out)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        if len(x.shape)==1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif len(x.shape)==2:
            x = x.unsqueeze(1)
        b, c, n = x.shape
        x = x.flatten(start_dim=0, end_dim=1)

        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
            x = self.dropout(x)
            n = int(x.shape[-2]/self.n_groups)
            
            # Verweben
            x = x.reshape((b, c, n, self.n_groups))
            x = x.transpose(-1,-2)
            x = x.flatten(start_dim=-2, end_dim=-1)
        return self.activation(self.outlayer(x))

class GroupedLinear(nn.Module):
    def __init__(self, dim_in, dim_out, n_groups=2):
        super(GroupedLinear, self).__init__()
        self.layer = nn.Conv1d(dim_in, dim_out, 1, groups=n_groups)
    
    def forward(self, x):
        out = x.unsqueeze(-1)
        out = self.layer(out)
        out.squeeze()
        return out