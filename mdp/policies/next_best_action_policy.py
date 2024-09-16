import torch


# class NextBestActionPolicy(torch.nn.Module):
#     def __init__(self, state_dim: int, hidden_dim: int, num_actions: int):
#         super().__init__()
#         self._linear_in = torch.nn.Linear(state_dim, hidden_dim)
#         self._linear_hidden0 = torch.nn.Linear(hidden_dim, hidden_dim)
#         self._linear_hidden1 = None #torch.nn.Linear(hidden_dim, hidden_dim)
#         self._linear_out = torch.nn.Linear(hidden_dim, num_actions)
#         self._init()
#
#     def _init(self):
#         for layer in (self._linear_in, self._linear_hidden0, self._linear_hidden1, self._linear_out):
#             if layer:
#                 layer.bias.data.fill_(0)
#                 torch.nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
#
#     def forward(self, state: torch.FloatTensor) -> torch.FloatTensor:
#         x = state.clone() # clone bc inplace operations were messing with the backward pass
#         for layer in (self._linear_in, self._linear_hidden0, self._linear_hidden1):
#             if layer:
#                 x = layer(x)
#                 x = x.relu()
#         x = self._linear_out(x)
#         x = x.softmax(dim=1)
#         return x


class NextBestActionPolicy(torch.nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, num_actions: int, dropout: float = 0.5):
        super().__init__()
        self._linear_in = torch.nn.Linear(state_dim, hidden_dim)
        self._linear_out = torch.nn.Linear(hidden_dim, num_actions)
        self._drop = torch.nn.Dropout(dropout)

    def _init(self):
        for layer in (self._linear_in, self._linear_out):
            if layer:
                layer.bias.data.fill_(0)
                torch.nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, state: torch.FloatTensor) -> torch.FloatTensor:
        state = state.clone() # clone bc inplace operations were messing with the backward pass
        x = self._linear_in(state)
        # x = self._drop(x)
        x = x.relu()
        x = self._linear_out(x)
        x = x.softmax(dim=1)
        return x
