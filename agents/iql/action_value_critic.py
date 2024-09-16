import torch


class ActionValueCritic(torch.nn.Module):
    """Adapted from https://github.com/BY571/Implicit-Q-Learning/blob/main/networks.py#L65"""
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self._input = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self._hidden = torch.nn.Linear(hidden_dim, hidden_dim)
        self._output = torch.nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.FloatTensor, action: torch.LongTensor) -> torch.FloatTensor:
        x = torch.cat((state, action.float()), dim=1)
        x = self._input(x).relu()
        x = self._hidden(x).relu()
        x = self._output(x)
        return x
