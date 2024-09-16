import torch


class ExpectileValueCritic(torch.nn.Module):
    """Adapted from https://github.com/BY571/Implicit-Q-Learning/blob/main/networks.py#L81"""
    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self._input = torch.nn.Linear(state_dim, hidden_dim)
        self._hidden = torch.nn.Linear(hidden_dim, hidden_dim)
        self._output = torch.nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.FloatTensor) -> torch.FloatTensor:
        x = self._input(state).relu()
        x = self._hidden(x).relu()
        x = self._output(x)
        return x
