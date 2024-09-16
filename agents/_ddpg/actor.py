import torch
from typing import Tuple

from utilities.device_manager import DeviceManager


class Actor(torch.nn.Module):
    def __init__(self, state_dim: int, action_dim: int, action_mins: Tuple[float, ...], action_maxs: Tuple[float, ...], hidden_size: int):
        super().__init__()
        device = DeviceManager.get_device()
        self._action_mins = torch.tensor([v for v in action_mins], device=device, dtype=torch.float32)
        self._action_maxs = torch.tensor([v for v in action_maxs], device=device, dtype=torch.float32)
        # following Lin et al. 2023
        self._net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, action_dim),
            torch.nn.Sigmoid() # bc we normalize the actions to [0, 1]
        ).float()

    def forward(self, state: torch.FloatTensor) -> torch.FloatTensor:
        raw_actions = self._net(state)
        clamped_actions = raw_actions.clamp(min=self._action_mins, max=self._action_maxs)
        return clamped_actions
