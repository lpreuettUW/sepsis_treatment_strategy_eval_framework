import torch
from typing import Tuple

from utilities.device_manager import DeviceManager


class Critic(torch.nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int):
        super().__init__()
        device = DeviceManager.get_device()
        # following Lin et al. 2023
        self._state_input_layer = torch.nn.Linear(state_dim, hidden_size).float()
        self._state_latent_action_layer = torch.nn.Linear(hidden_size + action_dim, hidden_size).float()
        self._output_layer = torch.nn.Linear(hidden_size, 1).float()

    def forward(self, state: torch.FloatTensor, action: torch.FloatTensor) -> torch.FloatTensor:
        state_latent = torch.nn.functional.leaky_relu(self._state_input_layer(state))
        state_latent_action_input = torch.cat((state_latent, action), dim=-1)
        state_latent_action = torch.nn.functional.leaky_relu(self._state_latent_action_layer(state_latent_action_input))
        value = self._output_layer(state_latent_action)
        return value
