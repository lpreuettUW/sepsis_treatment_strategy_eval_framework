import torch
from typing import Tuple


class DuelingDQN(torch.nn.Module):
    def __init__(self, state_dim: int, num_actions: int, hidden_dim: int):
        super(DuelingDQN, self).__init__()
        self._input_to_latent_stream = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.LeakyReLU()
        )
        # NOTE: Raghu et al. (2017) split the latent layer in half. The first half is used to compute the value stream, and the second half is used to compute the advantage stream.
        # Here, however, we pass the entire latent layer to value and advantage layers.
        self._latent_to_value_layer = torch.nn.Linear(hidden_dim, 1)
        self._latent_to_advantage_layer = torch.nn.Linear(hidden_dim, num_actions)

    def forward(self, states: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        latents = self._input_to_latent_stream(states)
        values = self._latent_to_value_layer(latents)
        advantages = self._latent_to_advantage_layer(latents)
        return values, advantages
