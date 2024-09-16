import torch
from typing import Tuple


class SparseAutoEncoder(torch.nn.Module):
    # Designed following Raghu's 2017 implementation: https://github.com/aniruddhraghu/sepsisrl/blob/master/continuous/autoencoder.ipynb
    def __init__(self, obs_dim: int, hidden_size: int, sparsity_parameter: float = 5e-2, sparsity_weight: float = 1e-4):
        super().__init__()
        self._encoder = torch.nn.Linear(obs_dim, hidden_size)
        self._decoder = torch.nn.Linear(hidden_size, obs_dim)
        self._sparsity_parameter = sparsity_parameter # p
        self._sparsity_weight = sparsity_weight # beta

    def forward(self, x: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        # We're going to do the forward pass and compute our own loss: Loss = MSE(x, x_hat) + KL_Div(latent weights, sparsity parameter)
        latent_state = self._encoder(x).sigmoid()
        reconstructed_x = self._decoder(latent_state)
        kl_div_loss = self._compute_kl_div(latent_state)
        mse = torch.nn.functional.mse_loss(reconstructed_x, x)
        loss: torch.FloatTensor = mse + self._sparsity_weight * kl_div_loss
        return x, loss

    def encode(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self._encoder(x).sigmoid()

    def _compute_kl_div(self, latent_state: torch.FloatTensor) -> torch.FloatTensor:
        sparsity_p = torch.full((latent_state.shape[0], latent_state.shape[1]), self._sparsity_parameter, device=latent_state.device)
        kl_div_loss: torch.FloatTensor = (sparsity_p * torch.log(sparsity_p / latent_state) + (1 - sparsity_p) * torch.log((1 - sparsity_p) / (1 - latent_state)))
        kl_div_loss = kl_div_loss.sum(dim=-1)
        kl_div_loss = kl_div_loss.mean()
        return kl_div_loss
