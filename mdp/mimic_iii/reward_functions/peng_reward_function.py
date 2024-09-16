import torch
import mlflow
import warnings
import numpy as np
import pandas as pd
from copy import deepcopy
from typing import Tuple, List, Dict, Any

from utilities.device_manager import DeviceManager
from mdp.mimic_iii.reward_functions.abstract_reward_function import AbstractRewardFunction


class PengRewardFunction(AbstractRewardFunction):
    def __init__(self, input_dim: int, hidden_sizes: Tuple[int, int] = (64, 32), weight_decay: float = 1e-4, lr: float = 1e-4):
        super().__init__()
        warnings.warn('The PengRewardFunction has an error in the update_rewards method. Do not use this class until the error is fixed. See "# BROKEN" comment.')
        self._device = DeviceManager.get_device()
        self._network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_sizes[0]),
            torch.nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            torch.nn.Linear(hidden_sizes[1], 1),
            torch.nn.Sigmoid()
        ).to(self._device)
        self._optimizer = torch.optim.Adam(self._network.parameters(), weight_decay=weight_decay, lr=lr)

    def update_rewards(self, mimic_df: pd.DataFrame, batch_size: int = 512):
        start_idx, stop_idx = 0, batch_size
        if 'state' in mimic_df.columns and 'next_state' in mimic_df.columns:
            state_key = 'state'
            next_state_key = 'next_state'
        elif 'normed_state' in mimic_df.columns and 'normed_next_state' in mimic_df.columns:
            state_key = 'normed_state'
            next_state_key = 'normed_next_state'
        else:
            raise ValueError('Could not find state and next_state columns in mimic_df')
        while start_idx < mimic_df.shape[0]:
            batch_view = mimic_df.iloc[start_idx:stop_idx]
            batch_valid_data_mask = ~batch_view[next_state_key].isna()
            batch_view = batch_view[batch_valid_data_mask]
            if batch_view.shape[0] > 0:
                states_tensor = torch.tensor(batch_view[state_key].tolist()).to(self._device)
                next_states_tensor = torch.tensor(batch_view[next_state_key].tolist()).to(self._device)
                with torch.no_grad():
                    state_mortality_preds = self.forward(states_tensor).squeeze(-1)
                    state_comp = (state_mortality_preds / (1 - state_mortality_preds)).log()
                    next_state_mortality_preds = self.forward(next_states_tensor).squeeze(-1)
                    next_state_comp = -(next_state_mortality_preds / (1 - next_state_mortality_preds)).log() # BROKEN: need to scale by next_state_mortality_preds
                    update_mask = batch_valid_data_mask
                    if start_idx > 0:
                        update_mask = np.concatenate((np.zeros(start_idx, dtype=bool), update_mask))
                    if stop_idx < mimic_df.shape[0]:
                        update_mask = np.concatenate((update_mask, np.zeros(mimic_df.shape[0] - stop_idx, dtype=bool)))
                    mimic_df.loc[update_mask, 'r:reward'] = (state_comp + next_state_comp).cpu().numpy() # NOTE: plus bc we put the negative sign in the next_state_comp
            start_idx, stop_idx = stop_idx, min(stop_idx + batch_size, mimic_df.shape[0])

    @property
    def raw_data_columns(self) -> List[str]:
        return list()

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = x.to(self._device)
        return self._network(x)

    def train_batch(self, x: torch.FloatTensor, y: torch.LongTensor) -> Tuple[int, float]:
        x, y = x.to(self._device), y.to(self._device)
        self._optimizer.zero_grad()
        y_hat = self.forward(x).squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy(y_hat, y.float())
        loss.backward()
        self._optimizer.step()
        acc = (torch.where(y_hat > 0.5, 1, 0) == y).long().sum()
        return acc.detach().cpu().item(), loss.detach().cpu().item()

    def evaluate(self, x: torch.FloatTensor, y: torch.LongTensor) -> Tuple[int, float]:
        x, y = x.to(self._device), y.to(self._device)
        y_hat = self.forward(x).squeeze(-1).detach()
        predicted_classes = torch.where(y_hat > 0.5, 1, 0).long()
        accuracy = (predicted_classes == y).sum().cpu().item()
        loss = torch.nn.functional.binary_cross_entropy(y_hat, y.float()).cpu().item()
        return accuracy, loss

    def predict(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = x.to(self._device)
        return self.forward(x).squeeze(-1)

    def get_weights(self) -> Dict[Any, Any]:
        return deepcopy(self._network.state_dict())

    def load_weights(self, weights: Dict[Any, Any]):
        self._network.load_state_dict(weights)

    def save(self, path: str):
        mlflow.pytorch.log_model(self._network, path + '_peng_reward_fn_network')

    def load(self, path: str):
        self._network = mlflow.pytorch.load_model(path + '_peng_reward_fn_network', map_location=self._device)

    def train(self):
        self._network.train()
        
    def eval(self):
        self._network.eval()
