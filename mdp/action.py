import torch
from typing import Optional


class Action:
    def __init__(self, dist: Optional[torch.distributions.Categorical] = None, action: Optional[torch.LongTensor] = None):
        if dist is None and action is None:
            raise ValueError('either dist or action must be provided')
        if dist is not None and action is not None:
            raise ValueError('can only provide either dist or action')
        if dist is not None:
            self._action = dist.sample()
            self._log_prob = dist.log_prob(self._action)
        else: #action is not None:
            self._action = action
            self._log_prob = None

    @property
    def action(self) -> torch.LongTensor:
        return self._action

    @property
    def log_prob(self) -> Optional[torch.FloatTensor]:
        return self._log_prob
