import torch
from abc import ABC, abstractmethod
from typing import Tuple

from mdp.action import Action
from utilities.device_manager import DeviceManager


class AbstractAgent(ABC):
    def __init__(self, policy: torch.nn.Module):
        super().__init__()
        self._device = DeviceManager.get_device()
        self._policy = policy.to(self._device)

    def get_action(self, env_state: torch.FloatTensor) -> Action | Tuple[Action, ...]:
        """Used for training"""
        probs = self._policy(env_state)
        if isinstance(probs, Tuple):
            dists = tuple(torch.distributions.Categorical(prob) for prob in probs)
            return tuple(Action(dist=dist) for dist in dists)
        else:
            dist = torch.distributions.Categorical(probs)
            #print(f'state: {self._state.cpu()}')
            #self._state = self._state.detach()
            return Action(dist=dist)

    def get_best_action(self, env_state: torch.FloatTensor) -> Tuple[Action, torch.FloatTensor] | Tuple[Tuple[Action, ...], Tuple[torch.FloatTensor, ...]]:
        """Used for prediction"""
        probs = self._policy(env_state)
        if isinstance(probs, Tuple):
            actions = tuple(Action(action=prob.argmax(dim=1)) for prob in probs)
            return actions, probs
        else:
            actions = probs.argmax(dim=1)
            actions = Action(action=actions)
            return actions, probs

    def get_action_probs(self, env_state: torch.FloatTensor) -> torch.FloatTensor | Tuple[torch.FloatTensor, ...]:
        probs = self._policy(env_state)
        return probs

    def train(self):
        self._policy.train()

    def eval(self):
        self._policy.eval()
