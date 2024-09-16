import torch
from abc import abstractmethod
from typing import Dict

from mdp.action import Action
from agents.abstract_agent import AbstractAgent


class AbstractBatchAgent(AbstractAgent):
    def __init__(self, policy: torch.nn.Module):
        super().__init__(policy)

    # region Abstract Functions/Methods

    @abstractmethod
    def batch_update(self, states: torch.FloatTensor, actions_taken: Action, rewards: torch.FloatTensor, next_states: torch.FloatTensor, dones: torch.BoolTensor) -> Dict[str, float]:
        """
        Perform batch update.
        :param states: Batch of environment states
        :param actions_taken: Batch of actions taken by agent
        :param rewards: Batch of observed rewards
        :param next_states: Batch of next environment state of environment. May be in batch form.
        :param dones: Batch of done flags. May be in batch form.
        :return: Dictionary of losses observed for the given batch.
        """
        raise NotImplementedError()

    # endregion

    # region Functions/Methods

    # region Public

    # endregion

    # region Private

    def _get_log_probs(self, states: torch.FloatTensor, actions: torch.LongTensor) -> torch.FloatTensor:
        probs = self._policy(states)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions.squeeze()).unsqueeze(-1)
        return log_probs

    # endregion

    # endregion
