import torch
from abc import abstractmethod
from typing import Dict, Tuple
#from overrides import override

from mdp.action import Action
from agents.abstract_agent import AbstractAgent


class AbstractSequenceAgent(AbstractAgent):
    def __init__(self, policy: torch.nn.Module):
        super().__init__(policy)

    # region Abstract Functions/Methods

    @abstractmethod
    def batch_update(self, tokens: torch.LongTensor, targets: torch.LongTensor, loss_pad_mask: torch.BoolTensor) -> Dict[str, float]:
        """
        Perform batch update.
        tokens: input tokens of shape (batch_size, seq_len) where seq_len is the number of <observation, action, reward, value> tuples in the sub-trajectory
        targets: target tokens of shape (batch_size, seq_len) where seq_len is the number of <observation, action, reward, value> tuples in the sub-trajectory
        loss_pad_mask: mask of non-pad tokens of shape (batch_size, seq_len) where seq_len is the number of <observation, action, reward, value> tuples in the sub-trajectory
        :return: Dictionary of losses observed for the given batch.
        """
        raise NotImplementedError()

    # region AbstractAgent

    @abstractmethod
    def get_action(self, env_state: torch.FloatTensor) -> Action | Tuple[Action, ...]:
        raise NotImplementedError()

    @abstractmethod
    def get_best_action(self, env_state: torch.FloatTensor) -> Tuple[Action, torch.FloatTensor] | Tuple[Tuple[Action, ...], Tuple[torch.FloatTensor, ...]]:
        raise NotImplementedError()

    # endregion

    # endregion

    # region Functions/Methods

    # region Public

    # endregion

    # region Private

    # endregion

    # endregion
