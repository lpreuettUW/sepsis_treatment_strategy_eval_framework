import torch
from abc import ABC, abstractmethod

from agents.abstract_agent import AbstractAgent
from utilities.device_manager import DeviceManager


class AbstractOPEMethod(ABC):
    def __init__(self, trajectory_dataset: torch.utils.data.Dataset, agent: AbstractAgent):
        super().__init__()
        self._trajectory_dataset = trajectory_dataset
        self._agent = agent
        self._device = DeviceManager.get_device()
        self._agent.eval()
        self._initialize()

    @abstractmethod
    def _initialize(self):
        """Initialize the OPE method using a dataset and abstract agent."""
        raise NotImplementedError()

    @abstractmethod
    def reinitialize(self, agent: AbstractAgent):
        """Reinitialize the OPE method."""
        raise NotImplementedError()

    @abstractmethod
    def compute_value(self) -> float:
        raise NotImplementedError()
