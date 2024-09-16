import pandas as pd
from typing import List, Optional
from abc import ABC, abstractmethod


class AbstractRewardFunction(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def update_rewards(self, mimic_df: pd.DataFrame):
        raise NotImplementedError('update_rewards method must be implemented in child class')

    @property
    @abstractmethod
    def raw_data_columns(self) -> Optional[List[str]]:
        raise NotImplementedError('raw_data_columns property must be implemented in child class')
