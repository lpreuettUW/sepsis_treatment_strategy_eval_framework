import pandas as pd
from typing import List, Optional

from mdp.mimic_iii.reward_functions.abstract_reward_function import AbstractRewardFunction


class Sparse(AbstractRewardFunction):
    def __init__(self, reward_scale: float = 1.0):
        super().__init__()
        self._reward_scale = reward_scale

    def update_rewards(self, mimic_df: pd.DataFrame):
        # this works because the default rewards are sparse (1 for living, -1 for dying, and 0 for all other transitions)
        mimic_df['r:reward'] = mimic_df['r:reward'] * self._reward_scale

    @property
    def raw_data_columns(self) -> Optional[List[str]]:
        return None
