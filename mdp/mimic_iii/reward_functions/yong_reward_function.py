import numpy as np
import pandas as pd
from typing import List, Optional

from mdp.mimic_iii.reward_functions.abstract_reward_function import AbstractRewardFunction


class YongRewardFunction(AbstractRewardFunction):
    """
    Young et al.'s reward function proposed in Reinforcement learning for sepsis treatment: A continuous action space solution.
    https://proceedings.mlr.press/v182/huang22a.html
    """
    def __init__(self, lambda_0: float = -0.25, lambda_1: float = -0.2):
        super().__init__()
        self._lambda_0 = lambda_0
        self._lambda_1 = lambda_1

    def update_rewards(self, mimic_df: pd.DataFrame):
        current_sofa_score = self._lambda_0 * np.tanh(mimic_df['o:SOFA_raw'] - 6)
        physiological_state_change_score = self._lambda_1 * (mimic_df['o:SOFA_next_raw'] - mimic_df['o:SOFA_raw'])
        # fill the NaN physiological state change scores (this occurs on the last step of each trajectory)
        physiological_state_change_score = physiological_state_change_score.fillna(0)
        mimic_df['r:reward'] = current_sofa_score + physiological_state_change_score

    @property
    def raw_data_columns(self) -> Optional[List[str]]:
        return ['o:SOFA']
