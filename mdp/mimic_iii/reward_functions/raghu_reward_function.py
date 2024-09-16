import numpy as np
import pandas as pd
from typing import Optional, List

from mdp.mimic_iii.reward_functions.abstract_reward_function import AbstractRewardFunction


class RaghuRewardFunction(AbstractRewardFunction):
    def __init__(self, terminal_reward_scale: float = 15.0, c_0: float = -0.025, c_1: float = -0.125, c_2: float = -2.0):
        super().__init__()
        self._terminal_reward_scale = terminal_reward_scale
        self._c_0 = c_0
        self._c_1 = c_1
        self._c_2 = c_2

    def update_rewards(self, mimic_df: pd.DataFrame):
        # scale rewards by terminal reward scale first, because the initial rewards are sparse (-1, 0, 1)
        mimic_df['r:reward'] = self._terminal_reward_scale * mimic_df['r:reward']
        # compute intermediate rewards (note: fillna is used to handle the last step of each trajectory)
        sofa_penalty_mask = np.logical_and(np.isclose(mimic_df['o:SOFA_raw'], mimic_df['o:SOFA_next_raw']), mimic_df['o:SOFA_next_raw'] > 0).astype(int)
        sofa_penalty_mask = sofa_penalty_mask.fillna(0)
        sofa_delta_reward = mimic_df['o:SOFA_next_raw'] - mimic_df['o:SOFA_raw']
        sofa_delta_reward = sofa_delta_reward.fillna(0)
        lactate_delta_reward = np.tanh(mimic_df['o:Arterial_lactate_next_raw'] - mimic_df['o:Arterial_lactate_raw'])
        lactate_delta_reward = lactate_delta_reward.fillna(0)
        mimic_df['r:reward'] += self._c_0 * sofa_penalty_mask + self._c_1 * sofa_delta_reward + self._c_2 * lactate_delta_reward

    @property
    def raw_data_columns(self) -> Optional[List[str]]:
        return ['o:SOFA', 'o:Arterial_lactate']
