import pandas as pd
from typing import Optional, List

from mdp.mimic_iii.reward_functions.abstract_reward_function import AbstractRewardFunction


class WuRewardFunction(AbstractRewardFunction):
    def __init__(self, terminal_reward_scale: float = 24.0, intermediate_reward_scale: float = 0.6):
        super().__init__()
        self._terminal_reward_scale = terminal_reward_scale # beta_t in their paper
        self._intermediate_reward_scale = intermediate_reward_scale # beta_s in their paper

    def update_rewards(self, mimic_df: pd.DataFrame):
        """
        Update the rewards in the mimic DataFrame.
        :param mimic_df: Mimic DataFrame.
        """
        # scale rewards by terminal reward scale first, because the initial rewards are sparse (-1, 0, 1)
        mimic_df['r:reward'] = mimic_df['r:reward'] * self._terminal_reward_scale
        # compute intermediate rewards
        physiological_state_change_score = self._intermediate_reward_scale * (mimic_df['o:SOFA_raw'] - mimic_df['o:SOFA_next_raw'])
        # fill the NaN physiological state change scores (this occurs on the last step of each trajectory)
        physiological_state_change_score = physiological_state_change_score.fillna(0)
        mimic_df['r:reward'] += physiological_state_change_score

    @property
    def raw_data_columns(self) -> Optional[List[str]]:
        return ['o:SOFA']
