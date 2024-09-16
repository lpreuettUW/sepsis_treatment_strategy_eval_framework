from typing import Literal

from mdp.mimic_iii.reward_functions.abstract_reward_function import AbstractRewardFunction
from mdp.mimic_iii.reward_functions.sparse import Sparse
from mdp.mimic_iii.reward_functions.yong_reward_function import YongRewardFunction
from mdp.mimic_iii.reward_functions.raghu_reward_function import RaghuRewardFunction
from mdp.mimic_iii.reward_functions.wu_reward_function import WuRewardFunction
from mdp.mimic_iii.reward_functions.peng_reward_function import PengRewardFunction


class Factory:
    @staticmethod
    def create(reward_function_name: Literal['sparse', 'yong', 'wu', 'raghu', 'peng']) -> AbstractRewardFunction:
        match reward_function_name.lower():
            case 'sparse':
                return Sparse(reward_scale=15.0)
            case 'yong':
                return YongRewardFunction(lambda_0=-0.25, lambda_1=-0.2)
            case 'wu':
                return WuRewardFunction(terminal_reward_scale=24.0, intermediate_reward_scale=0.6)
            case 'raghu':
                return RaghuRewardFunction(terminal_reward_scale=15.0, c_0=-0.025, c_1=-0.125, c_2=-2.0)
            case 'peng':
                return PengRewardFunction(input_dim=47, hidden_sizes=(64, 32), weight_decay=1e-4, lr=1e-3)
            case _:
                raise ValueError(f'Unknown reward function: {reward_function_name}')
