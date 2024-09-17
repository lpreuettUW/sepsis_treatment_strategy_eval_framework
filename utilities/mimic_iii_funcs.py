import os
import math
import torch
import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm
import pandas as pd
from typing import Literal, Tuple, Optional, List
from sklearn.preprocessing import MinMaxScaler


def load_standardized_mimic_data(split: int, load_with_raw_data_columns: Optional[List[str]] = None) -> pd.DataFrame:
    # load split data
    split_base_path = '<your_base_path>/off_policy_policy_evaluation/datasets/mimic_iii/stratified_splits'
    split_file = os.path.join(f'{split_base_path}', f'split_{split}.csv')
    split_data = pd.read_csv(split_file, index_col=0)
    # load mimic data
    mimic_path = '<your_base_path>/off_policy_policy_evaluation/datasets/mimic_iii/preprocessed_cohort/sepsis_final_data_withTimes_90_day_death_window.csv'
    mimic_data = pd.read_csv(mimic_path)
    # merge data
    merged_mimic_data = mimic_data.merge(split_data, on='traj', how='inner')
    assert merged_mimic_data.shape[0] == mimic_data.shape[0], 'Data was not merged correctly'
    if load_with_raw_data_columns is not None:
        raw_data = load_raw_mimic_data(split)
        for merge_col in ('traj', 'step'):
            if merge_col not in load_with_raw_data_columns:
                load_with_raw_data_columns.append(merge_col)
        merged_mimic_data = merged_mimic_data.merge(raw_data[load_with_raw_data_columns], on=['traj', 'step'], how='inner', suffixes=('', '_raw'))
        # add next values to facilitate computing rewards
        for col in load_with_raw_data_columns:
            if col not in ('traj', 'step'):
                merged_mimic_data[f'{col}_next_raw'] = np.roll(merged_mimic_data[f'{col}_raw'].values, -1)
                merged_mimic_data.loc[merged_mimic_data['step'] == merged_mimic_data['traj_length'] - 1, f'{col}_next_raw'] = None # there's no next state for the last step
    return merged_mimic_data


def load_raw_mimic_data(split: int) -> pd.DataFrame:
    # load split data
    split_base_path = '<your_base_path>/off_policy_policy_evaluation/datasets/mimic_iii/stratified_splits'
    split_file = os.path.join(f'{split_base_path}', f'split_{split}.csv')
    split_data = pd.read_csv(split_file, index_col=0)
    # load mimic data
    mimic_path = '<your_base_path>/off_policy_policy_evaluation/datasets/mimic_iii/preprocessed_cohort/sepsis_final_data_RAW_withTimes_90_day_death_window.csv'
    mimic_data = pd.read_csv(mimic_path)
    # merge data
    merged_mimic_data = mimic_data.merge(split_data, on='traj', how='inner')
    assert merged_mimic_data.shape[0] == mimic_data.shape[0], 'Data was not merged correctly'
    return merged_mimic_data


def drop_all_raw_data_columns(data: pd.DataFrame) -> pd.DataFrame:
    raw_data_columns = [col for col in data.columns if '_raw' in col]
    return data.drop(columns=raw_data_columns)


def preprocess_mimic_data(data: pd.DataFrame, preprocess_type: Literal['sparse_auto_encoder', 'iql_discrete', 'kmeans_sarsa', 'ope', 'ddpg_cont', 'peng_reward_fn', 'mortality_plots']) -> Tuple[pd.DataFrame, Optional[MinMaxScaler], Optional[MinMaxScaler]]:
    match preprocess_type:
        case 'sparse_auto_encoder':
            # drop columns
            columns_to_drop = ['traj', 'step', 'traj_length', 'm:presumed_onset', 'm:charttime', 'm:icustayid', 'a:action', 'a:cont_iv_fluid', 'a:cont_vp', 'r:reward', 'survived']
            data = data.drop(columns=columns_to_drop)
            assert data.shape[1] == 48, 'Data was not preprocessed correctly' # 48 = num_feats (47) + split (1)
            # min-max scaling
            state_scaler = MinMaxScaler()
            data.loc[:, data.columns != 'split'] = state_scaler.fit_transform(data.loc[:, data.columns != 'split'])
            return data, state_scaler, None
        case 'iql_discrete':
            # drop columns
            columns_to_drop = ['m:presumed_onset', 'm:charttime', 'm:icustayid', 'a:cont_iv_fluid', 'a:cont_vp', 'survived']
            data = data.drop(columns=columns_to_drop)
            # add done column
            data['done'] = data.step == data.traj_length - 1
            # min-max scaling
            state_mask = ((data.columns != 'split') & (data.columns != 'r:reward') & (data.columns != 'a:action')
                         & (data.columns != 'traj') & (data.columns != 'step') & (data.columns != 'traj_length')
                         & (data.columns != 'done'))
            state_scaler = MinMaxScaler()
            data.loc[:, state_mask] = state_scaler.fit_transform(data.loc[:, state_mask])
            # add state and next state columns
            states = data.loc[:, state_mask].values.tolist()
            data['state'] = states
            data['next_state'] = states[1:] + states[:1]  # roll the states by 1
            data.loc[data.done, 'next_state'] = None
            # remove unnecessary columns
            cols = data.columns.to_list()
            for col_to_keep in ('state', 'next_state', 'r:reward', 'a:action', 'traj', 'step', 'traj_length', 'split', 'done'):
                cols.remove(col_to_keep)
            data.drop(columns=cols, inplace=True)
            return data, state_scaler, None
        case 'ddpg_cont':
            # drop columns
            columns_to_drop = ['m:presumed_onset', 'm:charttime', 'm:icustayid', 'a:action', 'survived']
            data = data.drop(columns=columns_to_drop)
            # add done column
            data['done'] = data.step == data.traj_length - 1
            # min-max scaling
            state_mask = ((data.columns != 'split') & (data.columns != 'r:reward') & (data.columns != 'a:cont_iv_fluid')
                         & (data.columns != 'a:cont_vp') & (data.columns != 'traj') & (data.columns != 'step')
                         & (data.columns != 'traj_length') & (data.columns != 'done'))
            state_scaler = MinMaxScaler()
            data.loc[:, state_mask] = state_scaler.fit_transform(data.loc[:, state_mask])
            # scale actions
            action_scaler = MinMaxScaler()
            data.loc[:, ['a:cont_iv_fluid', 'a:cont_vp']] = action_scaler.fit_transform(data.loc[:, ['a:cont_iv_fluid', 'a:cont_vp']])
            # add state and next state columns
            states = data.loc[:, state_mask].values.tolist()
            data['state'] = states
            data['next_state'] = states[1:] + states[:1]  # roll the states by 1
            data.loc[data.done, 'next_state'] = None
            # remove unnecessary columns
            cols = data.columns.to_list()
            for col_to_keep in ('state', 'next_state', 'r:reward', 'a:cont_iv_fluid', 'a:cont_vp', 'traj', 'step', 'traj_length', 'split', 'done'):
                cols.remove(col_to_keep)
            data.drop(columns=cols, inplace=True)
            return data, state_scaler, action_scaler
        case 'kmeans_sarsa':
            # drop columns
            columns_to_drop = ['m:presumed_onset', 'm:charttime', 'm:icustayid', 'a:cont_iv_fluid', 'a:cont_vp', 'survived']
            data = data.drop(columns=columns_to_drop)
            # add done column
            data['done'] = data.step == data.traj_length - 1
            # add state and next state columns
            state_mask = ((data.columns != 'split') & (data.columns != 'r:reward') & (data.columns != 'a:action') & (data.columns != 'traj')
                          & (data.columns != 'step') & (data.columns != 'traj_length') & (data.columns != 'done'))
            states = data.loc[:, state_mask].values.tolist()
            data['state'] = states
            data['next_state'] = states[1:] + states[:1]  # roll the states by 1
            data.loc[data.done, 'next_state'] = None
            # remove unnecessary columns
            cols = data.columns.to_list()
            for col_to_keep in ('state', 'next_state', 'r:reward', 'a:action', 'traj', 'step', 'split', 'done', 'traj_length'):
                cols.remove(col_to_keep)
            data.drop(columns=cols, inplace=True)
            return data, None, None
        case 'mortality_plots':
            # drop columns
            columns_to_drop = ['m:presumed_onset', 'm:charttime', 'm:icustayid', 'a:cont_iv_fluid', 'a:cont_vp']
            data = data.drop(columns=columns_to_drop)
            # add done column
            data['done'] = data.step == data.traj_length - 1
            # add state and next state columns
            state_mask = ((data.columns != 'split') & (data.columns != 'r:reward') & (data.columns != 'a:action') & (data.columns != 'traj')
                          & (data.columns != 'step') & (data.columns != 'traj_length') & (data.columns != 'done') & (data.columns != 'survived'))
            states = data.loc[:, state_mask].values.tolist()
            data['state'] = states
            data['next_state'] = states[1:] + states[:1]  # roll the states by 1
            data.loc[data.done, 'next_state'] = None
            # remove unnecessary columns
            cols = data.columns.to_list()
            for col_to_keep in ('state', 'next_state', 'r:reward', 'a:action', 'traj', 'step', 'split', 'done', 'traj_length', 'survived'):
                cols.remove(col_to_keep)
            data.drop(columns=cols, inplace=True)
            return data, None, None
        case 'ope':
            # we need state representations for both behavior policy and evaluation policy
            # our current behavior policy is a k-means sarsa agent
            # to support this we will learn the MinMaxScaler and return it, BUT WE WILL NOT SCALE THE DATA
            # drop columns
            columns_to_drop = ['m:presumed_onset', 'm:charttime', 'm:icustayid', 'survived']
            data = data.drop(columns=columns_to_drop)
            # add done column
            data['done'] = data.step == data.traj_length - 1
            # min-max scaling
            state_mask = ((data.columns != 'split') & (data.columns != 'r:reward') & (data.columns != 'a:action')
                         & (data.columns != 'traj') & (data.columns != 'step') & (data.columns != 'traj_length')
                         & (data.columns != 'done') & (data.columns != 'a:cont_iv_fluid') & (data.columns != 'a:cont_vp'))
            state_scaler = MinMaxScaler()
            state_scaler.fit(data.loc[:, state_mask]) # learn the scaler, but don't transform the data
            # add state and next state columns
            states = data.loc[:, state_mask].values.tolist()
            min_maxed_states = state_scaler.transform(data.loc[:, state_mask]).tolist()
            data['raw_state'] = states
            data['raw_next_state'] = states[1:] + states[:1]  # roll the states by 1
            data.loc[data.done, 'raw_next_state'] = None
            data['normed_state'] = min_maxed_states
            data['normed_next_state'] = min_maxed_states[1:] + min_maxed_states[:1]  # roll the states by 1
            data.loc[data.done, 'normed_next_state'] = None
            # scale actions
            action_scaler = MinMaxScaler()
            data.loc[:, ['a:cont_iv_fluid', 'a:cont_vp']] = action_scaler.fit_transform(data.loc[:, ['a:cont_iv_fluid', 'a:cont_vp']])
            # remove unnecessary columns
            cols = data.columns.to_list()
            for col_to_keep in ('raw_state', 'normed_state', 'raw_next_state', 'normed_next_state', 'r:reward', 'a:action', 'traj', 'step', 'traj_length', 'split', 'done', 'a:cont_iv_fluid', 'a:cont_vp'):
                cols.remove(col_to_keep)
            data.drop(columns=cols, inplace=True)
            return data, state_scaler, action_scaler
        case 'peng_reward_fn':
            # drop columns
            columns_to_drop = ['traj', 'step', 'traj_length', 'm:presumed_onset', 'm:charttime', 'm:icustayid', 'a:action', 'a:cont_iv_fluid', 'a:cont_vp', 'r:reward',]
            data = data.drop(columns=columns_to_drop)
            assert data.shape[1] == 49, 'Data was not preprocessed correctly'  # 48 = num_feats (47) + split (1) + survived (1)
            # min-max scaling
            state_scaler = MinMaxScaler()
            scaler_mask = ((data.columns != 'split') & (data.columns != 'survived'))
            data.loc[:, scaler_mask] = state_scaler.fit_transform(data.loc[:, scaler_mask])
            return data, state_scaler, None
        case _:
            raise ValueError('Preprocess type not recognized')


def train_val_test_split_mimic_data(data: pd.DataFrame, split_type: Literal['sparse_auto_encoder', 'iql_discrete', 'kmeans_sarsa', 'ope', 'ddpg_cont', 'peng_reward_fn']) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor] | \
                                                                                                                       Tuple[
                                                                                                                            Tuple[torch.FloatTensor, torch.LongTensor, torch.FloatTensor, torch.FloatTensor, torch.BoolTensor],
                                                                                                                            Tuple[torch.FloatTensor, torch.LongTensor, torch.FloatTensor, torch.FloatTensor, torch.BoolTensor],
                                                                                                                            Tuple[torch.FloatTensor, torch.LongTensor, torch.FloatTensor, torch.FloatTensor, torch.BoolTensor]
                                                                                                                       ] | \
                                                                                                                       Tuple[
                                                                                                                            Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor, torch.FloatTensor,
                                                                                                                                  torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.BoolTensor,
                                                                                                                                  torch.BoolTensor],
                                                                                                                            Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor, torch.FloatTensor,
                                                                                                                                  torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.BoolTensor,
                                                                                                                                  torch.BoolTensor],
                                                                                                                            Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor, torch.FloatTensor,
                                                                                                                                  torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.BoolTensor,
                                                                                                                                  torch.BoolTensor],
                                                                                                                       ] | \
                                                                                                                       Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | \
                                                                                                                       Tuple[
                                                                                                                            Tuple[torch.FloatTensor, torch.LongTensor, torch.FloatTensor, torch.LongTensor],
                                                                                                                            Tuple[torch.FloatTensor, torch.LongTensor, torch.FloatTensor, torch.LongTensor],
                                                                                                                            Tuple[torch.FloatTensor, torch.LongTensor, torch.FloatTensor, torch.LongTensor]
                                                                                                                       ]:
    def extract_dataset(dataset_type: str, data_structure_type: Literal['np', 'torch']) -> Tuple[torch.FloatTensor, torch.LongTensor, torch.FloatTensor, torch.FloatTensor, torch.BoolTensor] | Tuple[NDArray[np.float32], NDArray[np.int64], NDArray[np.float32], NDArray[np.float32], NDArray[bool]]:
        data_subset = data[data['split'] == dataset_type]
        match data_structure_type:
            case 'np':
                actions: NDArray[np.int64]
                rewards: NDArray[np.float32]
                dones: NDArray[bool]
                states: NDArray[np.float32]
                next_states: NDArray[np.float32]
            case 'torch':
                actions: torch.LongTensor
                rewards: torch.FloatTensor
                dones: torch.BoolTensor
                states: torch.FloatTensor
                next_states: torch.FloatTensor
        if 'a:action' in data_subset.columns:
            assert 'a:cont_iv_fluid' not in data_subset.columns and 'a:cont_vp' not in data_subset.columns, 'Discrete and continuous actions are both present in the data'
            actions = data_subset['a:action'].to_numpy()
            discrete_actions = True
        else:
            assert 'a:cont_iv_fluid' in data_subset.columns and 'a:cont_vp' in data_subset.columns, 'Discrete and continuous actions are both missing from the data'
            actions_iv = data_subset['a:cont_iv_fluid'].to_numpy()
            actions_vp = data_subset['a:cont_vp'].to_numpy()
            actions = np.stack((actions_iv, actions_vp), axis=1).astype(np.float32) # NOTE: IV Fluids and Vasopressors Doses, respectively
            discrete_actions = False
        rewards = data_subset['r:reward'].to_numpy()
        dones = data_subset['done'].to_numpy()
        states = np.vstack(data_subset['state'].to_numpy()).astype(np.float32)
        next_states = data_subset['next_state'].to_numpy()
        for i in range(next_states.shape[0]):  # fill in None values with zeros
            if next_states[i] is None:
                next_states[i] = [0] * states.shape[1]
        next_states = np.vstack(next_states).astype(np.float32)
        if data_structure_type == 'torch':
            # noinspection PyTypeChecker
            actions = torch.from_numpy(actions)
            if discrete_actions:
                actions = actions.long()
            else:
                actions = actions.float()
            # noinspection PyTypeChecker
            rewards = torch.from_numpy(rewards).float()
            # noinspection PyTypeChecker
            dones = torch.from_numpy(dones).bool()
            # noinspection PyTypeChecker
            states = torch.from_numpy(states).float()
            # noinspection PyTypeChecker
            next_states = torch.from_numpy(next_states).float()
        return states, actions, next_states, rewards, dones
    match split_type:
        case 'sparse_auto_encoder':
            train_data_subset = data[data['split'] == 'train']
            train_data_subset = train_data_subset.drop(columns=['split'])
            # noinspection PyTypeChecker
            train_data: torch.FloatTensor = torch.from_numpy(train_data_subset.to_numpy()).float()
            val_data_subset = data[data['split'] == 'val']
            val_data_subset = val_data_subset.drop(columns=['split'])
            # noinspection PyTypeChecker
            val_data: torch.FloatTensor = torch.from_numpy(val_data_subset.to_numpy()).float()
            test_data_subset = data[data['split'] == 'test']
            test_data_subset = test_data_subset.drop(columns=['split'])
            # noinspection PyTypeChecker
            test_data: torch.FloatTensor = torch.from_numpy(test_data_subset.to_numpy()).float()
            return train_data, val_data, test_data
        case 'iql_discrete' | 'ddpg_cont':
            return (
                extract_dataset('train', 'torch'),
                extract_dataset('val', 'torch'),
                extract_dataset('test', 'torch')
            )
        case 'kmeans_sarsa':
            state_dim = len(data['state'].iloc[0])
            data.loc[:, 'next_state'] = data['next_state'].apply(lambda x: [0] * state_dim if x is None else x)
            split_dfs = tuple([data[data['split'] == split_type].drop(columns=['split']) for split_type in ['train', 'val', 'test']])
            # noinspection PyTypeChecker
            return split_dfs
        case 'ope':
            def extract_ope_dataset(dataset_type: str, df: pd.DataFrame) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor, torch.FloatTensor,
                                                                                  torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.BoolTensor,
                                                                                  torch.BoolTensor]:
                max_traj_len = df['traj_length'].max()
                data_subset = df[df['split'] == dataset_type]
                unique_traj_ids = data_subset['traj'].unique()
                state_dim = len(data_subset['raw_state'].iloc[0])
                raw_states = torch.zeros(unique_traj_ids.shape[0], max_traj_len, state_dim, dtype=torch.float)
                normed_states = torch.zeros_like(raw_states)
                discrete_actions = torch.zeros(unique_traj_ids.shape[0], max_traj_len, dtype=torch.long)
                continuous_actions = torch.zeros(unique_traj_ids.shape[0], max_traj_len, 2, dtype=torch.float)
                rewards = torch.zeros(unique_traj_ids.shape[0], max_traj_len, dtype=torch.float)
                raw_next_states = torch.zeros_like(raw_states)
                normed_next_states = torch.zeros_like(raw_states)
                dones = torch.zeros_like(rewards, dtype=torch.bool)
                missing_data_mask = torch.zeros_like(rewards, dtype=torch.bool)
                # iterate over unique traj ids
                for i, traj_id in tqdm(enumerate(unique_traj_ids), desc=f'Extracting {dataset_type} dataset', total=unique_traj_ids.shape[0], unit='traj', position=0, leave=True):
                    traj_view = data_subset[data_subset['traj'] == traj_id]
                    missing_data_mask[i, traj_view['traj_length'].values[0]:] = True
                    for _, row in traj_view.iterrows():
                        ts = row['step']
                        raw_states[i, ts] = torch.tensor(row['raw_state'], dtype=torch.float)
                        normed_states[i, ts] = torch.tensor(row['normed_state'], dtype=torch.float)
                        discrete_actions[i, ts] = row['a:action']
                        continuous_actions[i, ts, 0] = row['a:cont_iv_fluid']
                        continuous_actions[i, ts, 1] = row['a:cont_vp']
                        rewards[i, ts] = row['r:reward']
                        raw_next_states[i, ts] = torch.zeros(state_dim, dtype=torch.float) if row['raw_next_state'] is None else torch.tensor(row['raw_next_state'], dtype=torch.float)
                        normed_next_states[i, ts] = torch.zeros(state_dim, dtype=torch.float) if row['normed_next_state'] is None else torch.tensor(row['normed_next_state'], dtype=torch.float)
                        dones[i, ts] = row['done']
                return raw_states, normed_states, discrete_actions, continuous_actions, raw_next_states, normed_next_states, rewards, dones, missing_data_mask
            return (
                extract_ope_dataset('train', data),
                extract_ope_dataset('val', data),
                extract_ope_dataset('test', data)
            )
        case 'peng_reward_fn':
            def extract_peng_reward_fn_dataset(dataset_type: str, df: pd.DataFrame) -> Tuple[torch.FloatTensor, torch.LongTensor, torch.FloatTensor, torch.LongTensor]:
                data_subset = df[df['split'] == dataset_type]
                data_subset = data_subset.drop(columns=['split'])
                survived_mask = data_subset['survived']
                survived_data = torch.from_numpy(data_subset.loc[survived_mask].drop(columns=['survived']).to_numpy()).float()
                survived_labels = torch.zeros(survived_data.shape[0], dtype=torch.long) # label is mortality
                died_data = torch.from_numpy(data_subset.loc[~survived_mask].drop(columns=['survived']).to_numpy()).float()
                died_labels = torch.ones(died_data.shape[0], dtype=torch.long) # label is mortality
                # noinspection PyTypeChecker
                return survived_data, survived_labels, died_data, died_labels
            return (
                extract_peng_reward_fn_dataset('train', data),
                extract_peng_reward_fn_dataset('val', data),
                extract_peng_reward_fn_dataset('test', data)
            )
        case _:
            raise ValueError('Split type not recognized')


def get_cont_action_space_bounds(data: pd.DataFrame) -> Tuple[Tuple[Optional[float], Optional[float]], Tuple[Optional[float], Optional[float]]]:
    """
    Get the bounds of the continuous action space in the MIMIC-III dataset using all splits.
    :param data: MIMIC-III dataframe.
    :return: Tuples of the min and max values for the continuous IV fluid and vasopressor actions, respectively.
    """
    if 'a:cont_iv_fluid' in data.columns and 'a:cont_vp' in data.columns:
        min_cont_iv_fluid = data['a:cont_iv_fluid'].min()
        max_cont_iv_fluid = data['a:cont_iv_fluid'].max()
        min_cont_vp = data['a:cont_vp'].min()
        max_cont_vp = data['a:cont_vp'].max()
        return (min_cont_iv_fluid, min_cont_vp), (max_cont_iv_fluid, max_cont_vp)
    else:
        return (None, None), (None, None)



def encode_states(states: torch.FloatTensor, new_dim: int, state_space_ae: torch.nn.Module, batch_size: int, device: torch.device) -> torch.FloatTensor:
    encoded_states = torch.zeros(states.size(0), new_dim, device=device)
    states = states.to(device)
    for i in range(math.ceil(states.shape[0] / batch_size)):
        start_idx = i * batch_size
        stop_idx = min(start_idx + batch_size, states.shape[0])
        encoded_states[start_idx:stop_idx] = state_space_ae.encode(states[start_idx:stop_idx]).detach()
    return encoded_states.cpu()


class SparseAutoEncoderDataset(torch.utils.data.Dataset):
    def __init__(self, data: torch.FloatTensor):
        self.data = data

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx) -> torch.FloatTensor:
        # noinspection PyTypeChecker
        return self.data[idx]


class ImplicitQLearningDataset(torch.utils.data.Dataset):
    def __init__(self, states: torch.FloatTensor, actions: torch.LongTensor, next_states: torch.FloatTensor, rewards: torch.FloatTensor, dones: torch.BoolTensor):
        super().__init__()
        self._states = states
        self._actions = actions
        self._next_states = next_states
        self._rewards = rewards
        self._dones = dones

    def __len__(self) -> int:
        return self._states.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.FloatTensor, torch.LongTensor, torch.FloatTensor, torch.FloatTensor, torch.BoolTensor]:
        # noinspection PyTypeChecker
        return self._states[idx], self._actions[idx], self._next_states[idx], self._rewards[idx], self._dones[idx]


class PengRewardFnDataset(torch.utils.data.Dataset):
    def __init__(self, data: torch.FloatTensor, lbls: torch.LongTensor):
        super().__init__()
        self._data = data
        self._lbls = lbls

    def __len__(self) -> int:
        return self._data.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        # noinspection PyTypeChecker
        return self._data[idx], self._lbls[idx]


class PengRewardFnDataLoader:
    def __init__(self, survived_dataset: torch.utils.data.Dataset, died_dataset: torch.utils.data.Dataset, batch_size: int, shuffle: bool = True):
        if batch_size % 2 != 0:
            raise ValueError('Batch size must be even')
        self._survived_dataset = survived_dataset
        self._died_dataset = died_dataset
        self._half_batch_size = batch_size // 2
        self._shuffle = shuffle
        self._batches_per_epoch = math.ceil(max(len(survived_dataset), len(died_dataset)) / batch_size) # Note: survived dataset should be the largest
        self._remaining_died_indices = self._remaining_survived_indices = None
        self._died_reset = self._survived_reset = False

    def __len__(self) -> int:
        return self._batches_per_epoch

    def __iter__(self) -> 'PengRewardFnDataLoader':
        self._remaining_survived_indices = torch.arange(len(self._survived_dataset))
        self._remaining_died_indices = torch.arange(len(self._died_dataset))
        self._died_reset = self._survived_reset = False
        return self

    def __next__(self) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        if self._remaining_died_indices.size(0) == 0 or (self._remaining_died_indices.size(0) < self._remaining_survived_indices.size(0) and not self._survived_reset):
            self._died_reset = True
            num_needed = self._remaining_survived_indices.size(0) - self._remaining_died_indices.size(0)
            if self._shuffle:
                self._remaining_died_indices = torch.randperm(len(self._died_dataset))[:num_needed]
            else:
                self._remaining_died_indices = torch.arange(len(self._died_dataset))[:num_needed]
        if self._remaining_survived_indices.size(0) == 0 or (self._remaining_survived_indices.size(0) < self._remaining_died_indices.size(0) and not self._died_reset):
            self._survived_reset = True
            num_needed = self._remaining_died_indices.size(0) - self._remaining_survived_indices.size(0)
            if self._shuffle:
                self._remaining_survived_indices = torch.randperm(len(self._survived_dataset))[:num_needed]
            else:
                self._remaining_survived_indices = torch.arange(len(self._survived_dataset))[:num_needed]
        if self._died_reset and self._survived_reset:
            raise StopIteration
        # extract batch
        survived_data, survived_lbls = self._survived_dataset[self._remaining_survived_indices[:self._half_batch_size]]
        died_data, died_lbls = self._died_dataset[self._remaining_died_indices[:self._half_batch_size]]
        batch_data = torch.cat((survived_data, died_data), dim=0)
        batch_lbls = torch.cat((survived_lbls, died_lbls), dim=0)
        # update remaining indices
        self._remaining_survived_indices = self._remaining_survived_indices[self._half_batch_size:]
        self._remaining_died_indices = self._remaining_died_indices[self._half_batch_size:]
        # noinspection PyTypeChecker
        return batch_data, batch_lbls

