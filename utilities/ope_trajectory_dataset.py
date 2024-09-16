import torch
from typing import Tuple, Optional


class OPETrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self, behavior_traj_states: torch.FloatTensor, eval_traj_states: torch.FloatTensor, traj_discrete_actions: torch.LongTensor | torch.FloatTensor,
                 traj_cont_actions: torch.FloatTensor, behavior_traj_next_states: torch.FloatTensor, eval_traj_next_states: torch.FloatTensor, traj_rewards: torch.FloatTensor,
                 traj_dones: torch.BoolTensor, traj_missing_data_mask: torch.BoolTensor, flatten: bool = True):
        """
        Construct a dataset from a set of trajectories.
        Each tensor must have the same size in the first dimension, which is the number of trajectories.

        :param behavior_traj_states: The states of the trajectories for the behavior policy. Note that we assume these are discrete.
        :param eval_traj_states: The states of the trajectories for the evaluation policy.
        :param traj_discrete_actions: The actions of the trajectories.
        :param traj_cont_actions: The continuous actions of the trajectories.
        :param behavior_traj_next_states: The next states of the trajectories for the behavior policy. Note that we assume these are discrete.
        :param eval_traj_next_states: The next states of the trajectories for the evaluation policy.
        :param traj_rewards: The rewards of the trajectories.
        :param traj_dones: The dones of the trajectories.
        :param traj_missing_data_mask: The missing data mask of the trajectories.
        :param flatten: Whether to flatten the data or not.
        """
        super().__init__()
        self._states_behavior = behavior_traj_states
        self._states_eval = eval_traj_states
        self._discrete_actions = traj_discrete_actions
        self._continuous_actions = traj_cont_actions
        self._next_states_behavior = behavior_traj_next_states
        self._next_states_eval = eval_traj_next_states
        self._rewards = traj_rewards
        self._dones = traj_dones
        self._missing_data_mask = traj_missing_data_mask
        self._num_timesteps = self._states_behavior.size(1)
        self._initial_states_behavior = self._states_behavior[:, 0]
        self._initial_states_eval = self._states_eval[:, 0]
        self._flattened = flatten
        if flatten:
            self._states_behavior = self._states_behavior.view(-1, self._states_behavior.size(-1))
            self._states_eval = self._states_eval.view(-1, self._states_eval.size(-1))
            self._discrete_actions = self._discrete_actions.view(-1, 1)
            self._continuous_actions = self._continuous_actions.view(-1, self._continuous_actions.size(-1))
            self._rewards = self._rewards.view(-1, 1)
            self._next_states_behavior = self._next_states_behavior.view(-1, self._next_states_behavior.size(-1))
            self._next_states_eval = self._next_states_eval.view(-1, self._next_states_eval.size(-1))
            self._dones = self._dones.view(-1, 1)
            self._missing_data_mask = self._missing_data_mask.view(-1, 1)
        else:
            self._states_behavior = self._states_behavior.view(-1, self._num_timesteps, self._states_behavior.size(-1))
            self._states_eval = self._states_eval.view(-1, self._num_timesteps, self._states_eval.size(-1))
            self._discrete_actions = self._discrete_actions.view(-1, self._num_timesteps, 1)
            self._continuous_actions = self._continuous_actions.view(-1, self._num_timesteps, self._continuous_actions.size(-1))
            self._rewards = self._rewards.view(-1, self._num_timesteps, 1)
            self._next_states_behavior = self._next_states_behavior.view(-1, self._num_timesteps, self._next_states_behavior.size(-1))
            self._next_states_eval = self._next_states_eval.view(-1, self._num_timesteps, self._next_states_eval.size(-1))
            self._dones = self._dones.view(-1, self._num_timesteps, 1)
            self._missing_data_mask = self._missing_data_mask.view(-1, self._num_timesteps, 1)

    # region Dataset

    def __len__(self) -> int:
        return self._states_behavior.size(0)

    def __getitem__(self, item: int) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor, torch.FloatTensor,
                                              torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.BoolTensor,
                                              torch.BoolTensor]:
        states_behavior = self._states_behavior[item]
        states_eval = self._states_eval[item]
        discrete_actions = self._discrete_actions[item]
        continuous_actions = self._continuous_actions[item]
        rewards = self._rewards[item]
        next_states_behavior = self._next_states_behavior[item]
        next_states_eval = self._next_states_eval[item]
        dones = self._dones[item]
        missing_data_mask = self._missing_data_mask[item]
        # noinspection PyTypeChecker
        return states_behavior, states_eval, discrete_actions, continuous_actions, rewards, next_states_behavior, next_states_eval, dones, missing_data_mask

    # endregion

    # region Functions/Methods

    # region Public

    def get_unique_states(self, extract_using_behavior_states: bool) -> Tuple[Optional[torch.FloatTensor], torch.FloatTensor]:
        if self._states_behavior.dim() == 3:
            behavior_states_view = self._states_behavior.view(-1, self._states_behavior.size(-1))
            eval_states_view = self._states_eval.view(-1, self._states_eval.size(-1))
        else:
            behavior_states_view = self._states_behavior
            eval_states_view = self._states_eval
        if extract_using_behavior_states:
            # extract the indices of the unique states so we can get the unique states for both behavior and eval policies
            # copied from janblumenkamp's comment: https://github.com/pytorch/pytorch/issues/36748
            uniques, inverse = behavior_states_view.unique(dim=0, sorted=True, return_inverse=True)
            perm = torch.arange(inverse.size(0), dtype=torch.long)
            state_indices = perm.new_empty(uniques.size(0)).scatter_(0, inverse.flip(0), perm.flip(0))
            # extract states
            unique_behavior_states = behavior_states_view[state_indices]
            unique_eval_states = eval_states_view[state_indices]
            return unique_behavior_states, unique_eval_states
        else:
            uniques = eval_states_view.unique(dim=0)
            return None, uniques

    def get_initial_states(self) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        return self._initial_states_behavior, self._initial_states_eval

    def reshape_data(self, flatten: bool):
        if flatten and self._states_behavior.dim() == 3:
            self._states_behavior = self._states_behavior.view(-1, self._states_behavior.size(-1))
            self._states_eval = self._states_eval.view(-1, self._states_eval.size(-1))
            self._discrete_actions = self._discrete_actions.view(-1, 1)
            self._continuous_actions = self._continuous_actions.view(-1, self._continuous_actions.size(-1))
            self._rewards = self._rewards.view(-1, 1)
            self._next_states_behavior = self._next_states_behavior.view(-1, self._next_states_behavior.size(-1))
            self._next_states_eval = self._next_states_eval.view(-1, self._next_states_eval.size(-1))
            self._dones = self._dones.view(-1, 1)
            self._missing_data_mask = self._missing_data_mask.view(-1, 1)
        elif not flatten and self._states_behavior.dim() == 2:
            self._states_behavior = self._states_behavior.view(-1, self._num_timesteps, self._states_behavior.size(-1))
            self._states_eval = self._states_eval.view(-1, self._num_timesteps, self._states_eval.size(-1))
            self._discrete_actions = self._discrete_actions.view(-1, self._num_timesteps, self._discrete_actions.size(-1))
            self._continuous_actions = self._continuous_actions.view(-1, self._num_timesteps, self._continuous_actions.size(-1))
            self._rewards = self._rewards.view(-1, self._num_timesteps, self._rewards.size(-1))
            self._next_states_behavior = self._next_states_behavior.view(-1, self._num_timesteps, self._next_states_behavior.size(-1))
            self._next_states_eval = self._next_states_eval.view(-1, self._num_timesteps, self._next_states_eval.size(-1))
            self._dones = self._dones.view(-1, self._num_timesteps, self._dones.size(-1))
            self._missing_data_mask = self._missing_data_mask.view(-1, self._num_timesteps, self._missing_data_mask.size(-1))
        self._flattened = flatten

    # endregion

    # region Private

    # endregion

    # endregion

    # region Properties

    @property
    def behavior_state_dim(self) -> int:
        return self._states_behavior.size(-1)

    @property
    def eval_state_dim(self) -> int:
        return self._states_eval.size(-1)

    @property
    def action_dim(self) -> int:
        return self._discrete_actions.size(-1)

    @property
    def num_actions(self) -> int:
        if self._discrete_actions.dtype == torch.long or self._discrete_actions.dtype == torch.int:
            return self._discrete_actions.max().item() + 1
        else:
            raise ValueError('Action space is continuous.')

    @property
    def num_time_steps(self) -> int:
        return self._num_timesteps

    @property
    def is_flattened(self) -> bool:
        return self._flattened

    @is_flattened.setter
    def is_flattened(self, value: bool):
        self.reshape_data(value)

    # endregion
