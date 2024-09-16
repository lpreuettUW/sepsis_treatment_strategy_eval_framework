import torch
from typing import Tuple

from ope.abstract_ope_method import AbstractOPEMethod
from ope.fqe import FittedQEvaluation
from agents.abstract_agent import AbstractAgent


class WeightedDoublyRobust(AbstractOPEMethod):
    def __init__(self, trajectory_dataset: torch.utils.data.Dataset, agent: AbstractAgent, gamma: float, batch_size: int, fqe: FittedQEvaluation):
        self._gamma = gamma
        self._batch_size = batch_size
        self._trajectory_dataset_data_loader = torch.utils.data.DataLoader(trajectory_dataset, batch_size=self._batch_size, shuffle=False) # Not shuffle has yielded weird results..
        self._fqe = fqe
        self._behavior_policy_action_probs = self._eval_policy_action_probs = self._importance_weight_denominators = None
        super().__init__(trajectory_dataset, agent)

    # region AbstractOPEMethod

    def _initialize(self):
        # get unique states for indexing
        unique_states = self._trajectory_dataset.get_unique_states()
        # get behavior policy action probabilities
        self._compute_behavior_policy_action_probs(unique_states)
        # get evaluation policy action probabilities
        self._eval_policy_action_probs = self._agent.get_action_probs(unique_states)
        # compute importance weight denominators
        self._compute_importance_weight_denominators(unique_states)

    def compute_value(self) -> float:
        # get unique states for indexing
        unique_states = self._trajectory_dataset.get_unique_states().to(self._device)
        unique_states_rep = unique_states.unsqueeze(1).repeat(1, self._batch_size * self._trajectory_dataset.num_time_steps, 1).long()
        gammas = torch.logspace(0, self._trajectory_dataset.num_time_steps - 1, self._trajectory_dataset.num_time_steps, base=self._gamma, device=self._device).unsqueeze(-1)
        importance_weighted_component = torch.zeros(1, device=self._device)
        approximate_model_component = torch.zeros(1, device=self._device)
        # reshape trajectory dataset to return entire trajectories
        self._trajectory_dataset.reshape_data(flatten=False)
        for traj_states, traj_actions, traj_rewards, _, traj_dones in self._trajectory_dataset_data_loader:
            traj_states, traj_actions, traj_rewards, traj_dones = traj_states.long().to(self._device), traj_actions.to(self._device), traj_rewards.to(self._device), traj_dones.to(self._device)
            # compute importance weighted component: sum gamma^t * w_t * r_t for all timesteps, for all trajectories
            behavior_action_probs, eval_action_probs = self._get_state_action_probs(traj_states, traj_actions, unique_states_rep)
            timestep_weights = self._compute_timestep_weights(behavior_action_probs, eval_action_probs)
            # NOTE: we are duplicating the first time step weight - this differs from the implementation here where they use 1 / |Dataset|
            # See https://github.com/clvoloshin/COBS/blob/master/ope/algos/doubly_robust_v2.py#L71
            prev_timestep_weights = torch.cat([
                timestep_weights[:, 0].unsqueeze(-1),
                timestep_weights[:, :-1]
            ], dim=1)
            importance_weighted_component += (gammas * timestep_weights * traj_rewards).sum()
            # compute approximate model component: sum gamma^t * (w_t * Q(s_t, a_t) - w_(t-1) * V(s_t)) for all timesteps, for all trajectories
            traj_q_vals, traj_vals = self._fqe.compute_q_values_and_values(traj_states, traj_actions)
            weighted_traj_q_vals = timestep_weights * traj_q_vals
            weighted_traj_vals = prev_timestep_weights * traj_vals
            approximate_model_component += (gammas * (weighted_traj_q_vals - weighted_traj_vals)).sum()
        final_value = importance_weighted_component - approximate_model_component
        return final_value.cpu().item()

    # endregion

    def _compute_behavior_policy_action_probs(self, unique_states: torch.LongTensor):
        # ensure trajectory dataset is in flattened form
        self._trajectory_dataset.reshape_data(flatten=True)
        # get repeated unique states for indexing
        unique_states_rep = unique_states.unsqueeze(1).repeat(1, self._batch_size, 1).long()
        # get behavior policy action probabilities
        self._behavior_policy_action_probs = torch.zeros(unique_states.size(0), self._trajectory_dataset.num_actions)
        behavior_policy_state_counts = torch.zeros(unique_states.size(0))
        for states, actions, _, _, _ in self._trajectory_dataset_data_loader:
            states, actions = states.long().to(self._device), actions.to(self._device)
            states_rep = states.unsqueeze(0).repeat(unique_states_rep.size(0), 1, 1)
            # gives us index of unique states paired with index of batch states: output: (batch_size, 2)
            state_indices = (states_rep == unique_states_rep).all(dim=-1).nonzero()
            assert state_indices.size(0) == self._batch_size, 'something went wrong with the state indices...'
            state_indices = state_indices[:, 0]  # cut out batch index dimension
            index_action_pairs = torch.cat([state_indices.unsqueeze(-1), actions], dim=-1)
            unique_index_action_pairs, counts = index_action_pairs.unique(dim=0, return_counts=True)
            self._behavior_policy_action_probs[unique_index_action_pairs[:, 0], unique_index_action_pairs[:, 1]] += counts
            state_indices, counts = state_indices.unique(return_counts=True)
            behavior_policy_state_counts[state_indices] += counts
        behavior_policy_state_counts = behavior_policy_state_counts.unsqueeze(-1).repeat(1, self._behavior_policy_action_probs.size(-1))
        self._behavior_policy_action_probs /= behavior_policy_state_counts

    def _compute_importance_weight_denominators(self, unique_states: torch.LongTensor):
        """ Compute sum of importance weights for each trajectory up to each possible timestep.
        The result is a vector of length num_timesteps
        SUM (pi_e(a | s) / pi_b(a | s)) for all timesteps for all trajectories
        """
        # ensure trajectory dataset is in trajectory form
        self._trajectory_dataset.reshape_data(flatten=False)
        # get repeated unique states for indexing
        unique_states_rep = unique_states.unsqueeze(1).repeat(1, self._batch_size * self._trajectory_dataset.num_time_steps, 1).long()
        # importance weight denominators
        self._importance_weight_denominators = torch.zeros(self._trajectory_dataset.num_time_steps, 1)
        for traj_states, traj_actions, _, _, _ in self._trajectory_dataset_data_loader:
            traj_states, traj_actions = traj_states.long().to(self._device), traj_actions.to(self._device)
            behavior_policy_action_probs, eval_policy_action_probs = self._get_state_action_probs(traj_states, traj_actions, unique_states_rep)
            step_rhos = eval_policy_action_probs / behavior_policy_action_probs # individual timestep importance weights
            # final rho computation for each batch
            batch_rho = step_rhos.cumprod(dim=1)
            # sum rhos for each trajectory
            batch_rho_sum = batch_rho.sum(dim=0)
            # update importance weight denominators
            self._importance_weight_denominators += batch_rho_sum

    def _get_state_action_probs(self, states: torch.LongTensor, actions: torch.LongTensor, unique_states_rep: torch.LongTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """ Extract behavior and evaluation policy action probabilities for given state and action pairs.
        :param states: Batch of state trajectories: (batch, num_steps, state_dim).
        :param actions: Batch of action trajectories: (batch, num_steps, action_dim).
        :param unique_states_rep: Unique states repeated batch_size by num_time_steps times: (num_unique_states, batch_size x num_time_steps, state_dim).
        :return: Tuple of behavior and evaluation policy action probabilities for given state and action pairs, respectively.
        """
        assert states.size()[:-1] == actions.size()[:-1], 'states and actions must have same batch size and num steps'
        # OPTIMIZE: just replace states and actions - we are doing it like this to ensure we dont mess up the state/action ordering
        individual_action_view = actions.view(-1, actions.size(-1))
        individual_states_view = states.view(-1, states.size(-1))
        individual_states_rep = individual_states_view.unsqueeze(0).repeat(unique_states_rep.size(0), 1, 1)
        state_indices = (individual_states_rep == unique_states_rep).all(dim=-1).nonzero()
        assert state_indices.size(0) == self._batch_size * self._trajectory_dataset.num_time_steps, 'something went wrong with the state indices...'
        # now we need to reorder the state indices to match the batch state/action ordering
        state_indices_sort_indices = state_indices[:, 1].argsort()
        state_indices = state_indices[state_indices_sort_indices]
        assert (state_indices[:, 1] == torch.arange(self._batch_size * self._trajectory_dataset.num_time_steps)).all(), 'state indices were not sorted correctly'
        # cut out batch index dimension
        state_indices = state_indices[:, 0].unsqueeze(-1)
        # extract probs
        behavior_policy_action_probs = self._behavior_policy_action_probs[state_indices, individual_action_view]
        eval_policy_action_probs = self._eval_policy_action_probs[state_indices, individual_action_view]
        # reshape probs to match original dimensions: (batch, num_steps, 1)
        behavior_policy_action_probs = behavior_policy_action_probs.reshape(states.size(0), states.size(1), 1)
        eval_policy_action_probs = eval_policy_action_probs.reshape(states.size(0), states.size(1), 1)
        states_reconstruct = individual_states_view.clone().reshape(states.size(0), states.size(1), states.size(-1))
        assert (states == states_reconstruct).all(), 'states were not reconstructed correctly'
        actions_reconstruct = individual_action_view.clone().reshape(states.size(0), states.size(1), actions.size(-1))
        assert (actions == actions_reconstruct).all(), 'actions were not reconstructed correctly'
        return behavior_policy_action_probs, eval_policy_action_probs

    def _compute_timestep_weights(self, behavior_policy_action_probs: torch.FloatTensor, eval_policy_action_probs: torch.FloatTensor) -> torch.FloatTensor:
        batch_rho = (eval_policy_action_probs / behavior_policy_action_probs).cumprod(dim=1)
        final_weights = batch_rho / self._importance_weight_denominators
        return final_weights
