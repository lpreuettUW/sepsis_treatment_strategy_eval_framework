import math
import tqdm
import torch
import scipy
import mlflow
import tracemalloc
import numpy as np
import warnings
from typing import Tuple, Set, Optional, List

from ope.abstract_ope_method import AbstractOPEMethod
from ope.fqe import FittedQEvaluation
from agents.abstract_agent import AbstractAgent
from agents.abstract_sequence_agent import AbstractSequenceAgent
from utilities.sequence_agent_interface import SequenceAgentInterface


class MAGIC(AbstractOPEMethod):
    """Adapted from https://github.com/clvoloshin/COBS/blob/master/ope/algos/magic.py"""
    def __init__(self, trajectory_dataset: torch.utils.data.Dataset, agent: AbstractAgent, gamma: float, batch_size: int, fqe: FittedQEvaluation, j_steps: Set[int | float],
                 confidence_estimation_iters_k: int, wdr_eps: float = 1e-6, discrete_state_space: bool = False):
        if float('inf') not in j_steps:
            raise ValueError('j_steps must contain infinity')
        if confidence_estimation_iters_k < 1:
            raise ValueError('confidence_estimation_iters_k must be greater than 0')
        elif confidence_estimation_iters_k < 200:
            warnings.warn(f'confidence_estimation_iters_k ({confidence_estimation_iters_k}) < 200. This may result in a high variance estimate of the confidence interval. The authors recommend this value be as large as possible.')
        self._gamma = gamma
        self._batch_size = batch_size
        self._trajectory_dataset_data_loader = torch.utils.data.DataLoader(trajectory_dataset, batch_size=self._batch_size, shuffle=False) # Not shuffle has yielded weird results..
        self._fqe = fqe
        self._j_steps = list(j_steps) # we want this indexable
        self._confidence_estimation_iters_k = confidence_estimation_iters_k
        self._wdr_eps = wdr_eps
        self._discrete_state_space = discrete_state_space
        self._behavior_policy_action_probs = self._eval_policy_action_probs = self._importance_weight_denominators = self._j_step_weights = None
        self._i = 0
        super().__init__(trajectory_dataset, agent)

    # region AbstractOPEMethod

    def _initialize(self):
        self._agent.eval()
        if isinstance(self._agent, AbstractSequenceAgent):
            self._agent.reset_eval_context()
        # get unique states for indexing
        unique_behavior_states, unique_eval_states = self._trajectory_dataset.get_unique_states(extract_using_behavior_states=True)
        # get behavior policy action probabilities
        self._compute_behavior_policy_action_probs(unique_behavior_states, compute_eval_policy_action_probs=not self._discrete_state_space)
        if self._discrete_state_space:
            # get evaluation policy action probabilities
            self._eval_policy_action_probs = self._compute_evaluation_policy_action_probs(unique_behavior_states.to(self._device))
        # compute importance weight denominators
        self._compute_importance_weight_denominators(unique_behavior_states)

    def reinitialize(self, agent: AbstractAgent):
        self._agent = agent
        self._agent.eval()
        self._fqe.reinitialize(agent)
        unique_behavior_states, _ = self._trajectory_dataset.get_unique_states(extract_using_behavior_states=True)
        if self._discrete_state_space:
            self._eval_policy_action_probs = self._compute_evaluation_policy_action_probs(unique_behavior_states.to(self._device))
        else:
            self._compute_behavior_policy_action_probs(unique_behavior_states.to(self._device), compute_eval_policy_action_probs=True)
        self._compute_importance_weight_denominators(unique_behavior_states)
        self._j_step_weights = None

    def compute_value(self) -> float:
        # get unique states for indexing
        unique_behavior_states, _ = self._trajectory_dataset.get_unique_states(extract_using_behavior_states=True)
        unique_behavior_states = unique_behavior_states.to(self._device)
        unique_states_rep = unique_behavior_states.unsqueeze(1).repeat(1, self._batch_size * self._trajectory_dataset.num_time_steps, 1).long()
        gammas = torch.logspace(0, self._trajectory_dataset.num_time_steps - 1, self._trajectory_dataset.num_time_steps, base=self._gamma, device=self._device).unsqueeze(-1)
        # reshape trajectory dataset to return entire trajectories
        self._trajectory_dataset.reshape_data(flatten=False)
        traj_j_step_returns, traj_inf_step_returns, traj_control_variates = self._compute_j_step_returns(unique_states_rep, gammas)
        j_step_returns = traj_j_step_returns.sum(dim=0) # sum along the trajectory dimension
        # we are interested in the control variates to measure expected variance of MAGIC
        conf_lower, conf_upper = self._compute_confidence_interval(unique_states_rep, gammas, j_step_returns, traj_inf_step_returns)
        self._j_step_weights = self._compute_j_step_weights(j_step_returns, traj_j_step_returns, conf_lower, conf_upper)
        # compute final value
        final_value = j_step_returns.dot(self._j_step_weights)
        self._j_step_weights = self._j_step_weights.cpu().tolist()
        return final_value.cpu().item(), traj_control_variates.cpu()

    # endregion

    def _compute_evaluation_policy_action_probs(self, unique_states: torch.LongTensor):
        if isinstance(self._agent, AbstractSequenceAgent):
            raise NotImplementedError('Sequence agents are not yet supported')
            # self._eval_policy_action_probs = self._agent.get_action_probs(unique_states.to(self._device), torch.ones(unique_states.size(0), dtype=torch.bool, device=self._device))
        else:
            self._eval_policy_action_probs = self._agent.get_action_probs(unique_states.to(self._device))

    def _compute_behavior_policy_action_probs(self, unique_states: torch.LongTensor, compute_eval_policy_action_probs: bool):
        # ensure trajectory dataset is in flattened form
        self._trajectory_dataset.reshape_data(flatten=True)
        # get repeated unique states for indexing
        unique_states_rep = unique_states.unsqueeze(1).repeat(1, self._batch_size, 1).long()
        # get behavior policy action probabilities
        self._behavior_policy_action_probs = torch.zeros(unique_states.size(0), self._trajectory_dataset.num_actions, device=self._device)
        if compute_eval_policy_action_probs:
            self._eval_policy_action_probs = torch.zeros(unique_states.size(0), self._trajectory_dataset.num_actions, device=self._device)
        behavior_policy_state_counts = torch.full((unique_states.size(0),), self._wdr_eps, device=self._device) #torch.zeros(unique_states.size(0), device=self._device)
        for states_b, states_e, discrete_actions_b, _, _, _, _, _, missing_data_mask in tqdm.tqdm(self._trajectory_dataset_data_loader, desc='Computing Policy Action Probs', total=len(self._trajectory_dataset_data_loader), unit='batch', position=0, leave=True):
            valid_mask = missing_data_mask.squeeze(-1).logical_not()
            if valid_mask.any():
                if valid_mask.size(0) < unique_states_rep.size(1):
                    masked_unique_states_rep = unique_states_rep[:, :states_b.size(0)][:, valid_mask].to(self._device)
                else:
                    masked_unique_states_rep = unique_states_rep[:, valid_mask].to(self._device)
                states_b, discrete_actions_b = states_b[valid_mask].long().to(self._device), discrete_actions_b[valid_mask].to(self._device)
                states_rep = states_b.unsqueeze(0).repeat(masked_unique_states_rep.size(0), 1, 1)
                # gives us index of unique states paired with index of batch states: output: (batch_size, 2)
                state_indices = (states_rep == masked_unique_states_rep).all(dim=-1).nonzero()
                state_indices_sort_indices = state_indices[:, 1].argsort()
                state_indices = state_indices[state_indices_sort_indices]
                assert (state_indices[:, 1] == torch.arange(states_b.size(0), device=self._device)).all(), 'state indices were not sorted correctly'  # self._batch_size * self._trajectory_dataset.num_time_steps
                assert state_indices.size(0) == valid_mask.sum(), 'something went wrong with the state indices...'  # self._batch_size
                state_indices = state_indices[:, 0]  # cut out batch index dimension
                # get behavior unique state action pairs
                index_action_pairs = torch.cat([state_indices.unsqueeze(-1), discrete_actions_b], dim=-1)
                unique_index_action_pairs, counts = index_action_pairs.unique(dim=0, return_counts=True)
                self._behavior_policy_action_probs[unique_index_action_pairs[:, 0], unique_index_action_pairs[:, 1]] += counts
                if compute_eval_policy_action_probs:
                    # get eval actions
                    states_e = states_e[valid_mask].to(self._device)
                    actions_e, _ = self._agent.get_best_action(states_e)
                    # get eval unique state action pairs
                    index_action_pairs = torch.cat([state_indices.unsqueeze(-1), actions_e.action.unsqueeze(-1)], dim=-1)
                    unique_index_action_pairs, counts = index_action_pairs.unique(dim=0, return_counts=True)
                    self._eval_policy_action_probs[unique_index_action_pairs[:, 0], unique_index_action_pairs[:, 1]] += counts
                # update state counts
                state_indices, counts = state_indices.unique(return_counts=True)
                behavior_policy_state_counts[state_indices] += counts
        behavior_policy_state_counts = behavior_policy_state_counts.unsqueeze(-1).repeat(1, self._behavior_policy_action_probs.size(-1))
        self._behavior_policy_action_probs /= behavior_policy_state_counts
        if compute_eval_policy_action_probs:
            self._eval_policy_action_probs /= behavior_policy_state_counts

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
        self._importance_weight_denominators = torch.zeros(self._trajectory_dataset.num_time_steps, 1, device=self._device)
        for traj_states, _, traj_discrete_actions, _, _, _, _, _, traj_missing_data_mask in self._trajectory_dataset_data_loader:
            unique_states_rep_batch_slice = unique_states_rep[:, :traj_states.size(0) * self._trajectory_dataset.num_time_steps].to(self._device)
            traj_states, traj_discrete_actions = traj_states.long().to(self._device), traj_discrete_actions.to(self._device)
            behavior_policy_action_probs, eval_policy_action_probs = self._get_state_action_probs(traj_states, traj_discrete_actions, ~traj_missing_data_mask, unique_states_rep_batch_slice)
            step_rhos = eval_policy_action_probs / behavior_policy_action_probs # individual timestep importance weights
            # final rho computation for each batch
            batch_rho = step_rhos.cumprod(dim=1)
            # sum rhos for each trajectory
            batch_rho_sum = batch_rho.sum(dim=0)
            # update importance weight denominators
            self._importance_weight_denominators += batch_rho_sum
        # ensure we have no zeros
        zeros_mask = self._importance_weight_denominators.isclose(torch.tensor(0.0, device=self._device))
        self._importance_weight_denominators[zeros_mask] = self._wdr_eps

    def _get_state_action_probs(self, states: torch.LongTensor, actions: torch.LongTensor, valid_data_mask: torch.BoolTensor, unique_states_rep: torch.LongTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """ Extract behavior and evaluation policy action probabilities for given state and action pairs.
        :param states: Batch of state trajectories: (batch, num_steps, state_dim).
        :param actions: Batch of action trajectories: (batch, num_steps, action_dim).
        :param valid_data_mask: Mask indicating which data is valid (batch, nums_steps, 1).
        :param unique_states_rep: Unique states repeated batch_size by num_time_steps times: (num_unique_states, batch_size x num_time_steps, state_dim).
        :return: Tuple of behavior and evaluation policy action probabilities for given state and action pairs, respectively.
        """
        assert states.size()[:-1] == actions.size()[:-1], 'states and actions must have same batch size and num steps'
        # OPTIMIZE: just replace states and actions - we are doing it like this to ensure we dont mess up the state/action ordering
        individual_valid_data_mask = valid_data_mask.view(-1)
        individual_action_view = actions.view(-1, actions.size(-1))[individual_valid_data_mask]
        individual_states_view = states.view(-1, states.size(-1))[individual_valid_data_mask]
        individual_states_rep = individual_states_view.unsqueeze(0).repeat(unique_states_rep.size(0), 1, 1)
        masked_unique_states_rep = unique_states_rep[:, :individual_valid_data_mask.sum()]
        state_indices = (individual_states_rep == masked_unique_states_rep).all(dim=-1).nonzero()
        assert state_indices.size(0) == valid_data_mask.sum(), 'something went wrong with the state indices...' # self._batch_size * self._trajectory_dataset.num_time_steps
        # now we need to reorder the state indices to match the batch state/action ordering
        state_indices_sort_indices = state_indices[:, 1].argsort()
        state_indices = state_indices[state_indices_sort_indices]
        assert (state_indices[:, 1] == torch.arange(valid_data_mask.sum(), device=self._device)).all(), 'state indices were not sorted correctly' # self._batch_size * self._trajectory_dataset.num_time_steps
        # cut out batch index dimension
        state_indices = state_indices[:, 0].unsqueeze(-1)
        # extract probs
        behavior_policy_action_probs = self._behavior_policy_action_probs[state_indices, individual_action_view]
        eval_policy_action_probs = self._eval_policy_action_probs[state_indices, individual_action_view]
        # reshape probs to match original dimensions: (batch, num_steps, 1)
        final_behavior_policy_action_probs = torch.full((individual_valid_data_mask.size(0), 1), 1, dtype=torch.float32, device=self._device) # NOTE: we intentionally use 1 as the filler value here
        final_behavior_policy_action_probs[individual_valid_data_mask] = behavior_policy_action_probs
        final_behavior_policy_action_probs = final_behavior_policy_action_probs.reshape(states.size(0), states.size(1), 1)
        final_eval_policy_action_probs = torch.full((individual_valid_data_mask.size(0), 1), 1, dtype=torch.float32, device=self._device) # NOTE: we intentionally use 1 as the filler value here
        final_eval_policy_action_probs[individual_valid_data_mask] = eval_policy_action_probs
        final_eval_policy_action_probs = final_eval_policy_action_probs.reshape(states.size(0), states.size(1), 1)
        # states_reconstruct = individual_states_view.clone().reshape(states.size(0), states.size(1), states.size(-1))
        # assert (states == states_reconstruct).all(), 'states were not reconstructed correctly'
        # actions_reconstruct = individual_action_view.clone().reshape(states.size(0), states.size(1), actions.size(-1))
        # assert (actions == actions_reconstruct).all(), 'actions were not reconstructed correctly'
        return final_behavior_policy_action_probs, final_eval_policy_action_probs

    def _compute_timestep_weights(self, behavior_policy_action_probs: torch.FloatTensor, eval_policy_action_probs: torch.FloatTensor) -> torch.FloatTensor:
        batch_rho = (eval_policy_action_probs / behavior_policy_action_probs).cumprod(dim=1)
        final_weights = batch_rho / self._importance_weight_denominators
        return final_weights

    def _compute_j_step_returns(self, unique_states_rep: torch.LongTensor, gammas: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """ Compute j-step returns for all trajectories.
        Returns a tensor with each j-step return for each trajectory.
        Trajectories are ordered by index in the dataset.
        :param unique_states_rep: Unique states repeated batch_size by num_time_steps times: (num_unique_states, batch_size x num_time_steps, state_dim).
        :param gammas: Batch of discount factors: (num_steps, 1).
        :return: Trajectory j-step return values (|D|, |j_steps|).
        """
        j_step_returns = torch.zeros(len(self._trajectory_dataset), len(self._j_steps), device=self._device)
        inf_step_returns = torch.zeros(len(self._trajectory_dataset), 1, device=self._device)
        control_variates = torch.zeros(len(self._trajectory_dataset), len(self._j_steps), device=self._device)
        traj_batch_size = max(1, math.ceil(self._batch_size / self._trajectory_dataset.num_time_steps)) # scale the batch size by the number of timesteps because we are iterating over trajectories now (so we get a more responsive progress bar)
        start_stop_idx_pairs = [(i * traj_batch_size, min((i + 1) * traj_batch_size, len(self._trajectory_dataset))) for i in range(math.ceil(len(self._trajectory_dataset) / traj_batch_size))]
        for start_idx, stop_idx in tqdm.tqdm(start_stop_idx_pairs, desc='Computing J-Step Returns', total=len(start_stop_idx_pairs), unit='batch', position=0, leave=True):
            traj_states, _, traj_discrete_actions, _, traj_rewards, _, _, traj_dones, traj_missing_data_mask = self._trajectory_dataset[start_idx:stop_idx]
            traj_states, traj_discrete_actions, traj_rewards, traj_dones, traj_valid_data_mask = traj_states.long().to(self._device), traj_discrete_actions.to(self._device), traj_rewards.to(self._device), traj_dones.to(self._device), traj_missing_data_mask.logical_not().to(self._device)
            if traj_valid_data_mask.any():
                # compute importance weighted component: sum gamma^t * w_t * r_t for all timesteps, for all trajectories
                behavior_action_probs, eval_action_probs = self._get_state_action_probs(traj_states, traj_discrete_actions, traj_valid_data_mask, unique_states_rep)
                timestep_weights = self._compute_timestep_weights(behavior_action_probs, eval_action_probs)
                # TODO: fix prev_timestep_weights - this is not correct - the paper assumes rho_-1 = 1. So we should have 1 / (sum rho_t for all trajectories)
                # first_weight = torch.ones(traj_states.size(0), 1, device=self._device) / len(self._trajectory_dataset)
                # NOTE: we are duplicating the first time step weight - this differs from the implementation here where they use 1 / |Dataset|
                # See https://github.com/clvoloshin/COBS/blob/master/ope/algos/doubly_robust_v2.py#L71
                prev_timestep_weights = torch.cat([
                    timestep_weights[:, 0].unsqueeze(-1), # first_weight,
                    timestep_weights[:, :-1]
                ], dim=1)
                j_step_idx = 0
                for j_step in self._j_steps:
                    batch_j_step_return, batch_control_variate = self._compute_j_step_return(traj_states, traj_discrete_actions, traj_rewards, traj_valid_data_mask, timestep_weights, prev_timestep_weights, gammas, j_step)  # type: ignore
                    j_step_returns[start_idx:stop_idx, j_step_idx] = batch_j_step_return.squeeze(-1)
                    if batch_control_variate is not None:
                        control_variates[start_idx:stop_idx, j_step_idx] = batch_control_variate.squeeze(-1)
                    if j_step == float('inf'):
                        inf_step_returns[start_idx:stop_idx] = j_step_returns[start_idx:stop_idx, j_step_idx].unsqueeze(-1)
                    j_step_idx += 1
        return j_step_returns, inf_step_returns, control_variates

    def _compute_j_step_return(self, traj_states: torch.LongTensor, traj_actions: torch.LongTensor, traj_rewards: torch.FloatTensor, traj_valid_data_mask: torch.BoolTensor,
                               timestep_weights: torch.FloatTensor, prev_timestep_weights: torch.FloatTensor, gammas: torch.FloatTensor, j_step: int | float) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        """ Compute j-step return for a given trajectory batch.
            :param traj_states: Batch of state trajectories: (batch, num_steps, state_dim).
            :param traj_actions: Batch of action trajectories: (batch, num_steps, action_dim).
            :param traj_rewards: Batch of reward trajectories: (batch, num_steps, 1).
            :param traj_valid_data_mask: Batch of valid data masks: (batch, num_steps, 1).
            :param timestep_weights: Batch of timestep weights: (batch, num_steps, 1).
            :param prev_timestep_weights: Batch of previous timestep weights: (batch, num_steps, 1).
            :param gammas: Batch of discount factors: (num_steps, 1).
            :param j_step: The j-step to compute the return for.
            :return: Trajectory j-step return values (batch, 1).
        """
        # print('memory at start of compute_j_step_return', tracemalloc.get_traced_memory())
        if isinstance(self._agent, AbstractSequenceAgent):
            self._agent.reset_eval_context()
        match j_step:
            case -1: # model only
                # compute approximate model component: sum gamma^(j+1) * (w_j * V(s_(j+1))) for all trajectories
                first_states = torch.zeros(traj_states.size(0), 1, traj_states.size(-1), dtype=torch.long, device=self._device)
                first_actions = torch.zeros(traj_states.size(0), 1, 1, dtype=torch.long, device=self._device)
                for i in range(traj_states.size(0)):
                    if traj_valid_data_mask[i].any():
                        traj_valid_data_indices = traj_valid_data_mask[i].nonzero(as_tuple=True)[0]
                        first_states[i, 0] = traj_states[i, traj_valid_data_indices[0]]
                        first_actions[i, 0] = traj_actions[i, traj_valid_data_indices[0]]
                model_mask = traj_valid_data_mask.any(dim=1).squeeze(-1)
                _, traj_vals = self._fqe.compute_q_values_and_values(first_states[model_mask], first_actions[model_mask], tensors_need_reshape=True) # NOTE: THIS SHOULD WORK FOR TT
                masked_direct_method_value = (gammas[0] * timestep_weights[model_mask, 0] * traj_vals.squeeze(1)).squeeze(-1)
                direct_method_value = torch.zeros(traj_states.size(0), device=self._device)
                direct_method_value[model_mask] = masked_direct_method_value
                return direct_method_value, None
            case _:
                traj_valid_data_mask_sum = traj_valid_data_mask.sum(dim=1)
                if j_step > self._trajectory_dataset.num_time_steps:
                    batch_j_step = traj_valid_data_mask_sum
                    model_input_ts_delta = torch.zeros(traj_states.size(0), 1, dtype=torch.long, device=self._device)
                else:
                    # how many timesteps are valid for each trajectory?
                    less_than_j_step_mask = traj_valid_data_mask_sum < j_step
                    batch_j_step = torch.where(less_than_j_step_mask, traj_valid_data_mask_sum, j_step)
                    model_input_ts_delta = torch.where(less_than_j_step_mask, 0, 1)
                importance_weighted_component = torch.zeros(traj_states.size(0), 1, device=self._device)
                masked_direct_method_value = torch.zeros(traj_states.size(0), 1, device=self._device)
                control_variate_component = torch.zeros(traj_states.size(0), 1, device=self._device)
                # OPTIMIZE: this be slow af - quick and dirty to get results quickly... THIS IS AWFUL
                for i in range(traj_states.size(0)):
                    cur_j_step = batch_j_step[i]
                    cur_model_input_ts_delta = model_input_ts_delta[i]
                    cur_traj_valid_data_indices_0, cur_traj_valid_data_indices_1, cur_traj_valid_data_indices_2 = traj_valid_data_mask[i].unsqueeze(0).nonzero(as_tuple=True)
                    first_j_indices_0, first_j_indices_1, first_j_indices_2 = cur_traj_valid_data_indices_0[:cur_j_step + cur_model_input_ts_delta], cur_traj_valid_data_indices_1[:cur_j_step + cur_model_input_ts_delta], cur_traj_valid_data_indices_2[:cur_j_step + cur_model_input_ts_delta]
                    cur_traj_states = traj_states[i, first_j_indices_1].unsqueeze(0)
                    #assert (cur_traj_states[0] == traj_states[i, :cur_j_step + cur_model_input_ts_delta]).all(), 'issue'
                    cur_traj_actions = traj_actions[i, first_j_indices_1].unsqueeze(0)
                    #assert (cur_traj_actions[0] == traj_actions[i, :cur_j_step + cur_model_input_ts_delta]).all(), 'issue'
                    cur_traj_rewards = traj_rewards[i, first_j_indices_1].unsqueeze(0)
                    #assert cur_traj_rewards[0].allclose(traj_rewards[i, :cur_j_step + cur_model_input_ts_delta]), 'issue'
                    cur_timestep_weights = timestep_weights[i, first_j_indices_1].unsqueeze(0)
                    #assert cur_timestep_weights[0].allclose(timestep_weights[i, :cur_j_step + cur_model_input_ts_delta]), 'issue'
                    cur_prev_timestep_weights = prev_timestep_weights[i, first_j_indices_1].unsqueeze(0)
                    #assert cur_prev_timestep_weights[0].allclose(prev_timestep_weights[i, :cur_j_step + cur_model_input_ts_delta]), 'issue'

                    importance_weighted_component[i] = (gammas[:cur_j_step] * cur_timestep_weights[:, :cur_j_step] * cur_traj_rewards[:, :cur_j_step]).sum(dim=1).cpu().item()
                    #print('memory before q value and value', tracemalloc.get_traced_memory())
                    # get direct method's q-values and values for direct method and control variate computations
                    if isinstance(self._agent, AbstractSequenceAgent):
                        traj_q_vals, traj_vals = self._fqe.compute_q_values_and_values_for_sequence_agent(traj_states[i], traj_actions[i], traj_rewards[i], traj_valid_data_mask[i], cur_j_step + cur_model_input_ts_delta, tensors_need_reshape=True)
                    else:
                        traj_q_vals, traj_vals = self._fqe.compute_q_values_and_values(cur_traj_states, cur_traj_actions, tensors_need_reshape=True)
                    #print('memory after q value and value', tracemalloc.get_traced_memory())
                    # compute approximate model component: sum gamma^(j+1) * (w_j * V(s_(j+1))) for all trajectories
                    if cur_j_step < traj_valid_data_mask_sum[i]: # self._trajectory_dataset.num_time_steps:
                        masked_direct_method_value[i] = (gammas[cur_j_step + 1] * cur_timestep_weights[0, cur_j_step] * traj_vals[0, cur_j_step]).cpu().item()
                    else:
                        masked_direct_method_value[i] = 0
                    # compute control variate: sum gamma^t * (w_t * Q(s_t, a_t) - w_(t-1) * V(s_t)) for all timesteps, for all trajectories
                    weighted_traj_q_vals = cur_timestep_weights[:, :cur_j_step] * traj_q_vals[:, :cur_j_step]
                    weighted_traj_vals = cur_prev_timestep_weights[:, :cur_j_step] * traj_vals[:, :cur_j_step]
                    control_variate_component[i] = (gammas[:cur_j_step] * (weighted_traj_q_vals - weighted_traj_vals)).sum(dim=1).cpu().item()
                    # del cur_traj_states, cur_traj_actions, cur_traj_rewards, cur_timestep_weights, cur_prev_timestep_weights, traj_q_vals, traj_vals, weighted_traj_q_vals, weighted_traj_vals
                return importance_weighted_component + masked_direct_method_value - control_variate_component, control_variate_component

    def _compute_confidence_interval(self, unique_states_rep: torch.LongTensor, gammas: torch.FloatTensor, j_step_returns: torch.FloatTensor, traj_inf_step_returns: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """ Compute j-step returns for all trajectories.
        Returns a tensor with each j-step return for each trajectory.
        Trajectories are ordered by index in the dataset.
        :param unique_states_rep: Unique states repeated batch_size by num_time_steps times: (num_unique_states, batch_size x num_time_steps, state_dim).
        :param gammas: Batch of discount factors: (num_steps, 1).
        :param j_step_returns: j-step return values (|j_steps|, 1).
        :return: Trajectory j-step return values (|D|, |j_steps|).
        """
        #tracemalloc.start()
        # compute bootstrap sampled inf step returns
        k_conf_inf_step_returns = torch.zeros(self._confidence_estimation_iters_k, 1, device=self._device)
        for k in tqdm.tqdm(range(self._confidence_estimation_iters_k), desc='Computing confidence interval', unit=' iterations', colour='blue'):
            #print(f'\n\nk: {k} - {tracemalloc.get_traced_memory()}')
            bootstrap_traj_indices = torch.randint(len(self._trajectory_dataset), (len(self._trajectory_dataset),))
            bootstrap_traj_start_idx, bootstrap_traj_stop_idx = 0, min(self._batch_size, len(self._trajectory_dataset))
            while bootstrap_traj_start_idx < bootstrap_traj_indices.size(0):
                dataset_indices = bootstrap_traj_indices[bootstrap_traj_start_idx:bootstrap_traj_stop_idx]
                #traj_states, traj_actions, traj_rewards, _, traj_dones, traj_missing_data_mask = self._trajectory_dataset[dataset_indices]
                #traj_states, traj_actions, traj_rewards, traj_dones, traj_valid_data_mask = traj_states.long().to(self._device), traj_actions.to(self._device), traj_rewards.to(self._device), traj_dones.to(self._device), traj_missing_data_mask.logical_not().to(self._device)
                # compute importance weighted component: sum gamma^t * w_t * r_t for all timesteps, for all trajectories
                #print('memory before get_state_action_probs', tracemalloc.get_traced_memory())
                #behavior_action_probs, eval_action_probs = self._get_state_action_probs(traj_states, traj_actions, traj_valid_data_mask, unique_states_rep)
                #print('memory before compute_timestep_weights', tracemalloc.get_traced_memory())
                #timestep_weights = self._compute_timestep_weights(behavior_action_probs, eval_action_probs)
                #print('memory after compute_timestep_weights', tracemalloc.get_traced_memory())
                # NOTE: we are duplicating the first time step weight - this differs from the implementation here where they use 1 / |Dataset|
                # See https://github.com/clvoloshin/COBS/blob/master/ope/algos/doubly_robust_v2.py#L71
                # prev_timestep_weights = torch.cat([
                #     timestep_weights[:, 0].unsqueeze(-1),
                #     timestep_weights[:, :-1]
                # ], dim=1)
                #print('memory before compute_j_step_return', tracemalloc.get_traced_memory())
                # compute inf j-step return
                #inf_step_return = self._compute_j_step_return(traj_states, traj_actions, traj_rewards, traj_valid_data_mask, timestep_weights, prev_timestep_weights, gammas, float('inf')) # type: ignore
                inf_step_return = traj_inf_step_returns[dataset_indices]
                #print('memory after compute_j_step_return', tracemalloc.get_traced_memory())
                k_conf_inf_step_returns[k] += inf_step_return.sum()
                # update batch indices
                bootstrap_traj_start_idx = bootstrap_traj_stop_idx
                bootstrap_traj_stop_idx = min(len(self._trajectory_dataset), bootstrap_traj_stop_idx + dataset_indices.size(0))
        #tracemalloc.stop()
        # # ascending sort bootstrap sampled inf step returns
        # k_conf_inf_step_returns = k_conf_inf_step_returns.sort(dim=0).values
        # compute confidence interval
        k_inf_step_returns_std, k_inf_step_returns_mean = torch.std_mean(k_conf_inf_step_returns)
        k_inf_step_returns_sem = k_inf_step_returns_std / math.sqrt(k_conf_inf_step_returns.size(0)) # standard error of mean
        margin_of_error = scipy.stats.t.interval(0.9, k_conf_inf_step_returns.size(0) - 1)[1]
        conf_lower = k_inf_step_returns_mean - margin_of_error * k_inf_step_returns_sem
        conf_upper = k_inf_step_returns_mean + margin_of_error * k_inf_step_returns_sem
        # get inf step return
        inf_step_return = j_step_returns[self._j_steps.index(float('inf'))]
        # update confidence interval using WDR estimate
        conf_lower = torch.minimum(conf_lower, inf_step_return)
        conf_upper = torch.maximum(conf_upper, inf_step_return)
        return conf_lower, conf_upper

    def _compute_j_step_weights(self, j_step_returns: torch.FloatTensor, traj_j_step_returns: torch.FloatTensor, conf_lower: torch.FloatTensor, conf_upper: torch.FloatTensor) -> torch.FloatTensor:
        """ Compute j-step return weights using Blending IS and Model (BIM).
        :param j_step_returns: j-step return values (|j_steps|, 1).
        :param traj_j_step_returns: Trajectory j-step return values (|D|, |j_steps|).
        :param conf_lower: Lower bound of confidence interval.
        :param conf_upper: Upper bound of confidence interval.
        :return: j-step return weights (|j_steps|,).
        """
        # compute bias:
        # bias = 0 if j-step return is within confidence interval
        # bias = j-step return - conf_lower if j-step return < conf_lower
        # bias = j-step return - conf_upper if j-step return > conf_upper
        bias_vec = torch.zeros_like(j_step_returns)
        lower_mask = j_step_returns < conf_lower
        bias_vec[lower_mask] = j_step_returns[lower_mask] - conf_lower
        upper_mask = j_step_returns > conf_upper
        bias_vec[upper_mask] = j_step_returns[upper_mask] - conf_upper
        # compute sample covariance matrix
        covariance = torch.cov(traj_j_step_returns.T)
        # compute j-step return weights
        error = covariance + bias_vec * bias_vec.T
        constraint = {'type': 'eq', 'fun': lambda x: x.sum() - 1.0}
        bounds = [(0, 1)] * len(self._j_steps)
        mse_loss_fn = lambda x, error_: np.dot(np.dot(x, error_), x.T)
        op_ret = scipy.optimize.minimize(mse_loss_fn, np.zeros(len(self._j_steps)), args=error.detach().cpu().numpy(), bounds=bounds, constraints=constraint)
        assert op_ret.success, 'optimization failed'
        final_weights = torch.from_numpy(op_ret.x).float().to(self._device)
        return final_weights

    @property
    def j_steps(self) -> List[int | float]:
        return self._j_steps

    @property
    def j_step_weights(self) -> Optional[List[float]]:
        return self._j_step_weights
