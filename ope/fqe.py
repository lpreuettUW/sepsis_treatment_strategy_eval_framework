import torch
import mlflow
from copy import deepcopy
from tqdm.auto import tqdm
from typing import Tuple, Optional

from ope.abstract_ope_method import AbstractOPEMethod
from agents.abstract_agent import AbstractAgent
from agents.abstract_sequence_agent import AbstractSequenceAgent
from utilities.sequence_agent_interface import SequenceAgentInterface


class FittedQEvaluation(AbstractOPEMethod):
    def __init__(self, trajectory_dataset: torch.utils.data.Dataset, agent: AbstractAgent, k_itrs: Optional[int] = None, convergence_eps: Optional[float] = None,
                 max_train_itrs: Optional[int] = None, lr: Optional[float] = None, hidden_size: Optional[int] = None, batch_size: Optional[int] = None,
                 run_id: Optional[str] = None, seq_agent_interface: Optional[SequenceAgentInterface] = None, use_behavior_policy_states: bool = False,
                 verbose: bool = False, split_num: Optional[int] = None):
        if run_id is None and (k_itrs is None or convergence_eps is None or max_train_itrs is None or lr is None or hidden_size is None or batch_size is None):
            raise ValueError('Must specify either run_id or all of k_itrs, convergence_eps, max_train_itrs, lr, hidden_size, and batch_size.')
        if isinstance(agent, AbstractSequenceAgent) and seq_agent_interface is None:
            raise ValueError('Must specify seq_agent_interface if agent is AbstractSequenceAgent.')
        self._k_itrs = k_itrs
        self._convergence_eps = convergence_eps
        self._max_train_itrs = max_train_itrs
        self._batch_size = batch_size
        self._hidden_size = hidden_size
        if hidden_size is None:
            self._q_function = self._optimizer = self._trajectory_dataset_data_loader = None
        else:
            self._q_function = self._initialize_q_function((trajectory_dataset.behavior_state_dim if use_behavior_policy_states else trajectory_dataset.eval_state_dim), trajectory_dataset.action_dim)
            self._optimizer = torch.optim.Adam(self._q_function.parameters(), lr=lr)
            self._trajectory_dataset_data_loader = torch.utils.data.DataLoader(trajectory_dataset, batch_size=self._batch_size, shuffle=False)
        self._run_id = run_id
        self._seq_agent_interface = seq_agent_interface
        self._use_behavior_policy_states = use_behavior_policy_states
        self._verbose = verbose
        self._split_num = split_num
        self._cached_unique_state_action_probs = None
        super().__init__(trajectory_dataset, agent)

    def _initialize_q_function(self, state_dim: int, action_dim: int) -> torch.nn.Module:
        return torch.nn.Sequential(
            torch.nn.Linear(state_dim + action_dim, self._hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self._hidden_size, self._hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self._hidden_size, 1)
        )

    def _build_supervised_dataset(self) -> torch.utils.data.Dataset:
        self._q_function.eval()
        all_state_action_pairs = all_value_targets = None
        with torch.no_grad():
            for behavior_states, eval_states, discrete_actions, continuous_actions, rewards, behavior_next_states, eval_next_states, dones, missing_data_mask in self._trajectory_dataset_data_loader: #tqdm(self._trajectory_dataset_data_loader, desc='Building Supervised Dataset', unit='batch', colour='green'):
                valid_data_mask = missing_data_mask.squeeze(-1).logical_not()
                if valid_data_mask.any():
                    behavior_states, eval_states, discrete_actions, rewards, behavior_next_states, eval_next_states, dones = behavior_states[valid_data_mask], eval_states[valid_data_mask], discrete_actions[valid_data_mask], rewards[valid_data_mask], behavior_next_states[valid_data_mask], eval_next_states[valid_data_mask], dones[valid_data_mask]
                    behavior_states, eval_states, discrete_actions, rewards, behavior_next_states, eval_next_states, dones = behavior_states.to(self._device), eval_states.to(self._device), discrete_actions.to(self._device), rewards.to(self._device), behavior_next_states.to(self._device), eval_next_states.to(self._device), dones.to(self._device)
                    next_actions, probs = self._agent.get_best_action(eval_states)
                    next_q_values = self._q_function(torch.cat([(behavior_next_states if self._use_behavior_policy_states else eval_next_states), next_actions.action.unsqueeze(-1)], dim=-1))
                    target_q_values = rewards + (1.0 - dones.float()) * next_q_values
                    if all_state_action_pairs is None:
                        all_state_action_pairs = torch.cat([(behavior_states if self._use_behavior_policy_states else eval_states), discrete_actions], dim=-1)
                        all_value_targets = target_q_values
                    else:
                        all_state_action_pairs = torch.cat([all_state_action_pairs, torch.cat([(behavior_states if self._use_behavior_policy_states else eval_states), discrete_actions], dim=-1)], dim=0)
                        all_value_targets = torch.cat([all_value_targets, target_q_values], dim=0)
        supervised_dataset = torch.utils.data.TensorDataset(all_state_action_pairs, all_value_targets)
        return supervised_dataset

    def _build_supervised_dataset_for_sequence_agent(self) -> torch.utils.data.Dataset:
        raise NotImplementedError('This method is broken.')
        self._q_function.eval()
        all_state_action_pairs = all_value_targets = None
        with torch.no_grad():
            for states_traj, discrete_actions_traj, continuous_actions_traj, rewards_traj, next_states_traj, dones_traj, missing_data_mask_traj in tqdm(self._trajectory_dataset_data_loader, desc='Building Supervised Dataset', unit='batch', colour='green', position=0, leave=True):
                valid_data_mask = missing_data_mask_traj.squeeze(-1).logical_not()
                if valid_data_mask.any():
                    states_traj, discrete_actions_traj, rewards_traj, next_states_traj, dones_traj, valid_data_mask = states_traj.to(self._device), discrete_actions_traj.to(self._device), rewards_traj.to(self._device), next_states_traj.to(self._device), dones_traj.to(self._device), valid_data_mask.to(self._device)
                    next_actions_traj = self._seq_agent_interface.get_best_next_actions(states_traj, discrete_actions_traj, rewards_traj, next_states_traj, dones_traj, valid_data_mask)
                    next_actions = next_actions_traj.view(-1, 1)
                    next_states = next_states_traj.view(-1, next_states_traj.size(-1))
                    next_q_values = self._q_function(torch.cat([next_states, next_actions], dim=-1))
                    target_q_values = rewards_traj.view(-1, 1) + (1.0 - dones_traj.float().view(-1, 1)) * next_q_values
                    if all_state_action_pairs is None:
                        all_state_action_pairs = torch.cat([states_traj.view(-1, states_traj.size(-1)), discrete_actions_traj.view(-1, 1)], dim=-1)
                        all_value_targets = target_q_values
                    else:
                        all_state_action_pairs = torch.cat([all_state_action_pairs, torch.cat([states_traj.view(-1, states_traj.size(-1)), discrete_actions_traj.view(-1, 1)], dim=-1)], dim=0)
                        all_value_targets = torch.cat([all_value_targets, target_q_values], dim=0)
        supervised_dataset = torch.utils.data.TensorDataset(all_state_action_pairs, all_value_targets)
        return supervised_dataset

    def _fit_to_supervised_dataset(self, supervised_dataset: torch.utils.data.Dataset, k_itr: int) -> Tuple[int, float]:
        self._q_function.train()
        num_itrs = 0
        loss_delta = float('inf')
        batch_loader = torch.utils.data.DataLoader(supervised_dataset, batch_size=self._batch_size, shuffle=True)
        mse_loss = torch.nn.MSELoss()
        best_loss = best_weights = None
        epoch = 0
        while num_itrs < self._max_train_itrs and loss_delta > self._convergence_eps:
            loss_delta = 0
            prev_loss = None
            num_itrs += 1
            epoch_loss = 0
            for state_action_pairs, value_targets in batch_loader: #tqdm(batch_loader, desc=f'Fitting to Supervised Dataset (k_itr={k_itr})', unit='batch', colour='green'):
                state_action_pairs, value_targets = state_action_pairs.to(self._device), value_targets.to(self._device)
                self._optimizer.zero_grad()
                q_values = self._q_function(state_action_pairs)
                loss = mse_loss(q_values, value_targets)
                loss.backward()
                self._optimizer.step()
                if prev_loss is not None:
                    loss_delta = max(loss_delta, abs(prev_loss - loss.item()))
                if best_loss is None or loss.detach().cpu().item() < best_loss:
                    best_loss = loss.detach().cpu().item()
                    best_weights = deepcopy(self._q_function.state_dict())
                prev_loss = loss.item()
                epoch_loss += loss.item()
            epoch_loss /= len(batch_loader)
            mlflow.log_metrics({f'k_itr_{k_itr}_avg_loss': epoch_loss}, step=epoch)
            epoch += 1
            # restore best weights
            self._q_function.load_state_dict(best_weights)
        if self._verbose:
            print(f'fitting took {num_itrs} itrs')
            print(f'final loss delta: {loss_delta:.7f}')
        return num_itrs, loss_delta

    def _get_mapped_action_probs(self, states: torch.FloatTensor) -> torch.FloatTensor:
        if self._use_behavior_policy_states:
            if self._cached_unique_state_action_probs is None:
                self._cache_unique_state_action_probs()
            behavior_unique_states, _ = self._trajectory_dataset.get_unique_states(self._use_behavior_policy_states)
            behavior_unique_states_rep = behavior_unique_states.repeat(states.size(0), 1, 1).to(self._device)
            behavior_state_indices = (behavior_unique_states_rep == states.unsqueeze(1)).nonzero()[:, 1] # cut out batch index dim
            return self._cached_unique_state_action_probs[behavior_state_indices]
        else:
            return self._agent.get_action_probs(states)

    def _cache_unique_state_action_probs(self):
        # we need to map the continuous eval states to the discrete behavior policy states
        # we will take the mean of the action probs for each eval state that maps to a behavior state
        # repeat for all behavior states
        self._agent.eval()
        dataset_flattened_state = self._trajectory_dataset.is_flattened
        self._trajectory_dataset.is_flattened = True
        # get unique states
        behavior_unique_states, _ = self._trajectory_dataset.get_unique_states(self._use_behavior_policy_states)
        self._cached_unique_state_action_probs = torch.zeros(behavior_unique_states.size(0), self._trajectory_dataset.num_actions, device=self._device)
        behavior_state_observation_counts = torch.zeros(behavior_unique_states.size(0), device=self._device)
        # iterate over the dataset
        dataloader = torch.utils.data.DataLoader(self._trajectory_dataset, batch_size=128, shuffle=False)
        for behavior_states, eval_states, _, _, _, _, _, _, missing_data_mask in tqdm(dataloader, desc='Caching Unique State Action Probs', unit='batch', colour='green', position=0, leave=True):
            valid_data_mask = missing_data_mask.squeeze(-1).logical_not()
            if valid_data_mask.any():
                behavior_states = behavior_states[valid_data_mask].to(self._device)
                eval_states = eval_states[valid_data_mask].to(self._device)
                behavior_unique_states_rep = behavior_unique_states.repeat(behavior_states.size(0), 1, 1).to(self._device)
                # update behavior state observation counts
                behavior_state_observations = (behavior_unique_states_rep == behavior_states.unsqueeze(1)).sum(0)
                behavior_state_observation_counts += behavior_state_observations.squeeze(-1)
                # get unique behavior state indices
                behavior_state_indices = (behavior_unique_states_rep == behavior_states.unsqueeze(1)).squeeze(-1).nonzero()
                action_probs = self._agent.get_action_probs(eval_states).detach()
                discrete_state_action_probs_map = torch.zeros(behavior_unique_states.size(0), behavior_states.size(0), self._trajectory_dataset.num_actions, device=self._device) # behavior states x batch size x num actions
                discrete_state_action_probs_map[behavior_state_indices[:, 1], behavior_state_indices[:, 0]] += action_probs
                self._cached_unique_state_action_probs += discrete_state_action_probs_map.sum(1) # sum out the batch dim
        # normalize the action probs - first add epsilon to avoid division by zero
        behavior_state_observation_counts[behavior_state_observation_counts == 0] += 1e-6 # NOTE: the probability of observing a state may be zero, but the action probabilities wont matter if that is the case
        self._cached_unique_state_action_probs /= behavior_state_observation_counts.unsqueeze(-1).repeat(1, self._trajectory_dataset.num_actions)
        # revert dataset to original flattened state
        self._trajectory_dataset.is_flattened = dataset_flattened_state

    # region AbstractOPEMethod

    def _initialize(self):
        if not self._run_id:
            self._q_function.to(self._device)
            is_seq_agent = isinstance(self._agent, AbstractSequenceAgent)
            self._trajectory_dataset.reshape_data(not is_seq_agent)
            for k_itr in tqdm(range(self._k_itrs), desc='FQE', unit=' k_itr', position=0, leave=True):
                supervised_dataset = self._build_supervised_dataset_for_sequence_agent() if is_seq_agent else self._build_supervised_dataset()
                num_itrs, loss_delta = self._fit_to_supervised_dataset(supervised_dataset, k_itr)
                mlflow.log_metrics({'num_itrs': num_itrs, 'loss_delta': loss_delta}, step=k_itr)
        else:
            self.restore_model(self._run_id)

    def reinitialize(self, agent: AbstractAgent):
        self._agent = agent
        self._q_function = self._initialize_q_function((self._trajectory_dataset.behavior_state_dim if self._use_behavior_policy_states else self._trajectory_dataset.eval_state_dim), self._trajectory_dataset.action_dim)
        self._cached_unique_state_action_probs = None
        self._initialize()

    def compute_value(self) -> float:
        behavior_unique_states, eval_unique_states = self._trajectory_dataset.get_unique_states(self._use_behavior_policy_states)
        start_idx = 0
        stop_idx = min(self._batch_size, behavior_unique_states.size(0))
        q_values = torch.zeros(self._batch_size, 1, device=self._device)
        while start_idx < eval_unique_states.size(0):
            cur_batch_size = stop_idx - start_idx # we use this in case the last slice is smaller than the batch size
            eval_states_slice = eval_unique_states[start_idx:stop_idx].to(self._device)
            fqe_state = (behavior_unique_states[start_idx:stop_idx] if self._use_behavior_policy_states else eval_states_slice).to(self._device)
            actions, probs = self._agent.get_best_action(eval_states_slice)
            state_action_pairs = torch.cat([fqe_state, actions.action.unsqueeze(-1)], dim=-1)
            with torch.no_grad():
                cur_q_vals = self._q_function(state_action_pairs)
            q_values[:cur_batch_size] += cur_q_vals
            start_idx += self._batch_size
            stop_idx = min(stop_idx + self._batch_size, eval_unique_states.size(0))
        aggregate_sum = False
        if aggregate_sum:
            policy_value = q_values.sum().cpu().item()
        else:
            policy_value = q_values.sum().cpu().item() / eval_unique_states.size(0)
        return policy_value

    # endregion

    def compute_q_values_and_values(self, traj_states: torch.LongTensor, traj_actions: torch.LongTensor, tensors_need_reshape: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """ Compute the q-values and values of trajectory batch.
        We use FQE's learned q-function to compute values as follows:
        V(s) = SUM(pi(a | s) * Q(s, a)) for all actions a in A(s)
        where pi is the evaluation policy and Q(s, a) is the q-function learned by FQE.
        :param traj_states: Batch of state trajectories: (batch, num_steps, state_dim).
        :param traj_actions: Batch of action trajectories: (batch, num_steps, action_dim).
        :return: q-values and values: ((batch, num_steps, 1), (batch, num_steps, 1)).
        """
        self._q_function.eval()
        # compute q values
        if tensors_need_reshape:
            individual_states = traj_states.reshape(-1, traj_states.size(-1)).float()
            individual_actions = traj_actions.reshape(-1, traj_actions.size(-1)).float()
        else:
            individual_states = traj_states.view(-1, traj_states.size(-1)).float()
            individual_actions = traj_actions.view(-1, traj_actions.size(-1)).float()
        individual_state_action_pairs = torch.cat([individual_states, individual_actions], dim=-1)
        traj_q_vals = self._q_function(individual_state_action_pairs).view(traj_states.size(0), traj_states.size(1), 1)
        # compute values
        num_actions = self._trajectory_dataset.num_actions # cache it
        rep_interleaved_states = traj_states.repeat_interleave(num_actions, dim=0).float().view(-1, traj_states.size(-1))
        rep_actions = torch.arange(num_actions, dtype=torch.float, device=self._device).repeat(individual_states.size(0)).view(-1, 1) # NOTE: this only works bc action_dim is 1
        rep_state_action_pairs = torch.cat([rep_interleaved_states, rep_actions], dim=-1)
        traj_rep_q_vals = self._q_function(rep_state_action_pairs).view(traj_states.size(0), traj_states.size(1), num_actions) # this should be what we want
        if isinstance(self._agent, AbstractSequenceAgent):
            raise NotImplementedError('This method is broken for AbstractSequenceAgents.')
            traj_rep_probs = self._agent.get_action_probs(individual_states, torch.ones(individual_states.size(0), 1, dtype=torch.bool, device=self._device)).view(traj_states.size(0), traj_states.size(1), num_actions)  # this should be what we want
        else:
            # TODO: we need to map all the eval states which map to the behavior states here so we can compute the correct probabilities
            traj_rep_probs = self._get_mapped_action_probs(individual_states).view(traj_states.size(0), traj_states.size(1), num_actions) # this should be what we want
            # traj_rep_probs = self._agent.get_action_probs(individual_states).view(traj_states.size(0), traj_states.size(1), num_actions) # this should be what we want
        traj_values = (traj_rep_probs * traj_rep_q_vals).sum(dim=-1, keepdim=True)
        return traj_q_vals, traj_values

    def compute_q_values_and_values_for_sequence_agent(self, traj_states: torch.LongTensor, traj_actions: torch.LongTensor, traj_rewards: torch.FloatTensor, traj_valid_data_mask: torch.BoolTensor,
                                                       num_values_needed: int, tensors_need_reshape: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Works for a single trajectory."""
        self._q_function.eval()
        assert isinstance(self._agent, AbstractSequenceAgent)
        assert self._seq_agent_interface is not None
        # compute q values
        individual_valid_data_mask = traj_valid_data_mask.view(-1)
        if tensors_need_reshape:
            individual_states = traj_states.reshape(-1, traj_states.size(-1)).float()[individual_valid_data_mask]
            individual_actions = traj_actions.reshape(-1, traj_actions.size(-1)).float()[individual_valid_data_mask]
        else:
            individual_states = traj_states.view(-1, traj_states.size(-1)).float()[individual_valid_data_mask]
            individual_actions = traj_actions.view(-1, traj_actions.size(-1)).float()[individual_valid_data_mask]
        individual_state_action_pairs = torch.cat([individual_states, individual_actions], dim=-1)
        traj_q_vals = self._q_function(individual_state_action_pairs).view(-1, 1)
        # compute values
        num_actions = self._trajectory_dataset.num_actions  # cache it
        rep_interleaved_states = individual_states.repeat_interleave(num_actions, dim=0).float()#.view(-1, traj_states.size(-1))
        rep_actions = torch.arange(num_actions, dtype=torch.float, device=self._device).repeat(individual_states.size(0)).view(-1, 1)  # NOTE: this only works bc action_dim is 1
        rep_state_action_pairs = torch.cat([rep_interleaved_states, rep_actions], dim=-1)
        traj_rep_q_vals = self._q_function(rep_state_action_pairs).view(-1, num_actions)  # this should be what we want
        traj_rep_probs = self._seq_agent_interface.get_action_probs(traj_states.unsqueeze(0).float(), traj_actions.unsqueeze(0), traj_rewards.unsqueeze(0), traj_valid_data_mask.unsqueeze(0), reset_eval_context=False).view(-1, num_actions)
        traj_rep_probs = traj_rep_probs[individual_valid_data_mask]
        #traj_rep_probs = self._agent.get_action_probs(individual_states).view(traj_states.size(0), traj_states.size(1), num_actions)  # this should be what we want
        traj_values = (traj_rep_probs * traj_rep_q_vals).sum(dim=-1, keepdim=True)
        return traj_q_vals[:num_values_needed].unsqueeze(0), traj_values[:num_values_needed].unsqueeze(0)

    def log_model(self):
        model_name = 'fqe_q_function' if self._split_num is None else f'fqe_q_function_split_{self._split_num}'
        mlflow.pytorch.log_model(self._q_function, model_name)

    def restore_model(self, run_id: str):
        model_name = 'fqe_q_function' if self._split_num is None else f'fqe_q_function_split_{self._split_num}'
        self._q_function = mlflow.pytorch.load_model(f'runs:/{run_id}/{model_name}', map_location=self._device)
        self._q_function.eval()
