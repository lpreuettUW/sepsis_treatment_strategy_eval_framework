import torch

from mdp.action import Action
from agents.abstract_sequence_agent import AbstractSequenceAgent
from agents.implicit_q_learning import ImplicitQLearning
from utilities.device_manager import DeviceManager


class SequenceAgentInterface:
    def __init__(self, agent: AbstractSequenceAgent, iql_agent: ImplicitQLearning):
        self._agent = agent
        self._iql_agent = iql_agent
        self._device = DeviceManager.get_device()

    def get_best_next_actions(self, states_traj: torch.FloatTensor, actions_traj: torch.LongTensor, rewards_traj: torch.FloatTensor, next_states_traj: torch.FloatTensor,
                              dones_traj: torch.BoolTensor, missing_data_mask_traj: torch.BoolTensor) -> torch.LongTensor:
        """
        Get the best next actions for a trajectory of states and actions.
        :param states_traj: Trajectory of states.
        :param actions_traj: Trajectory of actions.
        :param rewards_traj: Trajectory of rewards.
        :param next_states_traj: Trajectory of next states.
        :param dones_traj: Trajectory of dones.
        :param missing_data_mask_traj: Trajectory of missing data masks.
        :return: Best next actions for the trajectory.
        """
        self._agent.eval()
        self._iql_agent.eval()
        states_traj, actions_traj, rewards_traj, next_states_traj, dones_traj, valid_data_mask_traj = states_traj.to(self._device), actions_traj.to(self._device), rewards_traj.to(self._device), next_states_traj.to(self._device), dones_traj.to(self._device), missing_data_mask_traj.to(self._device).logical_not()
        # extract values
        values = torch.zeros(states_traj.size(0) * states_traj.size(1), 1, device=self._device)
        valid_states = states_traj[valid_data_mask_traj.squeeze(-1)].view(-1, states_traj.size(-1))
        valid_values = self._iql_agent.compute_value(valid_states)
        values[valid_data_mask_traj.view(-1).squeeze(-1)] = valid_values
        values_traj = values.view(states_traj.size(0), states_traj.size(1), 1)
        # compute best next state actions
        next_state_actions = None
        self._agent.reset_eval_context()
        self._agent.get_best_action(states_traj[:, 0], valid_data_mask_traj[:, 0]) # ignore output
        for t in range(next_states_traj.size(1)):
            # update seq agent with previous state context
            self._agent.update_eval_context_with_observations(Action(action=actions_traj[:, t]), rewards_traj[:, t], values_traj[:, t], valid_data_mask_traj[:, t])
            next_states_valid_data_mask = valid_data_mask_traj[:, t + 1] if t < next_states_traj.size(1) - 1 else torch.ones_like(valid_data_mask_traj[:, t])
            cur_actions, _ = self._agent.get_best_action(next_states_traj[:, t], next_states_valid_data_mask)
            if next_state_actions is None:
                next_state_actions = cur_actions.action.unsqueeze(1)
            else:
                next_state_actions = torch.cat((next_state_actions, cur_actions.action.unsqueeze(1)), dim=1)

        return next_state_actions

    def get_action_probs(self, states_traj: torch.FloatTensor, actions_traj: torch.LongTensor, rewards_traj: torch.FloatTensor, valid_data_mask_traj: torch.BoolTensor, reset_eval_context: bool = True) -> torch.FloatTensor:
        self._agent.eval()
        self._iql_agent.eval()
        if reset_eval_context:
            self._agent.reset_eval_context()
        states_traj, actions_traj, rewards_traj, valid_data_mask_traj = states_traj.to(self._device), actions_traj.to(self._device), rewards_traj.to(self._device), valid_data_mask_traj.to(self._device)
        # extract values
        values = torch.zeros(states_traj.size(0) * states_traj.size(1), 1, device=self._device)
        valid_states = states_traj[valid_data_mask_traj.squeeze(-1)].view(-1, states_traj.size(-1))
        valid_values = self._iql_agent.compute_value(valid_states)
        values[valid_data_mask_traj.view(-1).squeeze(-1)] = valid_values
        values_traj = values.view(states_traj.size(0), states_traj.size(1), 1)
        # compute action probs
        traj_probs = torch.zeros(states_traj.size(0), states_traj.size(1), self._agent._num_actions, device=self._device)
        for t in range(states_traj.size(1)):
            probs = self._agent.get_action_probs(states_traj[:, t], valid_data_mask_traj[:, t])
            self._agent.update_eval_context_with_observations(Action(action=actions_traj[:, t]), rewards_traj[:, t], values_traj[:, t], valid_data_mask_traj[:, t])
            traj_probs[:, t] = probs
        return traj_probs
