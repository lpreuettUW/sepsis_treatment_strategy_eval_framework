import copy
import torch
import mlflow
from typing import Tuple, Final, Dict, Any

from mdp.policies.dueling_dqn import DuelingDQN # NOTE: we should pass this in as a parameter...but for now, we'll just import it
from mdp.action import Action
from agents.replay_buffers.prioritized_replay import PrioritizedReplayBuffer
from agents.abstract_agent import AbstractAgent


class D3QN(AbstractAgent): # TODO: we should probably use BatchAgent...
    """
    Offline implementation of Dueling Double Deep Q Network (D3QN) agent proposed by Raghu et al. (2017).
    This implementation leverages a prioritized replay buffer to match Raghu et al.'s implementation.
    Adapted from the following implementations:
    - https://github.com/aniruddhraghu/sepsisrl/blob/master/continuous/q_network.ipynb
    - https://github.com/philtabor/Deep-Q-Learning-Paper-To-Code/blob/master/DuelingDDQN/dueling_ddqn_agent.py
    """
    def __init__(self, gamma: float, lr: float, tau: float, batch_size: int, buffer_size: int, state_dim: int, action_dim: int, num_actions: int,
                 hidden_dim: int, reward_max: float, reg_lambda: float, per_alpha: float, per_beta: float, per_eps: float):
        """
        Initialize D3QN agent.

        :param gamma: Discount factor
        :param lr: Learning rate
        :param tau: Target network update rate (soft update)
        :param batch_size: Batch size
        :param buffer_size: Replay buffer size
        :param state_dim: State dimension
        :param action_dim: Action dimension
        :param num_actions: Number of actions
        :param hidden_dim: Network hidden dimension
        :param reward_max: Absolute maximum possible reward
        :param reg_lambda: Regularization lambda penalizing Q-values greater than the max possible reward
        :param per_alpha: Prioritization weight
        :param per_beta: Bias correction weight
        :param per_eps: Small constant to avoid zero priority
        """
        self._gamma = gamma
        self._tau = tau
        self._batch_size = batch_size
        self._num_actions = num_actions
        # Q-Network
        self._q_network = DuelingDQN(state_dim, num_actions, hidden_dim)
        super().__init__(self._q_network) # AbstractAgent will put the policy on the correct device
        self._target_q_network = DuelingDQN(state_dim, num_actions, hidden_dim).to(self._device)
        self._target_q_network.load_state_dict(self._q_network.state_dict())
        self._q_network_optimizer = torch.optim.Adam(self._q_network.parameters(), lr=lr)
        # Replay Buffer
        self._replay_buffer = PrioritizedReplayBuffer(state_dim, action_dim, buffer_size, per_alpha, per_beta, per_eps)
        # Constants
        self._reward_max: Final[float] = reward_max
        self._reg_lambda: Final[float] = reg_lambda

    # region AbstractAgent

    def get_action(self, env_state: torch.FloatTensor) -> Action | Tuple[Action, ...]:
        """
        Get action.

        :param env_state: Environment state.
        :return: Action.
        """
        actions, _ = self.get_best_action(env_state)
        return actions

    def get_best_action(self, env_state: torch.FloatTensor) -> Tuple[Action, torch.FloatTensor] | Tuple[Tuple[Action, ...], Tuple[torch.FloatTensor, ...]]:
        """
        Get best action.

        :param env_state: Environment state.
        :return: Tuple of Action and Probabilities.
        """
        states = env_state.to(self._device)
        with torch.no_grad():
            _, advantages = self._q_network(states)
            # NOTE: selecting the max advantage is equivalent to selecting the max Q-value
            actions = advantages.argmax(dim=-1)
            probs = torch.zeros_like(advantages)
            probs[torch.arange(actions.size(0), device=actions.device), actions] = 1.0 # Q Learning is deterministic
        actions = Action(action=actions)
        return actions, probs

    def get_action_probs(self, env_state: torch.FloatTensor) -> torch.FloatTensor | Tuple[torch.FloatTensor, ...]:
        """
        Get action probabilities.

        :param env_state: Environment state.
        :return: Action probabilities.
        """
        _, probs = self.get_best_action(env_state)
        return probs

    # endregion

    # region Public Functions/Methods

    def batch_train(self) -> Dict[str, float]:
        if self._replay_buffer.size < self._batch_size:
            raise ValueError('replay buffer does not have enough transitions to sample from')
        states, actions, rewards, next_states, dones, is_weights, transitions_indices = self._replay_buffer.sample(self._batch_size)
        states = states.to(self._device)
        actions = actions.to(self._device)
        rewards = rewards.to(self._device)
        next_states = next_states.to(self._device)
        dones = dones.to(self._device).long()
        is_weights = is_weights.to(self._device).unsqueeze(-1)
        # Compute Q-values
        values, advantages = self._q_network(states)
        q_values = (values + advantages - advantages.mean(dim=-1, keepdim=True)).gather(-1, actions)
        # Compute Next Actions using Q-network for Double Q-learning
        next_values, next_advantages = self._q_network(next_states)
        next_q_values = next_values + next_advantages - next_advantages.mean(dim=-1, keepdim=True)
        next_actions = next_q_values.argmax(dim=-1).unsqueeze(-1)
        # Compute Target Q-values
        target_next_values, target_next_advantages = self._target_q_network(next_states)
        target_next_q_values = target_next_values + target_next_advantages - target_next_advantages.mean(dim=-1, keepdim=True)
        target_q_values = rewards.unsqueeze(-1) + self._gamma * target_next_q_values.gather(-1, next_actions) * (1 - dones.unsqueeze(-1))
        # Compute TD Errors
        self._q_network_optimizer.zero_grad()
        total_loss, per_loss, reg_term, abs_td_error = self._compute_loss(q_values, target_q_values, is_weights)
        total_loss.backward()
        self._q_network_optimizer.step()
        # Update priorities in replay buffer
        self._replay_buffer.update_priorities(transitions_indices, abs_td_error.squeeze(-1).detach().cpu())
        # Soft update target network
        self._soft_update_target_network()
        # Increment training step
        self._replay_buffer.increment_step()
        return {
            'total_loss': total_loss.detach().cpu().item(),
            'per_loss': per_loss.detach().cpu().item(),
            'reg_term': reg_term.detach().cpu().item(),
            'abs_td_error': abs_td_error.mean().detach().cpu().item()
        }

    def evaluate(self, states: torch.FloatTensor, actions: torch.LongTensor, rewards: torch.FloatTensor, next_states: torch.FloatTensor, dones: torch.BoolTensor) -> Dict[str, float]:
        # NOTE: we could refactor and cleanup this dupe code...
        states = states.to(self._device)
        actions = actions.to(self._device)
        rewards = rewards.to(self._device)
        next_states = next_states.to(self._device)
        dones = dones.to(self._device).long()
        # Compute Q-values
        values, advantages = self._q_network(states)
        q_values = (values + advantages - advantages.mean(dim=-1, keepdim=True)).gather(-1, actions)
        # Compute Next Actions using Q-network for Double Q-learning
        next_values, next_advantages = self._q_network(next_states)
        next_q_values = next_values + next_advantages - next_advantages.mean(dim=-1, keepdim=True)
        next_actions = next_q_values.argmax(dim=-1).unsqueeze(-1)
        # Compute Target Q-values
        target_next_values, target_next_advantages = self._target_q_network(next_states)
        target_next_q_values = target_next_values + target_next_advantages - target_next_advantages.mean(dim=-1, keepdim=True)
        target_q_values = rewards.unsqueeze(-1) + self._gamma * target_next_q_values.gather(-1, next_actions) * (1 - dones.unsqueeze(-1))
        # Compute TD Errors
        abs_td_error = (target_q_values - q_values).abs()
        td_error = (target_q_values - q_values).pow(2)
        return {
            'td_error': td_error.sum().detach().cpu().item(),
            'abs_td_error': abs_td_error.sum().detach().cpu().item()
        }

    def fill_replay_buffer(self, dataloader: torch.utils.data.DataLoader):
        """
        Fill replay buffer with data for offline RL.

        :param dataloader: Offline RL DataLoader.
        """
        self._replay_buffer.fill(dataloader, discrete_actions=True)

    def save_model(self, name_prefix: str):
        """
        Save model.

        :param name_prefix: Prefix for model name.
        """
        mlflow.pytorch.log_model(self._q_network, name_prefix + '_q_network')
        mlflow.pytorch.log_model(self._target_q_network, name_prefix + '_target_q_network')

    def load_model(self, path: str):
        """
        Load model.

        :param path: Path to model (including the name prefix).
        """
        self._q_network = mlflow.pytorch.load_model(path + '_q_network', map_location=self._device)
        self._target_q_network = mlflow.pytorch.load_model(path + '_target_q_network', map_location=self._device)

    def get_weights(self) -> Dict[str, Dict[Any, Any]]:
        """
        Get model weights.

        :return: Dictionary of model weights.
        """
        return {
            'q_network': copy.deepcopy(self._q_network.state_dict()),
            'target_q_network': copy.deepcopy(self._target_q_network.state_dict())
        }

    def load_weights(self, model_state_dicts: Dict[str, Dict[Any, Any]]):
        """
        Load model weights.

        :param model_state_dicts: Dictionary of model weights.
        """
        if not all(key in model_state_dicts for key in ['q_network', 'target_q_network']):
            raise ValueError('weights dictionary missing required keys')
        self._q_network.load_state_dict(model_state_dicts['q_network'])
        self._target_q_network.load_state_dict(model_state_dicts['target_q_network'])

    def reset_prioritization_bias_correction_annealing(self, num_train_steps: int):
        """
        Reset bias correction annealing.
        :param num_train_steps: Number of training steps.
        """
        self._replay_buffer.reset_bias_annealing(num_train_steps)

    def eval(self):
        """
        Set agent to evaluation mode.
        """
        self._q_network.eval()
        self._target_q_network.eval()

    def train(self):
        """
        Set agent to training mode.
        """
        self._q_network.train()
        self._target_q_network.train()

    # endregion

    # region Private Functions/Methods

    def _compute_loss(self, q_val_preds: torch.FloatTensor, q_val_targets: torch.FloatTensor, is_weights: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Compute loss.

        :param q_val_preds: Predicted Q-values.
        :param q_val_targets: Target Q-values.
        :param is_weights: Importance sampling weights.
        :return: Tuple of Total Loss, PER Loss, Regularization, Absolute TD Error.
        """
        abs_td_error = (q_val_targets - q_val_preds).abs()
        td_error = (q_val_targets - q_val_preds).pow(2)
        per_loss = (td_error * is_weights).mean()
        # regularization term: penalize q-values greater than the max possible reward
        reg_term = (q_val_preds.abs() - self._reward_max).clamp_min(0.0).sum()
        # noinspection PyTypeChecker
        return per_loss + self._reg_lambda * reg_term, per_loss, reg_term, abs_td_error

    def _soft_update_target_network(self):
        """
        Soft update target network.
        """
        for target_param, param in zip(self._target_q_network.parameters(), self._q_network.parameters()):
            target_param.data.copy_(self._tau * param.data + (1.0 - self._tau) * target_param.data)

    # endregion
