import copy
import torch
import mlflow
import warnings
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Final, Dict, Any

from mdp.action import Action
from agents.replay_buffers.prioritized_replay import PrioritizedReplayBuffer
from agents.abstract_agent import AbstractAgent
from agents._ddpg.critic import Critic
from agents._ddpg.actor import Actor


class TD3(AbstractAgent):
    def __init__(self, gamma: float, actor_lr: float, critic_lr: float, tau: float, batch_size: int, buffer_size: int, state_dim: int, action_dim: int,
                 action_mins: Tuple[float, ...], action_maxs: Tuple[float, ...], hidden_dim: int, reward_max: float, reg_lambda: float, per_alpha: float,
                 per_beta: float, per_eps: float, actor_delay_update: int, action_loss_lambda: float):
        """
        Initialize D3QN agent.

        :param gamma: Discount factor
        :param actor_lr: Actor learning rate
        :param critic_lr: Critic learning rate
        :param tau: Target network update rate (soft update)
        :param batch_size: Batch size
        :param buffer_size: Replay buffer size
        :param state_dim: State dimension
        :param action_dim: Action dimension
        :param hidden_dim: Network hidden dimension
        :param action_mins: Minimum action values
        :param action_maxs: Maximum action values
        :param reward_max: Absolute maximum possible reward
        :param reg_lambda: Regularization lambda penalizing Q-values greater than the max possible reward
        :param per_alpha: Prioritization weight
        :param per_beta: Bias correction weight
        :param per_eps: Small constant to avoid zero priority
        :param actor_delay_update: Delay in updating the actor network
        :param action_loss_lambda: Weight for the action prediction loss
        """
        warnings.warn('The TD3 agents has an error in the batch_train method. Do not use this class until the error is fixed. See "# BROKEN" comment.')
        self._gamma = gamma
        self._tau = tau
        self._batch_size = batch_size
        # Actors
        self._actor = Actor(state_dim, action_dim, action_mins, action_maxs, hidden_dim)
        super().__init__(self._actor) # AbstractAgent will put the policy on the correct device
        self._actor_target = Actor(state_dim, action_dim, action_mins, action_maxs, hidden_dim,).to(self._device)
        self._actor_target.load_state_dict(self._actor.state_dict())
        self._actor_optimizer = torch.optim.Adam(self._actor.parameters(), lr=actor_lr)
        # Critics
        self._critic_a = Critic(state_dim, action_dim, hidden_dim).to(self._device)
        self._critic_target_a = Critic(state_dim, action_dim, hidden_dim).to(self._device)
        self._critic_target_a.load_state_dict(self._critic_a.state_dict())
        self._critic_optimizer_a = torch.optim.Adam(self._critic_a.parameters(), lr=critic_lr)
        self._critic_b = Critic(state_dim, action_dim, hidden_dim).to(self._device)
        self._critic_target_b = Critic(state_dim, action_dim, hidden_dim).to(self._device)
        self._critic_target_b.load_state_dict(self._critic_b.state_dict())
        self._critic_optimizer_b = torch.optim.Adam(self._critic_b.parameters(), lr=critic_lr)
        # Replay Buffer
        self._replay_buffer = PrioritizedReplayBuffer(state_dim, action_dim, buffer_size, per_alpha, per_beta, per_eps)
        # Constants
        self._reward_max: Final[float] = reward_max
        self._reg_lambda: Final[float] = reg_lambda
        self._actor_update_delay: Final[int] = actor_delay_update
        self._action_loss_lambda: Final[float] = action_loss_lambda
        # TODO: maybe add noise like they do in the original paper

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
            actions = self._actor(states)
            # NOTE: these probs will cause issues for OPE MAGIC - TODO: fix this
            probs = torch.ones_like(actions) # Q Learning is deterministic AND our action space is continuous for this agent, so the probability is 1 for action taken and all other actions have probability 0
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

    def batch_train(self, step: int, action_scaler: MinMaxScaler) -> Dict[str, float]:
        if self._replay_buffer.size < self._batch_size:
            raise ValueError('replay buffer does not have enough transitions to sample from')
        states, actions, rewards, next_states, dones, is_weights, transitions_indices = self._replay_buffer.sample(self._batch_size)
        states = states.to(self._device)
        actions = actions.to(self._device)
        rewards = rewards.to(self._device).unsqueeze(-1)
        next_states = next_states.to(self._device)
        dones = dones.to(self._device).long().unsqueeze(-1)
        is_weights = is_weights.to(self._device).unsqueeze(-1)
        # compute q value targets
        target_q_values_a = self._critic_target_a(next_states, self._actor_target(next_states))
        target_q_values_b = self._critic_target_b(next_states, self._actor_target(next_states))
        target_q_values = rewards + self._gamma * torch.min(target_q_values_a, target_q_values_b) * (1 - dones)
        # compute q value predictions
        q_values_a = self._critic_a(states, actions)
        q_values_b = self._critic_b(states, actions)
        # Compute TD Errors
        self._critic_optimizer_a.zero_grad()
        critic_a_total_loss, per_loss_a, reg_term_a, abs_td_error_a = self._compute_critic_loss(q_values_a, target_q_values, is_weights)
        critic_a_total_loss.backward(retain_graph=True) # NOTE: retain_graph=True bc we use target_q_values in both critic_a and critic_b loss calculations
        self._critic_optimizer_a.step()
        self._critic_optimizer_b.zero_grad()
        critic_b_total_loss, per_loss_b, reg_term_b, abs_td_error_b = self._compute_critic_loss(q_values_b, target_q_values, is_weights)
        critic_b_total_loss.backward()
        self._critic_optimizer_b.step()
        # Update priorities in replay buffer
        self._replay_buffer.update_priorities(transitions_indices, abs_td_error_a.squeeze(-1).detach().cpu())
        # Construct metric dict
        metric_dict = {
            'critic_a_total_loss': critic_a_total_loss.detach().cpu().item(),
            'per_loss_a': per_loss_a.detach().cpu().item(),
            'reg_term_a': reg_term_a.detach().cpu().item(),
            'abs_td_error_a': abs_td_error_a.mean().detach().cpu().item(),
            'critic_b_total_loss': critic_b_total_loss.detach().cpu().item(),
            'per_loss_b': per_loss_b.detach().cpu().item(),
            'reg_term_b': reg_term_b.detach().cpu().item(),
            'abs_td_error_b': abs_td_error_b.mean().detach().cpu().item(),
        }
        update_actor = (step + 1) % self._actor_update_delay == 0
        if update_actor:
            # Compute actor loss
            self._actor_optimizer.zero_grad()
            action_preds = self._actor(states)
            # Original TD3 actor_loss = -self._critic_a(states, action_preds).mean()
            q_val_loss = -torch.min(self._critic_a(states, action_preds), self._critic_b(states, action_preds)).mean()
            # NOTE: we compare the current actions with the previous actions as they propose in the original paper. This seems weird...
            valid_next_state_mask = dones.logical_not().squeeze(-1)
            valid_next_states = next_states[valid_next_state_mask]
            next_action_preds = self._actor_target(valid_next_states) # BROKEN: we should be comparing action_preds with actions
            valid_next_states_clinicial_actions = actions[valid_next_state_mask]
            action_pred_loss = torch.nn.functional.mse_loss(next_action_preds, valid_next_states_clinicial_actions)
            actor_loss = q_val_loss + self._action_loss_lambda * action_pred_loss
            actor_loss.backward()
            self._actor_optimizer.step()
            # Mean Action
            action_preds = action_preds.detach().cpu().numpy()
            action_preds = action_scaler.inverse_transform(action_preds).mean(0)
            mean_iv_fluid_action = action_preds[0]
            mean_vp_action = action_preds[1]
            # Update metric dict
            metric_dict['actor_loss'] = actor_loss.detach().cpu().item()
            metric_dict['mean_iv_fluid_action'] = mean_iv_fluid_action
            metric_dict['mean_vp_action'] = mean_vp_action
        # Soft update target network
        self._soft_update_target_networks(update_actor)
        # Increment training step
        self._replay_buffer.increment_step()
        return metric_dict

    def evaluate(self, states: torch.FloatTensor, actions: torch.LongTensor, rewards: torch.FloatTensor, next_states: torch.FloatTensor,
                 dones: torch.BoolTensor) -> Dict[str, float]:
        # NOTE: we could refactor and cleanup this dupe code...
        states = states.to(self._device)
        actions = actions.to(self._device)
        rewards = rewards.to(self._device).unsqueeze(-1)
        next_states = next_states.to(self._device)
        dones = dones.to(self._device).long().unsqueeze(-1)
        # compute q value targets
        target_q_values_a = self._critic_target_a(next_states, self._actor_target(next_states))
        target_q_values_b = self._critic_target_b(next_states, self._actor_target(next_states))
        target_q_values = rewards + self._gamma * torch.min(target_q_values_a, target_q_values_b) * (1 - dones)
        # compute q value predictions
        q_values = self._critic_a(states, actions)
        # Compute TD Errors
        abs_td_error = (target_q_values - q_values).abs()
        td_error = (target_q_values - q_values).pow(2)
        # Compute mean actions
        action_preds = self._actor(states)
        mean_iv_fluid_action = action_preds[:, 0].detach().mean().cpu().item()
        mean_vp_action = action_preds[:, 1].detach().mean().cpu().item()
        return {
            'td_error': td_error.sum().detach().cpu().item(),
            'abs_td_error': abs_td_error.sum().detach().cpu().item(),
            'mean_iv_fluid_action': mean_iv_fluid_action,
            'mean_vp_action': mean_vp_action
        }

    def fill_replay_buffer(self, dataloader: torch.utils.data.DataLoader):
        """
        Fill replay buffer with data for offline RL.

        :param dataloader: Offline RL DataLoader.
        """
        self._replay_buffer.fill(dataloader, discrete_actions=False)

    def save_model(self, name_prefix: str):
        """
        Save model.

        :param name_prefix: Prefix for model name.
        """
        mlflow.pytorch.log_model(self._actor, name_prefix + '_actor')
        mlflow.pytorch.log_model(self._actor_target, name_prefix + '_actor_target')
        mlflow.pytorch.log_model(self._critic_a, name_prefix + '_critic_a')
        mlflow.pytorch.log_model(self._critic_target_a, name_prefix + '_critic_target_a')
        mlflow.pytorch.log_model(self._critic_b, name_prefix + '_critic_b')
        mlflow.pytorch.log_model(self._critic_target_b, name_prefix + '_critic_target_b')

    def load_model(self, path: str):
        """
        Load model.

        :param path: Path to model (including the name prefix).
        """
        self._actor = mlflow.pytorch.load_model(path + '_actor', map_location=self._device)
        self._actor_target = mlflow.pytorch.load_model(path + '_actor_target', map_location=self._device)
        self._critic_a = mlflow.pytorch.load_model(path + '_critic_a', map_location=self._device)
        self._critic_target_a = mlflow.pytorch.load_model(path + '_critic_target_a', map_location=self._device)
        self._critic_b = mlflow.pytorch.load_model(path + '_critic_b', map_location=self._device)
        self._critic_target_b = mlflow.pytorch.load_model(path + '_critic_target_b', map_location=self._device)

    def get_weights(self) -> Dict[str, Dict[Any, Any]]:
        """
        Get model weights.

        :return: Dictionary of model weights.
        """
        return {
            'actor': copy.deepcopy(self._actor.state_dict()),
            'actor_target': copy.deepcopy(self._actor_target.state_dict()),
            'critic_a': copy.deepcopy(self._critic_a.state_dict()),
            'critic_target_a': copy.deepcopy(self._critic_target_a.state_dict()),
            'critic_b': copy.deepcopy(self._critic_b.state_dict()),
            'critic_target_b': copy.deepcopy(self._critic_target_b.state_dict())
        }

    def load_weights(self, model_state_dicts: Dict[str, Dict[Any, Any]]):
        """
        Load model weights.

        :param model_state_dicts: Dictionary of model weights.
        """
        if not all(key in model_state_dicts for key in ['actor', 'actor_target', 'critic_a', 'critic_target_a', 'critic_b', 'critic_target_b']):
            raise ValueError('weights dictionary missing required keys')
        self._actor.load_state_dict(model_state_dicts['actor'])
        self._actor_target.load_state_dict(model_state_dicts['actor_target'])
        self._critic_a.load_state_dict(model_state_dicts['critic_a'])
        self._critic_target_a.load_state_dict(model_state_dicts['critic_target_a'])
        self._critic_b.load_state_dict(model_state_dicts['critic_b'])
        self._critic_target_b.load_state_dict(model_state_dicts['critic_target_b'])

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
        self._actor.eval()
        self._actor_target.eval()
        self._critic_a.eval()
        self._critic_target_a.eval()
        self._critic_b.eval()
        self._critic_target_b.eval()

    def train(self):
        """
        Set agent to training mode.
        """
        self._actor.train()
        self._actor_target.train()
        self._critic_a.train()
        self._critic_target_a.train()
        self._critic_b.train()
        self._critic_target_b.train()

    # endregion

    # region Private Functions/Methods

    def _compute_critic_loss(self, q_val_preds: torch.FloatTensor, q_val_targets: torch.FloatTensor, is_weights: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
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

    def _soft_update_target_networks(self, update_actor: bool):
        """
        Soft update target networks.
        """
        network_pairs = [(self._critic_target_a, self._critic_a), (self._critic_target_b, self._critic_b)]
        if update_actor:
            network_pairs.append((self._actor_target, self._actor))
        for target_net, net in network_pairs:
            for target_param, param in zip(target_net.parameters(), net.parameters()):
                target_param.data.copy_(self._tau * param.data + (1.0 - self._tau) * target_param.data)

    # endregion
