import copy
import torch
import mlflow
from typing import Dict, Any

from mdp.action import Action
from agents.abstract_batch_agent import AbstractBatchAgent
from agents.iql.action_value_critic import ActionValueCritic
from agents.iql.expectile_value_critic import ExpectileValueCritic
from mdp.policies.next_best_action_policy import NextBestActionPolicy


class ImplicitQLearning(AbstractBatchAgent):
    """
    Adapted from the following implementations
    1. https://github.com/BY571/Implicit-Q-Learning/blob/main/agent.py (PyTorch)
    2. https://github.com/ikostrikov/implicit_q_learning/blob/master/learner.py (JAX)
    """

    def __init__(self, state_dim: int, action_dim: int, num_actions: int,
                 policy_hidden_dim: int, critic_hidden_dim: int, expectile_val_hidden_dim: int,
                 policy_lr: float, critic_lr: float, expectile_val_lr: float,
                 gamma: float, expectile: float = 0.8, temperature: float = 0.1, clip_norm: float = 1.0,
                 tau: float = 5e-3, weight_decay: float = 0.0):
        self._gamma = gamma
        self._expectile = expectile
        self._temperature = temperature
        self._clip_norm = clip_norm
        self._tau = tau
        # Policy
        policy = NextBestActionPolicy(state_dim, policy_hidden_dim, num_actions) # AbstractAgent will send this to device
        self._policy_optimizer = torch.optim.Adam(policy.parameters(), lr=policy_lr)
        super().__init__(policy)
        # Critic A and Critic A Target
        self._critic_a = ActionValueCritic(state_dim, critic_hidden_dim, action_dim).to(self._device)
        self._critic_a_target = ActionValueCritic(state_dim, critic_hidden_dim, action_dim).to(self._device)
        self._critic_a_target.load_state_dict(self._critic_a.state_dict())
        self._critic_a_optimizer = torch.optim.Adam(self._critic_a.parameters(), lr=critic_lr, weight_decay=weight_decay)
        # Critic B and Critic B Target
        self._critic_b = ActionValueCritic(state_dim, critic_hidden_dim, action_dim).to(self._device)
        self._critic_b_target = ActionValueCritic(state_dim, critic_hidden_dim, action_dim).to(self._device)
        self._critic_b_target.load_state_dict(self._critic_b.state_dict())
        self._critic_b_optimizer = torch.optim.Adam(self._critic_b.parameters(), lr=critic_lr, weight_decay=weight_decay)
        # Expectile Value Critic
        self._expectile_val = ExpectileValueCritic(state_dim, expectile_val_hidden_dim).to(self._device)
        self._expectile_val_optimizer = torch.optim.Adam(self._expectile_val.parameters(), lr=expectile_val_lr, weight_decay=weight_decay)

    # region Methods/Functions

    # region Public

    def compute_value(self, state: torch.FloatTensor) -> torch.FloatTensor:
        # Note: we could use AWR (see policy loss) to extract the value here, but that isn't realistic for high dimensional action spaces
        with torch.no_grad():
            return self._expectile_val(state)

    def train(self):
        self._policy.train()
        self._critic_a.train()
        self._critic_b.train()
        self._expectile_val.train()

    def eval(self):
        self._policy.eval()
        self._critic_a.eval()
        self._critic_b.eval()
        self._expectile_val.eval()

    def get_weights(self) -> Dict[str, Dict[Any, Any]]:
        return {
            'policy': copy.deepcopy(self._policy.state_dict()),
            'critic_a': copy.deepcopy(self._critic_a.state_dict()),
            'critic_b': copy.deepcopy(self._critic_b.state_dict()),
            'expectile_val': copy.deepcopy(self._expectile_val.state_dict())
        }

    def load_weights(self, model_state_dicts: Dict[str, Dict[Any, Any]]):
        if not all(key in model_state_dicts for key in ['policy', 'critic_a', 'critic_b', 'expectile_val']):
            raise ValueError('weights dictionary missing required keys')
        self._policy.load_state_dict(model_state_dicts['policy'])
        self._critic_a.load_state_dict(model_state_dicts['critic_a'])
        self._critic_b.load_state_dict(model_state_dicts['critic_b'])
        self._expectile_val.load_state_dict(model_state_dicts['expectile_val'])

    def save_model(self, path: str):
        mlflow.pytorch.log_model(self._policy, path + '_policy')
        mlflow.pytorch.log_model(self._critic_a, path + '_critic_a')
        mlflow.pytorch.log_model(self._critic_b, path + '_critic_b')
        mlflow.pytorch.log_model(self._expectile_val, path + '_expectile_val')

    def load_model(self, path: str):
        self._policy = mlflow.pytorch.load_model(path + '_policy')
        self._critic_a = mlflow.pytorch.load_model(path + '_critic_a')
        self._critic_b = mlflow.pytorch.load_model(path + '_critic_b')
        self._expectile_val = mlflow.pytorch.load_model(path + '_expectile_val')

    def compute_losses(self, states: torch.FloatTensor, actions_taken: torch.LongTensor, rewards: torch.FloatTensor, next_states: torch.FloatTensor, dones: torch.BoolTensor) -> Dict[str, float]:
        # copying from batch_update to compute losses
        if not (rewards.size(0) == actions_taken.size(0) and states.size(0) == next_states.size(0)):
            raise ValueError('unequal experience batch sizes')
        with torch.no_grad():
            q_a = self._critic_a_target(states, actions_taken)
            q_b = self._critic_b_target(states, actions_taken)
            min_q = torch.min(q_a, q_b)
            expectile_val = self._expectile_val(states)
            next_expectile_val = self._expectile_val(next_states)
            q_target = rewards + self._gamma * (1 - dones.long()) * next_expectile_val
            expectile_loss = self._compute_expectile_loss(min_q, states)
            policy_loss = self._compute_policy_loss(states, actions_taken, min_q, expectile_val)
            critic_a_loss = self._compute_critic_loss(self._critic_a, q_target, states, actions_taken)
            critic_b_loss = self._compute_critic_loss(self._critic_b, q_target, states, actions_taken)
        # ------------------ construct loss dictionary ------------------
        loss_dict = {
            'expectile_loss': expectile_loss.detach().cpu().item(),
            'policy_loss': policy_loss.detach().cpu().item(),
            'critic_a_loss': critic_a_loss.detach().cpu().item(),
            'critic_b_loss': critic_b_loss.detach().cpu().item()
        }
        return loss_dict

    # endregion

    # region Private

    def _compute_expectile_loss(self, min_q: torch.FloatTensor, states: torch.FloatTensor) -> torch.FloatTensor:
        expectile_val = self._expectile_val(states)
        expectile_diff = min_q - expectile_val
        weight = torch.where(expectile_diff > 0, self._expectile, 1 - self._expectile)
        expectile_loss = (weight * expectile_diff.pow(2)).mean()
        return expectile_loss

    def _compute_policy_loss(self, states: torch.FloatTensor, actions: torch.LongTensor, min_q: torch.FloatTensor, expectile_val: torch.FloatTensor) -> torch.FloatTensor:
        exp_a = ((min_q - expectile_val) * self._temperature).exp()
        exp_a = torch.min(exp_a, torch.tensor([100]).to(self._device))
        log_probs = self._get_log_probs(states, actions)
        policy_loss = -(exp_a * log_probs).mean()
        return policy_loss

    def _compute_critic_loss(self, critic: torch.nn.Module, q_target: torch.FloatTensor, states: torch.FloatTensor, actions: torch.LongTensor) -> torch.FloatTensor:
        q = critic(states, actions)
        critic_loss = (q - q_target).pow(2).mean()
        return critic_loss

    def _soft_update_critic(self, critic: torch.nn.Module, target: torch.nn.Module):
        for target_param, param in zip(target.parameters(), critic.parameters()):
            target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param.data)

    # endregion

    # endregion

    # region AbstractBatchAgent

    def batch_update(self, states: torch.FloatTensor, actions_taken: torch.LongTensor, rewards: torch.FloatTensor, next_states: torch.FloatTensor, dones: torch.BoolTensor) -> Dict[str, float]:
        if not (rewards.size(0) == actions_taken.size(0) and states.size(0) == next_states.size(0)):
            raise ValueError('unequal experience batch sizes')
        with torch.no_grad():
            q_a = self._critic_a_target(states, actions_taken)
            q_b = self._critic_b_target(states, actions_taken)
            min_q = torch.min(q_a, q_b)
            expectile_val = self._expectile_val(states)
            next_expectile_val = self._expectile_val(next_states)
        q_target = (rewards + self._gamma * (1 - dones.long()) * next_expectile_val.squeeze(-1)).unsqueeze(-1)
        # ------------------ update expectile value critic ------------------
        self._expectile_val_optimizer.zero_grad()
        expectile_loss = self._compute_expectile_loss(min_q, states)
        expectile_loss.backward()
        self._expectile_val_optimizer.step()
        # ------------------ update policy ------------------
        self._policy_optimizer.zero_grad()
        policy_loss = self._compute_policy_loss(states, actions_taken, min_q, expectile_val)
        policy_loss.backward()
        self._policy_optimizer.step()
        # ------------------ update critic a ------------------
        self._critic_a_optimizer.zero_grad()
        critic_a_loss = self._compute_critic_loss(self._critic_a, q_target, states, actions_taken)
        critic_a_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._critic_a.parameters(), self._clip_norm)
        self._critic_a_optimizer.step()
        # ------------------ update critic b ------------------
        self._critic_b_optimizer.zero_grad()
        critic_b_loss = self._compute_critic_loss(self._critic_b, q_target, states, actions_taken)
        critic_b_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._critic_b.parameters(), self._clip_norm)
        self._critic_b_optimizer.step()
        # ------------------ update target networks ------------------
        self._soft_update_critic(self._critic_a, self._critic_a_target)
        self._soft_update_critic(self._critic_b, self._critic_b_target)
        # ------------------ construct loss dictionary ------------------
        loss_dict = {
            'expectile_loss': expectile_loss.detach().cpu().item(),
            'policy_loss': policy_loss.detach().cpu().item(),
            'critic_a_loss': critic_a_loss.detach().cpu().item(),
            'critic_b_loss': critic_b_loss.detach().cpu().item()
        }
        return loss_dict

    # endregion
