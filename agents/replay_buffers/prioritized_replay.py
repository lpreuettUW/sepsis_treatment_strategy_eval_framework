import torch
from typing import Final, Tuple, Optional


class PrioritizedReplayBuffer:
    """
    Prioritized replay buffer proposed by Schaul et al. (2015).
    Adapted from the following implementations:
    - https://github.com/aniruddhraghu/sepsisrl/blob/master/continuous/q_network.ipynb
    - https://github.com/philtabor/Deep-Q-Learning-Paper-To-Code/blob/master/DuelingDDQN/replay_memory.py
    """
    def __init__(self, state_dim: int, action_dim: int, buffer_size: int, alpha: float, beta: float, eps: float):
        """
        Initialize Prioritized Replay Buffer.

        :param state_dim: State dimension
        :param action_dim: Action dimension
        :param buffer_size: Replay buffer size
        :param alpha: Prioritization weight
        :param beta: Bias correction weight
        :param eps: Small constant to avoid zero priority
        """
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._buffer_size = buffer_size
        self._alpha: Final[float] = alpha
        self._beta_start: Final[float] = beta
        self._eps: Final[float] = eps
        self._buffered_states = torch.zeros((buffer_size, state_dim), dtype=torch.float32)
        self._buffered_actions = torch.zeros((buffer_size, action_dim), dtype=torch.long)
        self._buffered_rewards = torch.zeros((buffer_size,), dtype=torch.float32)
        self._buffered_next_states = torch.zeros((buffer_size, state_dim), dtype=torch.float32)
        self._buffered_dones = torch.zeros((buffer_size,), dtype=torch.bool)
        self._priorities = torch.zeros((buffer_size,), dtype=torch.float32)
        self._max_priority: Final[float] = 1.0
        self._index = self._current_size = self._current_step = 0
        self._num_train_steps = None

    def add(self, state: torch.FloatTensor, action: torch.FloatTensor, reward: torch.FloatTensor, next_state: torch.FloatTensor, done: bool):
        """
        Add transition to buffer with maximum priority.

        :param state: State tensor
        :param action: Action tensor
        :param reward: Reward tensor
        :param next_state: Next state tensor
        :param done: Done flag
        """
        self._buffered_states[self._index] = state
        self._buffered_actions[self._index] = action
        self._buffered_rewards[self._index] = reward
        self._buffered_next_states[self._index] = next_state
        self._buffered_dones[self._index] = done
        self._priorities[self._index] = self._max_priority
        self._index = (self._index + 1) % self._buffer_size
        self._current_size = min(self._current_size + 1, self._buffer_size)

    def sample(self, batch_size: int) -> Tuple[torch.FloatTensor, torch.LongTensor, torch.FloatTensor, torch.FloatTensor,
                                               torch.BoolTensor, torch.FloatTensor, torch.LongTensor]:
        """
        Sample transitions from buffer.

        :param batch_size: Batch size
        :return: Tuple of Sampled transitions and indices (used for updating priorities)
        """
        if self._current_size < batch_size:
            raise ValueError('replay buffer does not have enough transitions to sample from')
        indices = self._priorities[:self._current_size].multinomial(batch_size, replacement=False)
        states = self._buffered_states[indices]
        actions = self._buffered_actions[indices]
        rewards = self._buffered_rewards[indices]
        next_states = self._buffered_next_states[indices]
        dones = self._buffered_dones[indices]
        is_weights = self._compute_importance_sampling_weights(indices)
        # noinspection PyTypeChecker
        return states, actions, rewards, next_states, dones, is_weights, indices

    def update_priorities(self, indices: torch.LongTensor, abs_td_error: torch.FloatTensor):
        """
        Compute new priorities from observed TD errors.

        :param indices: Indices of transitions
        :param abs_td_error: Absolute TD error
        """
        # update priorities - clipped to avoid large priorities and priorities of zero
        self._priorities[indices] = (abs_td_error.pow(self._alpha) / self._priorities[:self._current_size].max()).clamp(self._eps, self._max_priority)

    def fill(self, dataloader: torch.utils.data.DataLoader, discrete_actions: bool):
        """
        Resize and fill replay buffer with data for offline RL.

        :param dataloader: Offline RL DataLoader.
        :param discrete_actions: Discrete action space flag
        """
        # resize buffer
        self._buffer_size = len(dataloader) * dataloader.batch_size # NOTE: this is not going to be exact
        self._current_size = self._index = 0
        self._buffered_states = torch.zeros((self._buffer_size, self._buffered_states.size(1)), dtype=torch.float32)
        self._buffered_actions = torch.zeros((self._buffer_size, self._buffered_actions.size(1)), dtype=(torch.long if discrete_actions else torch.float32))
        self._buffered_rewards = torch.zeros((self._buffer_size,), dtype=torch.float32)
        self._buffered_next_states = torch.zeros((self._buffer_size, self._buffered_next_states.size(1)), dtype=torch.float32)
        self._buffered_dones = torch.zeros((self._buffer_size,), dtype=torch.bool)
        self._priorities = torch.full((self._buffer_size,), self._max_priority, dtype=torch.float32)
        # extract data
        for i, (state, action, next_state, reward, done) in enumerate(dataloader):
            start_idx = i * dataloader.batch_size
            stop_idx = start_idx + state.size(0)
            self._buffered_states[start_idx:stop_idx] = state
            if discrete_actions:
                self._buffered_actions[start_idx:stop_idx, 0] = action # NOTE: this works bc action dim is 1
            else:
                self._buffered_actions[start_idx:stop_idx] = action
            self._buffered_next_states[start_idx:stop_idx] = next_state
            self._buffered_rewards[start_idx:stop_idx] = reward
            self._buffered_dones[start_idx:stop_idx] = done
            self._index += state.size(0)
            self._current_size += state.size(0)

    def reset_bias_annealing(self, num_train_steps: int):
        """
        Reset bias annealing.

        :param num_train_steps: Number of training steps
        """
        self._num_train_steps = num_train_steps
        self._current_step = 0

    def increment_step(self):
        """
        Increment current step.
        """
        self._current_step += 1

    def _compute_importance_sampling_weights(self, indices: torch.LongTensor) -> torch.FloatTensor:
        """
        Compute importance sampling weights for prioritized replay buffer.

        :param indices: Indices of transitions to compute weights for
        :return: Importance sampling weights
        """
        # compute all importance sampling weights then select the ones we need - we do this to scale the weights by the max
        is_weights = (self._current_size * self._priorities).pow(-self._beta) # ((1 / N) * (1 / P(i)))^(beta)
        is_weights /= is_weights.max() # scale by max for stability - ensure we only scale loss down
        is_weights = is_weights[indices] # select the ones we need
        return is_weights

    @property
    def size(self) -> int:
        """
        :return: Number of transitions in the buffer
        """
        return self._current_size

    @property
    def _beta(self) -> float:
        """
        :return: Bias correction weight for current step
        """
        if self._num_train_steps is None:
            return self._beta_start
        else:
            return self._beta_start + (1.0 - self._beta_start) * (self._current_step / self._num_train_steps)
