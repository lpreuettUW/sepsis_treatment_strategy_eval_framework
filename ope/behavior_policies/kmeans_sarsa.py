import os
import mlflow
import tempfile
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Optional
from sklearn.cluster import KMeans


class KMeansSarsa:
    """
    Adapted from Raghu et al.'s 2017 implementation
    - https://github.com/aniruddhraghu/sepsisrl/blob/master/preprocessing/clustering_final_processing.ipynb
    - https://github.com/aniruddhraghu/sepsisrl/blob/master/discrete/sarsa_episodic.ipynb
    """
    _Dead_State_Delta = 0
    _Alive_State_Delta = 1

    def __init__(self, n_clusters: int, n_actions: int, num_episodes: int, reward_scale: Optional[float], gamma: float = 1.0, alpha: float = 0.1, n_init: int = 5, max_iter: int = 300, verbose: bool = True):
        """
        Cluster states using KMeans and learn Q-values using SARSA
        :param n_clusters: Number of KMeans clusters
        :param n_actions: Number of actions
        :param num_episodes: Number of episodes to train the SARSA model
        :param reward_scale: Scaling factor for terminal state rewards (NOTE: we assume sparse rewards)
        :param gamma: SARSA discount factor
        :param alpha: SARSA Learning rate
        :param n_init: Number of times the KMeans algorithm will be run with different centroid seeds
        :param max_iter: Maximum number of iterations of the KMeans algorithm for a single run
        :param verbose: Verbosity of KMeans
        """
        self._n_clusters = n_clusters
        self._n_actions = n_actions
        self._num_episodes = num_episodes
        self._gamma = gamma
        self._alpha = alpha
        self._kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter, verbose=verbose)
        self._Q = np.zeros((n_clusters + 2, n_actions)) # plus two for Dead and Alive absorbing states
        if reward_scale is not None:
            # set Dead and Alive state values - we know these values a priori (NOTE: we assume sparse rewards)
            self._Q[n_clusters + KMeansSarsa._Dead_State_Delta, :] = -abs(reward_scale) # Dead state
            self._Q[n_clusters + KMeansSarsa._Alive_State_Delta, :] = abs(reward_scale) # Alive state
        self._Q_star = None # optimal actions: shape (n_clusters + 2,)

    def learn_state_space(self, continuous_states: np.ndarray): # TODO: use the two absorbing states...
        self._kmeans.fit(continuous_states)

    def learn_q_values(self, df: pd.DataFrame, convergence_theta: float, metric_prefix: str, log_interval: int = 1000) -> int: # TODO: use the two absorbing states...
        df = self._prepare_data_for_sarsa(df)
        unique_traj_ids = df['traj'].unique()
        eps_with_no_change = ep = 0
        for ep in tqdm(range(self._num_episodes), desc='Episode', total=self._num_episodes, unit='episode ', colour='green', position=0, leave=True):
            delta = 0
            # select random episode
            traj_id = np.random.choice(unique_traj_ids)
            # ensure trajectory is sorted by step
            traj_df = df[df['traj'] == traj_id].sort_values(by=['step'], axis=0)
            assert traj_df['discrete_state'].iloc[-1] == traj_df['discrete_next_state'].iloc[-2], 'Preprocess Error: Last state in trajectory must be the second to last next state in the trajectory'
            # play episode
            for i in range(traj_df.shape[0]):
                # extract current state, action, reward, next state, and done
                state = traj_df.iloc[i]['discrete_state']
                action = traj_df.iloc[i]['a:action']
                reward = traj_df.iloc[i]['r:reward']
                next_state = traj_df.iloc[i]['discrete_next_state']
                done = traj_df.iloc[i]['done']
                next_action = 0 if done else traj_df.iloc[i+1]['a:action']
                # update Q-values
                td_error = (reward - self._Q[state, action]) if done else reward + self._gamma * self._Q[next_state, next_action] - self._Q[state, action]
                if td_error > delta:
                    delta = td_error
                self._Q[state, action] += self._alpha * td_error
            if ep % log_interval == 0:
                mlflow.log_metrics({
                    f'{metric_prefix}_delta': delta,
                    f'{metric_prefix}_net_q_val': self._Q.sum()
                }, step=ep)
            if delta < convergence_theta:
                eps_with_no_change += 1
                if eps_with_no_change >= 10:
                    print('Convergence detected. Exiting training loop.')
                    break
            else:
                eps_with_no_change = 0
        self._Q_star = self._extract_q_star()
        return ep # return number of episodes trained

    def evaluate(self, df: pd.DataFrame) -> float:
        df = self._prepare_data_for_sarsa(df)
        unique_traj_ids = df['traj'].unique()
        total_td_error = 0
        total_steps = 0
        for traj_id in tqdm(unique_traj_ids, desc='Evaluation', total=len(unique_traj_ids), unit='trajectory ', colour='blue', position=0, leave=True):
            traj_df = df[df['traj'] == traj_id].sort_values(by=['step'], axis=0)
            assert traj_df['discrete_state'].iloc[-1] == traj_df['discrete_next_state'].iloc[-2], 'Preprocess Error: Last state in trajectory must be the second to last next state in the trajectory'
            for i in range(traj_df.shape[0]):
                state = traj_df.iloc[i]['discrete_state']
                action = traj_df.iloc[i]['a:action']
                reward = traj_df.iloc[i]['r:reward']
                next_state = traj_df.iloc[i]['discrete_next_state']
                done = traj_df.iloc[i]['done']
                next_action = 0 if done else traj_df.iloc[i + 1]['a:action']
                # compute TD error
                td_error = (reward - self._Q[state, action]) if done else reward + self._gamma * self._Q[next_state, next_action] - self._Q[state, action]
                total_td_error += td_error
                total_steps += 1 # we could extract this from the dataframe
        assert total_steps == df.shape[0], 'Total steps do not match DataFrame size'
        return total_td_error / df.shape[0]

    def discretize_states(self, continuous_states: np.ndarray) -> np.ndarray: # TODO: use the two absorbing states...
        return self._kmeans.predict(continuous_states)

    def get_actions(self, discrete_states: np.ndarray) -> np.ndarray:
        return self._Q_star[discrete_states]

    def get_q_values(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        return self._Q[states, actions]

    def save_model(self, name_prefix: str, save_sarsa: bool = True, save_kmeans: bool = True):
        if save_sarsa:
            with tempfile.NamedTemporaryFile() as f:
                np.save(f, self._Q)
                mlflow.log_artifact(f.name, name_prefix + '_sarsa')
        if save_kmeans:
            mlflow.sklearn.log_model(self._kmeans, name_prefix + '_kmeans')

    def load_model(self, run_id: str, name_prefix: str, load_sarsa: bool = True, load_kmeans: bool = True):
        run = mlflow.get_run(run_id)
        base_path = os.path.join(run.info.artifact_uri[7:], name_prefix) # cut out 'file://'
        if load_sarsa:
            sarsa_base_path = base_path + '_sarsa'
            # NOTE: we assume there is only one file in the directory AND if it doesn't exist, FileNotFoundError will be thrown
            self._Q = np.load(os.path.join(sarsa_base_path, next(os.scandir(sarsa_base_path)).name))
            self._Q_star = self._extract_q_star()
        if load_kmeans:
            kmeans_base_path = base_path + '_kmeans'
            self._kmeans = mlflow.sklearn.load_model(kmeans_base_path)
            self._n_clusters = self._kmeans.cluster_centers_.shape[0]

    def _extract_q_star(self) -> np.ndarray:
        return np.argmax(self._Q, axis=1)

    def _prepare_data_for_sarsa(self, df: pd.DataFrame) -> pd.DataFrame:
        """ We need to update next states with Dead and Alive states"""
        df = df.copy()
        df.drop(columns=['state', 'next_state'], inplace=True)
        df.loc[df['done'], 'discrete_next_state'] = df.loc[df['done'], :].apply(lambda row: self._n_clusters + (KMeansSarsa._Alive_State_Delta if row['r:reward'] > 0 else KMeansSarsa._Dead_State_Delta),
                                                                                axis=1)
        # add extra step to each trajectory to account for the terminal states
        done_steps_view = df.loc[df['done']]
        terminal_step_data_dict = {
            'step': done_steps_view['traj_length'].tolist(),
            'traj': done_steps_view['traj'].tolist(),
            'traj_length': done_steps_view['traj_length'].tolist(),
            'discrete_state': done_steps_view['discrete_next_state'].tolist(),
            'discrete_next_state': done_steps_view['discrete_next_state'].tolist(),
            'a:action': [0] * done_steps_view.shape[0], # action doesn't matter - all values are the same in the terminal state (and we cant take actions in a terminal state)
            'r:reward': done_steps_view['r:reward'].tolist(),
            'done': done_steps_view['done'].tolist()
        }
        # remove done flag from original trajectories
        df['done'] = False
        # remove rewards from original trajectories (NOTE: we assume sparse rewards)
        df['r:reward'] = 0.0
        df = pd.concat((df, pd.DataFrame.from_dict(terminal_step_data_dict)), ignore_index=True)
        df.loc[:, 'traj_length'] += 1 # update for extra timestep
        assert not df.isna().any().any(), 'NaN values found in DataFrame'
        return df
