import os
import mlflow
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

from mdp.mimic_iii.action_spaces.discrete import Discrete as MimicIIIDiscreteActionSpace
from mdp.mimic_iii.reward_functions.factory import Factory as MimicIIIRewardFunctionFactory
from utilities import mimic_iii_funcs
from ope.behavior_policies.kmeans_sarsa import KMeansSarsa

if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--gpu', type=str, default='-1', help='specify the GPU to use')
    args = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # parameters
    kmeans_n_clusters = 1250
    kmeans_n_init = 5
    kmeans_max_iter = 300
    sarsa_gamma = 1.0
    sarsa_alpha = 0.1
    sarsa_num_episodes = 250000
    sarsa_convergence_theta = 1e-6
    reward_fn_name = 'sparse'
    reward_scale = 15.0 if reward_fn_name == 'sparse' else None
    peng_reward_fn_runid = ''
    num_splits = 10
    reward_fn = MimicIIIRewardFunctionFactory.create(reward_fn_name)

    mlflow_path = os.path.join('file:///', 'Users', 'larry', 'Documents', 'UWT', 'Thesis Work', 'rec_sys', 'models', 'mimic-iii_reward_fn_eval', 'vm', 'mlruns')
    mlflow.set_tracking_uri(mlflow_path)
    experiment_name = 'KMeans SARSA Behavior Policy Retrospective: sparse - split 5'
    run_name = 'run 0'
    mlflow_experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = mlflow.create_experiment(experiment_name) if mlflow_experiment is None else mlflow_experiment.experiment_id

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as mlflow_run:
        # log parameters
        param_dict = {
            'kmeans_n_clusters': kmeans_n_clusters,
            'kmeans_n_init': kmeans_n_init,
            'kmeans_max_iter': kmeans_max_iter,
            'sarsa_gamma': sarsa_gamma,
            'sarsa_alpha': sarsa_alpha,
            'sarsa_num_episodes': sarsa_num_episodes,
            'sarsa_convergence_theta': sarsa_convergence_theta,
            'reward_fn_name': reward_fn_name,
            'num_splits': num_splits
        }
        mlflow.log_params(param_dict)
        splits = [5]  # range(num_splits)
        for split in tqdm(range(num_splits), desc='Split', total=num_splits, unit='split ', colour='blue', position=0, leave=True):
            if reward_fn_name == 'peng':
                if not peng_reward_fn_runid:
                    raise RuntimeError('Peng reward function runid must be specified for Peng reward function')
                reward_fn.load(f'runs:/{peng_reward_fn_runid}/split_{split}')
            # load data
            mimic_df = mimic_iii_funcs.load_standardized_mimic_data(split, reward_fn.raw_data_columns)
            if reward_fn_name != 'peng':
                reward_fn.update_rewards(mimic_df)  # scale rewards
                mimic_df = mimic_iii_funcs.drop_all_raw_data_columns(mimic_df)
            mimic_df, _, _ = mimic_iii_funcs.preprocess_mimic_data(mimic_df, 'kmeans_sarsa')
            if reward_fn_name == 'peng':
                reward_fn.update_rewards(mimic_df)  # scale rewards
            # NOTE: we train using the test dataset - we are learning the behavior policy to evaluate learned policies (using OPE)
            _, _, test_df = mimic_iii_funcs.train_val_test_split_mimic_data(mimic_df, 'kmeans_sarsa')
            # initialize behavior policy learner
            behavior_policy = KMeansSarsa(n_clusters=kmeans_n_clusters, n_actions=len(MimicIIIDiscreteActionSpace), num_episodes=sarsa_num_episodes,
                                          reward_scale=reward_scale, gamma=sarsa_gamma, alpha=sarsa_alpha, n_init=kmeans_n_init, max_iter=kmeans_max_iter)
            behavior_policy.learn_state_space(np.vstack(test_df['state'].to_numpy()).astype(np.float32))
            # cache discrete states
            test_df['discrete_state'] = behavior_policy.discretize_states(np.vstack(test_df['state'].to_numpy()).astype(np.float32))
            test_df['discrete_next_state'] = behavior_policy.discretize_states(np.vstack(test_df['next_state'].to_numpy()).astype(np.float32))
            # learn Q-values
            num_train_eps = behavior_policy.learn_q_values(test_df, sarsa_convergence_theta, metric_prefix=f'split_{split}')
            mlflow.log_metric(f'num_train_eps', num_train_eps, step=split)
            # evaluate model
            mean_td_error = behavior_policy.evaluate(test_df)
            mlflow.log_metric(f'test_mean_td_error', mean_td_error, step=split)
            # save model
            behavior_policy.save_model(name_prefix=f'split_{split}', save_sarsa=True, save_kmeans=True)
