import os
import torch
import mlflow
from tqdm.auto import tqdm
from argparse import ArgumentParser

from ope.behavior_policies.kmeans_sarsa import KMeansSarsa
from ope.fqe import FittedQEvaluation
from ope.magic import MAGIC
from agents.d3qn import D3QN
from agents.implicit_q_learning import ImplicitQLearning
from agents.td3 import TD3
from mdp.mimic_iii.action_spaces.discrete import Discrete as MimicIIIDiscreteActionSpace
from mdp.mimic_iii.reward_functions.factory import Factory as MimicIIIRewardFunctionFactory
from utilities.device_manager import DeviceManager
from utilities import mimic_iii_funcs
from utilities.ope_trajectory_dataset import OPETrajectoryDataset


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--gpu', type=str, default='-1', help='specify the GPU to use')
    args = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    torch.autograd.set_detect_anomaly(True)

    device = DeviceManager.get_device()

    # parameters
    state_space_runid = 'd5233d8c966846b0b01406a162c26fe7'
    behavior_policy_runid = '97629e709d7f45ec851200c6b0d497df'
    iql_runid = ''
    d3qn_runid = '5944007810ef4e318106533adbd6ccb6'
    td3_runid = ''
    reward_fn_name = 'sparse'
    peng_reward_fn_runid = ''
    num_splits = 10
    gamma = 0.99
    # fqe
    fqe_k_itrs = 100
    fqe_convergence_eps = 1e-3
    fqe_max_train_itrs = 25
    fqe_lr = 1e-4
    fqe_hidden_size = 32
    fqe_batch_size = 8192
    fqe_use_behavior_state = True
    fqe_run_id = ''
    # magic
    magic_j_steps = {float('inf'), -1, 3, 5, 7, 10}
    magic_k_conf_iters = 2000
    magic_batch_size = 1024
    magic_eps = 1e-6
    # iql settings
    state_space_dim = 200
    policy_lr = 1e-4
    critic_lr = 1e-4
    expectile_val_lr = 1e-4
    policy_hidden_size = 256
    critic_hidden_size = 256
    expectile_hidden_size = 256
    agent_weight_decay = 0.0
    expectile = 0.8
    temperature = 0.1
    clip_norm = 1.0
    tau = 5e-3
    # d3qn
    d3qn_policy_lr = 1e-4
    d3qn_tau = 5e-3
    d3qn_per_alpha = 0.6
    d3qn_per_beta = 0.9
    d3qn_per_eps = 1e-2
    d3qn_reg_lambda = 5.0
    d3qn_hidden_size = 128
    # td3
    td3_actor_lr = 3e-3
    td3_critic_lr = 3e-5
    td3_tau = 5e-3
    td3_per_alpha = 0.6
    td3_per_beta = 0.9
    td3_per_eps = 1e-2
    td3_hidden_size = 32
    td3_reg_lambda = 5.0  # regularization lambda penalizing Q-values greater than the max possible reward
    td3_actor_delay_update = 2
    td3_action_loss_lambda = 100.0
    # create reward function
    reward_fn = MimicIIIRewardFunctionFactory.create(reward_fn_name)

    mlflow_path = os.path.join('file:///', '<your_base_path>', 'mimic-iii', 'mlruns')
    mlflow.set_tracking_uri(mlflow_path)
    experiment_name = 'MAGIC Retrospective'
    run_name = f'D3QN - Reward Function: {reward_fn_name}'
    mlflow_experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = mlflow.create_experiment(experiment_name) if mlflow_experiment is None else mlflow_experiment.experiment_id

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as mlflow_run:
        param_dict = {
            'fqe_k_itrs': fqe_k_itrs,
            'fqe_convergence_eps': fqe_convergence_eps,
            'fqe_max_train_itrs': fqe_max_train_itrs,
            'fqe_lr': fqe_lr,
            'fqe_hidden_size': fqe_hidden_size,
            'fqe_batch_size': fqe_batch_size,
            'fqe_use_behavior_state': fqe_use_behavior_state,
            'gamma': gamma,
            'reward_fn_name': reward_fn_name,
            'peng_reward_fn_runid': peng_reward_fn_runid,
            'num_splits': num_splits,
            'behavior_policy_runid': behavior_policy_runid,
            'state_space_runid': state_space_runid,
            'd3qn_runid': d3qn_runid,
            'iql_runid': iql_runid,
            'fqe_run_id': fqe_run_id,
            'magic_j_steps': magic_j_steps,
            'magic_k_conf_iters': magic_k_conf_iters,
            'magic_batch_size': magic_batch_size,
            'magic_eps': magic_eps,
        }
        mlflow.log_params(param_dict)
        control_variates = None
        splits = range(num_splits) # [5]
        for split in tqdm(splits, desc='Split', total=num_splits, unit='split ', colour='green', position=0, leave=True):
            if reward_fn_name == 'peng':
                if not peng_reward_fn_runid:
                    raise RuntimeError('Peng reward function runid must be specified for Peng reward function')
                reward_fn.load(f'runs:/{peng_reward_fn_runid}/split_{split}')
            # load and preprocess mimic data
            mimic_df = mimic_iii_funcs.load_standardized_mimic_data(split, reward_fn.raw_data_columns)
            if reward_fn_name != 'peng':
                reward_fn.update_rewards(mimic_df)  # scale rewards
                mimic_df = mimic_iii_funcs.drop_all_raw_data_columns(mimic_df)
            mimic_df, min_max_scaler, continuous_action_scaler = mimic_iii_funcs.preprocess_mimic_data(mimic_df, 'ope')
            if reward_fn_name == 'peng':
                reward_fn.update_rewards(mimic_df)  # scale rewards
            max_reward = mimic_df['r:reward'].abs().max()
            action_mins, action_maxs = mimic_iii_funcs.get_cont_action_space_bounds(mimic_df)  # NOTE: IV Fluids and Vasopressors Doses, respectively
            (
                (raw_train_states, normed_train_states, train_discrete_actions, train_cont_actions, raw_train_next_states, normed_train_next_states, train_rewards, train_dones, train_missing_data_mask),
                (raw_val_states, normed_val_states, val_discrete_actions, val_cont_actions, raw_val_next_states, normed_val_next_states, val_rewards, val_dones, val_missing_data_mask),
                (raw_test_states, normed_test_states, test_discrete_actions, test_cont_actions, raw_test_next_states, normed_test_next_states, test_rewards, test_dones, test_missing_data_mask)
            ) = mimic_iii_funcs.train_val_test_split_mimic_data(mimic_df, 'ope')
            # load behavior policy
            behavior_policy = KMeansSarsa(n_clusters=8, n_actions=len(MimicIIIDiscreteActionSpace), num_episodes=1, reward_scale=15.0) # NOTE: n_clusters gets updated in load_model
            behavior_policy.load_model(behavior_policy_runid, f'split_{split}')
            # discretize states
            discretized_test_states = torch.from_numpy(behavior_policy.discretize_states(raw_test_states.view(-1, raw_test_states.size(-1)).numpy())).reshape(raw_test_states.size()[:-1]).float().unsqueeze(-1)
            discretized_test_next_states = torch.from_numpy(behavior_policy.discretize_states(raw_test_next_states.view(-1, raw_test_next_states.size(-1)).numpy())).reshape(raw_test_next_states.size()[:-1]).float().unsqueeze(-1)
            # load state space autoencoder
            state_space_ae = mlflow.pytorch.load_model(f'runs:/{state_space_runid}/final_sparse_ae_split_{split}', map_location=device)
            state_space_ae.eval()
            # encode states
            encoded_test_states = state_space_ae.encode(normed_test_states.to(device)).cpu()
            encoded_test_next_states = state_space_ae.encode(normed_test_next_states.to(device)).cpu()
            # create trajectory dataset
            test_traj_dataset = OPETrajectoryDataset(discretized_test_states, encoded_test_states, test_discrete_actions, test_cont_actions, discretized_test_next_states,
                                                     encoded_test_next_states, test_rewards, test_dones, test_missing_data_mask, flatten=True)
            # load IQL agent
            # iql_agent = ImplicitQLearning(state_dim=state_space_dim, action_dim=1,
            #                               num_actions=len(MimicIIIDiscreteActionSpace), policy_hidden_dim=policy_hidden_size,
            #                               critic_hidden_dim=critic_hidden_size, expectile_val_hidden_dim=expectile_hidden_size, policy_lr=policy_lr,
            #                               critic_lr=critic_lr, expectile_val_lr=expectile_val_lr,
            #                               gamma=gamma, expectile=expectile, temperature=temperature, clip_norm=clip_norm, tau=tau)
            # iql_agent.load_model(f'runs:/{iql_runid}/final_iql_model_split_{split}')
            # iql_agent.eval()
            d3qn_agent = D3QN(state_dim=state_space_dim, action_dim=1, hidden_dim=d3qn_hidden_size, reward_max=max_reward,
                              gamma=gamma, tau=tau, lr=policy_lr, buffer_size=0, per_alpha=d3qn_per_alpha,
                              per_beta=d3qn_per_beta, per_eps=d3qn_per_eps, reg_lambda=d3qn_reg_lambda, batch_size=0,
                              num_actions=len(MimicIIIDiscreteActionSpace))
            d3qn_agent.load_model(f'runs:/{d3qn_runid}/final_d3qn_model_split_{split}')
            d3qn_agent.eval()
            # create FQE
            fqe = FittedQEvaluation(test_traj_dataset, d3qn_agent, k_itrs=fqe_k_itrs, convergence_eps=fqe_convergence_eps, max_train_itrs=fqe_max_train_itrs,
                                    lr=fqe_lr, hidden_size=fqe_hidden_size, batch_size=fqe_batch_size, use_behavior_policy_states=fqe_use_behavior_state,
                                    run_id=fqe_run_id, split_num=split)
            if not fqe_run_id:
                fqe.log_model()
            policy_value = fqe.compute_value()
            print(f'({split}) FQE Policy Value: {policy_value:.5f}')
            mlflow.log_metric('fqe', policy_value, step=split)
            # create MAGIC
            magic = MAGIC(test_traj_dataset, d3qn_agent, gamma, magic_batch_size, fqe, magic_j_steps, magic_k_conf_iters, magic_eps)
            policy_value, cur_control_variates = magic.compute_value()
            if control_variates is None:
                control_variates = cur_control_variates
            else:
                control_variates = torch.concat((control_variates, cur_control_variates), dim=0)
            print(f'({split}) MAGIC Policy Value: {policy_value:.5f}')
            print(f'({split}) J-Steps: {magic.j_steps}')
            print(f'({split}) J-Step Weights: {magic.j_step_weights}')
            mlflow.log_metric('magic', policy_value, step=split)
            mlflow.log_metrics({f'j_step_{j_step}': weight for j_step, weight in zip(magic.j_steps, magic.j_step_weights)}, step=split)
        # compute control variate stats
        control_variate_means = control_variates.mean(dim=0)
        control_variate_stds = control_variates.std(dim=0)
        for j_step, mean, std in zip(magic.j_steps, control_variate_means, control_variate_stds):
            print(f'Control Variate (J-Step: {j_step}): {mean:.5f} +/- {std:.5f}')
            mlflow.log_metric(f'control_variate_mean_{j_step}', mean)
            mlflow.log_metric(f'control_variate_std_{j_step}', std)
