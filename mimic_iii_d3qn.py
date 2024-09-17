import os
import torch
import mlflow
from tqdm import tqdm
from typing import Dict
from argparse import ArgumentParser

from agents.d3qn import D3QN
from mdp.mimic_iii.action_spaces.discrete import Discrete as MimicIIIDiscreteActionSpace
from mdp.mimic_iii.reward_functions.factory import Factory as MimicIIIRewardFunctionFactory
from mdp.mimic_iii.state_spaces.sparse_autoencoder import SparseAutoEncoder as MimicIIISparseAutoEncoderStateSpace
from utilities.device_manager import DeviceManager
from utilities import mimic_iii_funcs


def do_evaluation(agent: D3QN, dataloader: torch.utils.data.DataLoader, num_samps: int, device_: torch.device, dataset_type_: str) -> Dict[str, float]:
    agent.eval()
    losses_dict = None
    for states, actions, next_states, rewards, dones in tqdm(dataloader, desc=f'{dataset_type_} Batch', total=len(dataloader), unit='batch ', colour='blue'):
        states, actions, next_states, rewards, dones = states.to(device_), actions.to(device_), next_states.to(device_), rewards.to(device_), dones.to(device_)
        if actions.ndim == 1:
            actions = actions.unsqueeze(-1)
        # agent will send data to device
        batch_losses_dict = agent.evaluate(states, actions, rewards, next_states, dones)
        if losses_dict is None:
            losses_dict = batch_losses_dict
        else:
            for key, value in batch_losses_dict.items():
                losses_dict[key] += value
    # log train losses
    for key in losses_dict.keys():
        losses_dict[key] /= num_samps # NOTE: we do this bc D3QN sums the losses
    return losses_dict


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--gpu', type=str, default='-1', help='specify the GPU to use')
    args = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    torch.autograd.set_detect_anomaly(True)

    device = DeviceManager.get_device()

    # parameters
    # agent
    policy_lr = 1e-4
    tau = 5e-3
    per_alpha = 0.6
    per_beta = 0.9
    per_eps = 1e-2
    hidden_size = 128
    reg_lambda = 5.0 # regularization lambda penalizing Q-values greater than the max possible reward
    # env
    gamma = 0.99
    num_train_steps = 60000 # Raghu et al. (2017) used 60k train steps
    state_space_runid = 'fda6c8118cd54d36a258db25e37969a2'
    state_space_dim = 200
    reward_fn_name = 'sparse'
    peng_reward_fn_runid = '393b38ea4c334946aa914c6c53c472c3'
    # training
    batch_size = 32
    val_step_mod = 1000
    log_mod = 100
    num_splits = 10

    # mlflow stuffs
    mlflow_path = os.path.join('file:///', '<your_base_path>', 'mimic-iii', 'mlruns')
    mlflow.set_tracking_uri(mlflow_path)
    experiment_name = 'D3QN MIMIC III: Reward Function Eval'
    run_name = f'Reward Function: {reward_fn_name}'
    mlflow_experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = mlflow.create_experiment(experiment_name) if mlflow_experiment is None else mlflow_experiment.experiment_id
    # create reward function
    reward_fn = MimicIIIRewardFunctionFactory.create(reward_fn_name)

    initial_weights = None

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as mlflow_run:
        param_dict = {
            'policy_lr': policy_lr,
            'per_alpha': per_alpha,
            'per_beta': per_beta,
            'per_eps': per_eps,
            'hidden_size': hidden_size,
            'tau': tau,
            'gamma': gamma,
            'num_train_steps': num_train_steps,
            'reward_fn_name': reward_fn_name,
            'peng_reward_fn_runid': peng_reward_fn_runid,
            'batch_size': batch_size,
            'reward_fn_type': type(reward_fn),
            'val_step_mod': val_step_mod,
            'log_mod': log_mod,
            'num_splits': num_splits,
            'state_space_runid': state_space_runid,
            'state_space_dim': state_space_dim,
        }
        mlflow.log_params(param_dict)
        for split in tqdm(range(num_splits), desc='Split', total=num_splits, unit='split ', colour='green'):
            if reward_fn_name == 'peng':
                if not peng_reward_fn_runid:
                    raise RuntimeError('Peng reward function runid must be specified for Peng reward function')
                reward_fn.load(f'runs:/{peng_reward_fn_runid}/split_{split}')
            # load and preprocess mimic data
            mimic_df = mimic_iii_funcs.load_standardized_mimic_data(split, reward_fn.raw_data_columns)
            if reward_fn_name != 'peng':
                reward_fn.update_rewards(mimic_df)  # scale rewards
                mimic_df = mimic_iii_funcs.drop_all_raw_data_columns(mimic_df)
            mimic_df, min_max_scaler, _ = mimic_iii_funcs.preprocess_mimic_data(mimic_df, 'iql_discrete')
            if reward_fn_name == 'peng':
                reward_fn.update_rewards(mimic_df)  # scale rewards
            max_reward = mimic_df['r:reward'].abs().max()
            (
                (train_states, train_actions, train_next_states, train_rewards, train_dones),
                (val_states, val_actions, val_next_states, val_rewards, val_dones),
                (test_states, test_actions, test_next_states, test_rewards, test_dones)
            ) = mimic_iii_funcs.train_val_test_split_mimic_data(mimic_df, 'iql_discrete')
            # load state space
            state_space_ae: MimicIIISparseAutoEncoderStateSpace = mlflow.pytorch.load_model(f'runs:/{state_space_runid}/final_sparse_ae_split_{split}').to(device)
            state_space_ae.eval()
            # encode states
            train_states = mimic_iii_funcs.encode_states(train_states, state_space_dim, state_space_ae, batch_size, device)
            train_next_states = mimic_iii_funcs.encode_states(train_next_states, state_space_dim, state_space_ae, batch_size, device)
            val_states = mimic_iii_funcs.encode_states(val_states, state_space_dim, state_space_ae, batch_size, device)
            val_next_states = mimic_iii_funcs.encode_states(val_next_states, state_space_dim, state_space_ae, batch_size, device)
            # create dataloaders
            train_dataset = mimic_iii_funcs.ImplicitQLearningDataset(train_states, train_actions, train_next_states, train_rewards, train_dones)
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_dataset = mimic_iii_funcs.ImplicitQLearningDataset(val_states, val_actions, val_next_states, val_rewards, val_dones)
            val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            # initialize agent with the same initial weights for each split
            d3qn_agent = D3QN(state_dim=state_space_dim, action_dim=1, hidden_dim=hidden_size, reward_max=max_reward,
                              gamma=gamma, tau=tau, lr=policy_lr, buffer_size=len(train_dataset), per_alpha=per_alpha,
                              per_beta=per_beta, per_eps=per_eps, reg_lambda=reg_lambda, batch_size=batch_size, num_actions=len(MimicIIIDiscreteActionSpace))
            d3qn_agent.reset_prioritization_bias_correction_annealing(num_train_steps)
            # load/cache initial weights
            if initial_weights is None:
                initial_weights = d3qn_agent.get_weights()
            else:
                d3qn_agent.load_weights(initial_weights)
            # fill replay buffer
            d3qn_agent.fill_replay_buffer(train_dataloader)
            # train agent
            best_weights = best_combined_loss = None # TODO: I dont trust the loss to help us choose this
            for batch in tqdm(range(num_train_steps), desc='Batch', total=num_train_steps, unit='batch ', colour='blue'):
                batch_losses_dict = d3qn_agent.batch_train()
                if (batch + 1) % log_mod == 0:
                    # update loss dict keys
                    batch_losses_dict = {f'train_{key}_split_{split}': value for key, value in batch_losses_dict.items()}
                    mlflow.log_metrics(batch_losses_dict, step=batch)
            # evaluate agent
            for dataloader, num_samples, dataset_type in zip([train_dataloader, val_dataloader], [len(train_dataset), len(val_dataset)], ['train', 'val']):
                loss_dict = do_evaluation(d3qn_agent, dataloader, len(train_dataset), device, dataset_type)
                loss_dict = {f'eval_{dataset_type}_{key}_split_{split}': value for key, value in loss_dict.items()}
                mlflow.log_metrics(loss_dict, step=batch)
            # save model
            d3qn_agent.save_model(f'final_d3qn_model_split_{split}')
