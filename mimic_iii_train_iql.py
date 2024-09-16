import os
import copy
import torch
import mlflow
from tqdm import tqdm
from typing import Tuple, Dict, Optional
from argparse import ArgumentParser

from agents.implicit_q_learning import ImplicitQLearning
from agents.abstract_batch_agent import AbstractBatchAgent
from agents.abstract_agent import AbstractAgent
from mdp.policies.next_best_action_policy import NextBestActionPolicy
from mdp.action import Action
from mdp.mimic_iii.action_spaces.discrete import Discrete as MimicIIIDiscreteActionSpace
from mdp.mimic_iii.state_spaces.sparse_autoencoder import SparseAutoEncoder as MimicIIISparseAutoEncoderStateSpace
from mdp.mimic_iii.reward_functions.factory import Factory as MimicIIIRewardFunctionFactory
from utilities.device_manager import DeviceManager
from utilities import mimic_iii_funcs


def batch_epoch_evaluation(agent: ImplicitQLearning, dataloader_: torch.utils.data.DataLoader, device_: torch.device, train: bool) -> Dict[str, float]:
    """
    Train the agent on a batch of data.
    :param agent: Implicit Q-Learning agent.
    :param dataloader_: DataLoader containing the batch of data.
    :param device_: Device to use for training.
    :param state_space_ae_: State space autoencoder.
    :param train: Whether to train the agent.
    :return: Losses dictionary.
    """
    losses_dict = None
    agent.train() if train else agent.eval()
    for states, actions, next_states, rewards, dones in dataloader_:
        states, actions, next_states, rewards, dones = states.to(device_), actions.to(device_), next_states.to(device_), rewards.to(device_), dones.to(device_)
        if actions.ndim == 1:
            actions = actions.unsqueeze(-1)
        batch_losses_dict = agent.batch_update(states, actions, rewards, next_states, dones) if train else agent.compute_losses(states, actions, rewards, next_states, dones)
        if losses_dict is None:
            losses_dict = batch_losses_dict
        else:
            for key, value in batch_losses_dict.items():
                losses_dict[key] += value
    # log train losses
    for key in losses_dict.keys():
        losses_dict[key] /= len(dataloader_)
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
    critic_lr = 1e-4
    expectile_val_lr = 1e-4
    policy_hidden_size = 128
    critic_hidden_size = 128
    expectile_hidden_size = 128
    agent_weight_decay = 1e-4
    expectile = 0.8
    temperature = 0.1
    clip_norm = 1.0
    tau = 5e-3
    # env
    gamma = 0.99
    num_epochs = 300
    state_space_runid = 'fda6c8118cd54d36a258db25e37969a2'
    state_space_dim = 200
    reward_fn_name = 'sparse'
    peng_reward_fn_runid = '393b38ea4c334946aa914c6c53c472c3'
    # training
    batch_size = 512
    val_epoch_mod = 10
    num_splits = 10

    # mlflow stuffs
    mlflow_path = os.path.join('file:///', 'Users', 'larry', 'Documents', 'UWT', 'Thesis Work', 'rec_sys', 'models', 'mimic-iii', 'mlruns')
    mlflow.set_tracking_uri(mlflow_path)
    experiment_name = 'IQL Mimic III: Reward Function Eval'
    run_name = f'Reward Function: {reward_fn_name}'
    mlflow_experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = mlflow.create_experiment(experiment_name) if mlflow_experiment is None else mlflow_experiment.experiment_id
    # create reward function
    reward_fn = MimicIIIRewardFunctionFactory.create(reward_fn_name)

    initial_weights = None

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as mlflow_run:
        param_dict = {
            'policy_lr': policy_lr,
            'critic_lr': critic_lr,
            'expectile_val_lr': expectile_val_lr,
            'agent_weight_decay': agent_weight_decay,
            'policy_hidden_size': policy_hidden_size,
            'critic_hidden_size': critic_hidden_size,
            'expectile_hidden_size': expectile_hidden_size,
            'expectile': expectile,
            'temperature': temperature,
            'clip_norm': clip_norm,
            'tau': tau,
            'gamma': gamma,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'reward_fn_type': type(reward_fn),
            'val_epoch_mod': val_epoch_mod,
            'num_splits': num_splits,
            'state_space_runid': state_space_runid,
            'state_space_dim': state_space_dim,
            'reward_fn_name': reward_fn_name
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
            iql_agent = ImplicitQLearning(state_dim=state_space_dim, action_dim=1,
                                          num_actions=len(MimicIIIDiscreteActionSpace), policy_hidden_dim=policy_hidden_size,
                                          critic_hidden_dim=critic_hidden_size, expectile_val_hidden_dim=expectile_hidden_size, policy_lr=policy_lr,
                                          critic_lr=critic_lr, expectile_val_lr=expectile_val_lr,
                                          gamma=gamma, expectile=expectile, temperature=temperature, clip_norm=clip_norm, tau=tau,
                                          weight_decay=agent_weight_decay)
            if initial_weights is None:
                initial_weights = iql_agent.get_weights()
            else:
                iql_agent.load_weights(initial_weights)

            best_weights = best_combined_loss = None
            for epoch in tqdm(range(num_epochs), desc='Epoch', total=num_epochs, unit='epoch ', colour='blue'):
                epoch_losses_dict = batch_epoch_evaluation(iql_agent, train_dataloader, device, train=True)
                # update loss dict keys
                epoch_losses_dict = {f'train_{key}_split_{split}': value for key, value in epoch_losses_dict.items()}
                mlflow.log_metrics(epoch_losses_dict, step=epoch)
                # evaluate on validation set
                if (epoch + 1) % val_epoch_mod == 0:
                    val_losses_dict = batch_epoch_evaluation(iql_agent, val_dataloader, device, train=False)
                    # update loss dict keys
                    val_losses_dict = {f'val_{key}_split_{split}': value for key, value in val_losses_dict.items()}
                    mlflow.log_metrics(val_losses_dict, step=epoch)
                    # compute score
                    combined_score = 0
                    for val_loss in val_losses_dict.values():
                        combined_score += 0.8 * val_loss
                    for train_loss in epoch_losses_dict.values():
                        combined_score += 0.2 * train_loss
                    if best_combined_loss is None or combined_score < best_combined_loss:
                        best_combined_loss = val_loss
                        best_weights = iql_agent.get_weights()
            # reload best weights
            iql_agent.load_weights(best_weights)
            # save model
            iql_agent.save_model(f'final_iql_model_split_{split}')
