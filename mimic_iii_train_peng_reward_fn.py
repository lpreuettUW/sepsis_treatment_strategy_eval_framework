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


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--gpu', type=str, default='-1', help='specify the GPU to use')
    args = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    torch.autograd.set_detect_anomaly(True)

    device = DeviceManager.get_device()

    # parameters
    # peng reward function
    reward_fn_name = 'peng'
    num_splits = 10
    batch_size = 128
    num_epochs = 50
    val_mod = 10

    # mlflow stuffs
    mlflow_path = os.path.join('file:///', 'Users', 'larry', 'Documents', 'UWT', 'Thesis Work', 'rec_sys', 'models', 'mimic-iii', 'mlruns')
    mlflow.set_tracking_uri(mlflow_path)
    experiment_name = 'Peng Reward Function'
    run_name = 'run 0'
    mlflow_experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = mlflow.create_experiment(experiment_name) if mlflow_experiment is None else mlflow_experiment.experiment_id

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as mlflow_run:
        param_dict = {
            'reward_fn_name': reward_fn_name,
            'num_splits': num_splits,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'val_mod': val_mod
        }
        mlflow.log_params(param_dict)
        for split in tqdm(range(num_splits), desc='Split', total=num_splits, unit='split', colour='green', position=0, leave=True):
            # create reward function
            reward_fn = MimicIIIRewardFunctionFactory.create(reward_fn_name)
            # load and preprocess mimic data
            mimic_df = mimic_iii_funcs.load_standardized_mimic_data(split)
            mimic_df, min_max_scaler, _ = mimic_iii_funcs.preprocess_mimic_data(mimic_df, 'peng_reward_fn')
            (
                (train_survived_data, train_survived_lbls, train_died_data, train_died_lbls),
                (val_survived_data, val_survived_lbls, val_died_data, val_died_lbls),
                (test_survived_data, test_survived_lbls, test_died_data, test_died_lbls)
            ) = mimic_iii_funcs.train_val_test_split_mimic_data(mimic_df, 'peng_reward_fn')
            # create dataloaders
            train_survived_dataset = mimic_iii_funcs.PengRewardFnDataset(train_survived_data, train_survived_lbls)
            train_died_dataset = mimic_iii_funcs.PengRewardFnDataset(train_died_data, train_died_lbls)
            train_dataloader = mimic_iii_funcs.PengRewardFnDataLoader(train_survived_dataset, train_died_dataset, batch_size, shuffle=True)
            val_data = torch.cat((val_survived_data, val_died_data), dim=0)
            val_lbls = torch.cat((val_survived_lbls, val_died_lbls), dim=0)
            val_dataset = mimic_iii_funcs.PengRewardFnDataset(val_data, val_lbls)
            val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            best_weights = best_score = None
            for epoch in tqdm(range(num_epochs), desc='Epoch', total=num_epochs, unit='epoch', colour='blue', position=1, leave=True):
                reward_fn.train()
                epoch_loss, epoch_acc, num_train_obs = 0.0, 0, 0
                for i, (data, lbls) in enumerate(train_dataloader):
                    train_acc, train_loss = reward_fn.train_batch(data, lbls)
                    epoch_loss += train_loss
                    epoch_acc += train_acc
                    num_train_obs += data.size(0)
                epoch_loss /= num_train_obs
                epoch_acc /= num_train_obs
                mlflow.log_metrics({ 'train_loss': epoch_loss, 'train_acc': epoch_acc }, step=epoch)
                if (epoch + 1) % val_mod == 0:
                    reward_fn.eval()
                    # validate reward function
                    val_loss, val_acc = 0.0, 0
                    for i, (data, lbls) in enumerate(val_dataloader):
                        cur_acc, cur_loss = reward_fn.evaluate(data, lbls)
                        val_acc += cur_acc
                        val_loss += cur_loss
                    val_loss /= len(val_dataloader)
                    val_acc /= len(val_dataset)
                    mlflow.log_metrics({ 'val_loss': val_loss, 'val_acc': val_acc }, step=epoch)
                    if best_score is None or val_loss < best_score:
                        best_score = 0.8 * val_loss + 0.2 * epoch_loss
                        best_weights = reward_fn.get_weights()
            # load best weights
            reward_fn.load_weights(best_weights)
            reward_fn.save(f'split_{split}')
