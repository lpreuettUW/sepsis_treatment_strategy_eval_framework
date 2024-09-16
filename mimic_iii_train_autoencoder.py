import os
import copy
import torch
import mlflow
from tqdm import tqdm
from argparse import ArgumentParser

from utilities import mimic_iii_funcs
from mdp.mimic_iii.state_spaces.sparse_autoencoder import SparseAutoEncoder
from utilities.device_manager import DeviceManager


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--gpu', type=str, default='-1', help='specify the GPU to use')
    args = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    torch.autograd.set_detect_anomaly(True)
    device = DeviceManager.get_device()

    # parameters
    obs_dim = 47
    num_splits = 10
    batch_size = 100
    num_epochs = 20
    hidden_size = 200
    sparsity_parameter = 5e-2
    sparsity_weight = 1e-4
    learning_rate = 1e-4
    val_epoch_mod = 2

    mlflow_path = os.path.join('file:///', 'Users', 'larry', 'Documents', 'UWT', 'Thesis Work', 'rec_sys', 'models', 'mimic-iii', 'mlruns')
    mlflow.set_tracking_uri(mlflow_path)
    experiment_name = 'Sparse Autoencoder'
    run_name = f'Updated Splits: Normed Mimic Data via Preprocessing Script and Min-Max Scaling'
    mlflow_experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = mlflow.create_experiment(experiment_name) if mlflow_experiment is None else mlflow_experiment.experiment_id

    initial_weights = None

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as mlflow_run:
        # log parameters
        param_dict = {
            'obs_dim': obs_dim,
            'num_splits': num_splits,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'hidden_size': hidden_size,
            'sparsity_parameter': sparsity_parameter,
            'sparsity_weight': sparsity_weight,
            'learning_rate': learning_rate,
            'val_epoch_mod': val_epoch_mod
        }
        mlflow.log_params(param_dict)
        for split in tqdm(range(num_splits), desc='Split', total=num_splits, unit='split ', colour='green', position=0, leave=True):
            # initialize model with the same initial weights for each split
            sparse_autoencoder = SparseAutoEncoder(obs_dim, hidden_size, sparsity_parameter, sparsity_weight).to(device)
            if initial_weights is None:
                initial_weights = copy.deepcopy(sparse_autoencoder.state_dict())
            else:
                sparse_autoencoder.load_state_dict(initial_weights)
            optimizer = torch.optim.Adam(sparse_autoencoder.parameters(), lr=learning_rate)
            # load and preprocess mimic data
            mimic_df = mimic_iii_funcs.load_standardized_mimic_data(split)
            mimic_df, min_max_scaler, _ = mimic_iii_funcs.preprocess_mimic_data(mimic_df, 'sparse_auto_encoder')
            train_data, val_data, _ = mimic_iii_funcs.train_val_test_split_mimic_data(mimic_df, 'sparse_auto_encoder')
            # create train, val, and test dataloaders
            train_dataset = mimic_iii_funcs.SparseAutoEncoderDataset(train_data)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_dataset = mimic_iii_funcs.SparseAutoEncoderDataset(val_data)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            # train model
            best_weights = best_combined_loss = None
            for epoch in tqdm(range(num_epochs), desc='Epoch', total=num_epochs, unit='epoch ', colour='blue', position=1, leave=True):
                epoch_loss = 0
                sparse_autoencoder.train()
                for data in train_loader:
                    data = data.to(device)
                    optimizer.zero_grad()
                    x, loss = sparse_autoencoder(data)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.cpu().item()
                # log loss
                epoch_loss /= len(train_loader)
                mlflow.log_metric(f'train_loss_split_{split}', epoch_loss, step=epoch)
                # evaluate on validation set
                if (epoch + 1) % val_epoch_mod == 0:
                    val_loss = 0
                    sparse_autoencoder.eval()
                    with torch.no_grad():
                        for data in val_loader:
                            data = data.to(device)
                            x, loss = sparse_autoencoder(data)
                            val_loss += loss.cpu().item()
                    val_loss /= len(val_loader)
                    mlflow.log_metric(f'val_loss_split{split}', val_loss, step=epoch)
                    # update best weights
                    combined_loss = 0.2 * epoch_loss + 0.8 * val_loss
                    if best_combined_loss is None or combined_loss < best_combined_loss:
                        best_combined_loss = combined_loss
                        best_weights = copy.deepcopy(sparse_autoencoder.state_dict())
            # log best weights
            sparse_autoencoder.load_state_dict(best_weights)
            mlflow.pytorch.log_model(sparse_autoencoder, f'final_sparse_ae_split_{split}')


