# This file is part of Ophthalmo-CDSS.
# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# See https://creativecommons.org/licenses/by-nc-sa/4.0/ for details.

import sklearn
from src.forecasting.architectures import BasicRNNRegressor, BasicLSTMRegressor, AdvancedLSTMRegressor
from src.data.data_loader import load_X_y_from_file
from config import ML_DATA, PROGNOSIS_WEIGHTS
import os
import torch
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
from torch.nn import MSELoss, L1Loss


def train(regressor, epochs, loss,
          learning_rate, batch_size, lr_reduction_factor,
          early_stopping_epochs, early_stopping_difference,
          X_train, y_train, X_test, y_test,
          save_path=None):
    train_losses = []
    val_losses = []
    print(f"Training {regressor.__class__.__name__} for {epochs} epochs with a learning rate of {learning_rate} "
          f"and a batch size of {batch_size} on {regressor.device}")
    best_val_loss = np.inf
    best_weights = None
    best_epoch = 0
    for train_dict in regressor.fit(X_train, y_train, criterion=loss, epochs=epochs, learning_rate=learning_rate,
                                    lr_reduction_factor=lr_reduction_factor,
                                    batch_size=batch_size,
                                    X_val=X_test, y_val=y_test):
        train_loss = train_dict["train_loss"]
        val_loss = train_dict["val_loss"]
        epoch = train_dict["epoch"]

        if best_val_loss == np.inf or val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = regressor.state_dict()
            best_epoch = epoch
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if (early_stopping_epochs is not None and
                val_loss - best_val_loss > early_stopping_difference
                and epoch - best_epoch > early_stopping_epochs):
            print(f"Stopping early since val loss has not increased by {early_stopping_difference}"
                  f" in last {early_stopping_epochs} epochs.")
            break

    train_df = pd.DataFrame({"train_loss": train_losses, "val_loss": val_losses})
    training_figure = px.line(data_frame=train_df, x=train_df.index, y=["train_loss", "val_loss"],
                              template="plotly_dark")
    if save_path:
        print(f"Saving best model to {save_path} from epoch {best_epoch} with a validation loss of {best_val_loss}.")
        regressor.load_state_dict(best_weights)
        regressor.save_model(save_path)
    return training_figure, train_df


if __name__ == '__main__':
    dir = os.path.join(ML_DATA, "SmoothedInterpolator_90")
    lrs = [0.01]
    lr_reduction_factors = [0.9]
    batch_sizes = [128]
    losses = [L1Loss()]
    dropout_rates = [0.1]
    hidden_sizes = [16, 32]
    num_layers = [2, 4]
    models_per_dataset = len(lrs) * len(lr_reduction_factors) * len(batch_sizes) * len(losses) * len(dropout_rates) * len(hidden_sizes) * len(num_layers)
    for folder_name in ["visual_acuity", "v_fluids", "n_fluids"]:
        folder = os.path.join(dir, folder_name)
        todo = [f for f in os.listdir(folder) if f.endswith(".npz") and not ("3months" in f or "6months" in f)]

        x_scaler = StandardScaler()  # Initialize the StandardScaler
        y_scaler = StandardScaler()  # Initialize the StandardScaler

        for filename in todo:
            months = filename.split("_")[-1].split(".")[0]
            filepath = os.path.join(folder, filename)
            X_train, y_train, X_test, y_test = load_X_y_from_file(filepath,
                                                                  to_device="cpu")
            y_train = y_train.squeeze()
            y_test = y_test.squeeze()

            # Reshape 3D arrays to 2D arrays for scaling
            N_train_samples, seq_len_train, n_input_features_train = X_train.shape
            N_test_samples, seq_len_test, n_input_features_test = X_test.shape

            X_train_reshaped = X_train.reshape(-1, n_input_features_train)
            X_test_reshaped = X_test.reshape(-1, n_input_features_test)
            # Fit the scaler on training data and transform both training and test data
            X_train_scaled = x_scaler.fit_transform(X_train_reshaped)
            X_test_scaled = x_scaler.transform(X_test_reshaped)

            # Reshape back to 3D arrays
            X_train_scaled = X_train_scaled.reshape(N_train_samples, seq_len_train, n_input_features_train)
            X_test_scaled = X_test_scaled.reshape(N_test_samples, seq_len_test, n_input_features_test)

            y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).squeeze()
            y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).squeeze()

            # Save the scaler
            scaler_save_path = os.path.join(PROGNOSIS_WEIGHTS, f"{filename.replace('.npz', '')}_x_scaler.pkl")
            with open(scaler_save_path, 'wb') as scaler_file:
                pickle.dump(x_scaler, scaler_file)
            scaler_save_path = os.path.join(PROGNOSIS_WEIGHTS, f"{filename.replace('.npz', '')}_y_scaler.pkl")
            with open(scaler_save_path, 'wb') as scaler_file:
                pickle.dump(y_scaler, scaler_file)

            X_train_scaled_tensor = torch.from_numpy(X_train_scaled).float()
            y_train_scaled_tensor = torch.from_numpy(y_train_scaled).float()
            X_test_scaled_tensor = torch.from_numpy(X_test_scaled).float()
            y_test_scaled_tensor = torch.from_numpy(y_test_scaled).float()
            csv_save_path = os.path.join(PROGNOSIS_WEIGHTS,
                                         folder_name,
                                         f"train_results_{filename.split('_')[-1].replace('.npz', '.csv')}")
            os.makedirs(os.path.dirname(csv_save_path), exist_ok=True)
            data = []
            models_trained = 1
            for lr in lrs:
                for lr_reduction_factor in lr_reduction_factors:
                    for batch_size in batch_sizes:
                        for loss in losses:
                            for dropout_rate in dropout_rates:
                                for hidden_size in hidden_sizes:
                                    for num_layer in num_layers:
                                        print(f"Model nr. {models_trained} of {models_per_dataset}\t{models_trained / models_per_dataset:.2%} done.")
                                        regressor = AdvancedLSTMRegressor(input_size=X_train_scaled_tensor.shape[-1],
                                                                          hidden_size=hidden_size,
                                                                          num_layers=num_layer,
                                                                          out_features=1,
                                                                          device="cuda" if torch.cuda.is_available() else "cpu",
                                                                          dropout_rate=dropout_rate, )

                                        save_path = os.path.join(PROGNOSIS_WEIGHTS,
                                                                 folder_name,
                                                                 f"{filename.replace('.npz', '')}",
                                                                 f"{regressor.__class__.__name__}_"
                                                                 f"hidden_{hidden_size}_"
                                                                 f"layers_{num_layer}_"
                                                                 f"lr_{lr}_"
                                                                 f"redf_{lr_reduction_factor}_"
                                                                 f"batch_{batch_size}_"
                                                                 f"loss_{loss.__class__.__name__}_"
                                                                 f"drop_{dropout_rate}.pt")
                                        if os.path.exists(save_path):
                                            models_trained += 1
                                            continue
                                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                                        print(f"Months\tLR\tFactor\tBatch\tDropout\tHidden\tNum. Layers")
                                        print(f"{months}\t"
                                              f"{lr} \t{lr_reduction_factor}     \t{batch_size}    \t"
                                              f"{dropout_rate}      \t"
                                              f"{hidden_size}     \t{num_layer}          ")
                                        fig, df = train(regressor, epochs=500, learning_rate=lr, batch_size=batch_size,
                                                        lr_reduction_factor=lr_reduction_factor, loss=loss,
                                                        early_stopping_difference=0., early_stopping_epochs=100,
                                                        X_train=X_train_scaled_tensor, y_train=y_train_scaled_tensor,
                                                        X_test=X_test_scaled_tensor, y_test=y_test_scaled_tensor,
                                                        save_path=save_path)
                                        data.append({"Learning Rate": lr,
                                                     "LR Reduction Factor": lr_reduction_factor,
                                                     "Batch Size": batch_size,
                                                     "Loss": loss.__class__.__name__,
                                                     "Dropout Rate": dropout_rate,
                                                     "Number of Layers": num_layer,
                                                     "Hidden size": hidden_size,
                                                     "Best epoch": df["val_loss"].argmin(),
                                                     "Best loss": df["val_loss"].min()})
                                        model_frame = pd.DataFrame(data)
                                        model_frame.to_csv(csv_save_path)
                                        models_trained += 1
                                        del regressor
