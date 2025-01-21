# This file is part of Ophthalmo-CDSS.
# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# See https://creativecommons.org/licenses/by-nc-sa/4.0/ for details.

import torch
import torch.nn as nn
import streamlit as st
import plotly.express as px
import numpy as np
from tqdm import tqdm
from logging import getLogger
from typing import Union
import torch.optim.lr_scheduler as lr_scheduler

logger = getLogger(__name__)


class Regressor(nn.Module):
    """
    Base model for forecasting
    """

    def __init__(self, device):
        super(Regressor, self).__init__()
        self.device = device
        self.to(device)

    def forward(self, x):
        raise NotImplementedError

    def fit(self, X, y,
            epochs, learning_rate, batch_size,
            early_stopping: Union[bool, int] = False,
            X_val=None, y_val=None,
            lr_reduction_factor=2,
            criterion=nn.MSELoss()):
        """
        Trains the model using the provided training data and parameters.

        Parameters:
            X (torch.Tensor): The input features for training.
            y (torch.Tensor): The target labels for training.
            epochs (int): The number of training epochs.
            learning_rate (float): The initial learning rate for the optimizer.
            batch_size (int): The number of samples per batch during training.
            early_stopping (Union[bool, int], optional): If `False`, early stopping is disabled.
                If `True`, training stops if validation loss does not improve for 10 epochs.
                If an integer is provided, it specifies the patience (in epochs) for early stopping.
            X_val (torch.Tensor, optional): The input features for validation. Default is `None`.
            y_val (torch.Tensor, optional): The target labels for validation. Default is `None`.
            lr_reduction_factor (float, optional): The factor by which the learning rate is reduced
                when validation loss plateaus. Default is `2`.
            criterion (torch.nn.Module, optional): The loss function to use. Default is `nn.MSELoss()`.

        Yields:
            dict: A dictionary containing the training and validation loss, epoch number, and other metrics:
                - "train_loss" (float): The average training loss for the epoch.
                - "val_loss" (float): The average validation loss for the epoch (if validation data is provided).
                - "epoch" (int): The current epoch number.

        Notes:
            - The model parameters are updated using the Adam optimizer.
            - The learning rate is adjusted based on the validation loss using a `ReduceLROnPlateau` scheduler.
            - Progress is displayed using a progress bar (via `tqdm`), showing metrics such as training loss,
              validation loss, learning rate, mean predictions, and actual target values.
            - If early stopping is enabled, training halts if the validation loss does not improve
              for a specified number of epochs.

        Example:
            ```python
            model = MyModel()
            train_losses = []
            val_losses = []

            for metrics in model.fit(X_train, y_train, epochs=50, learning_rate=0.001, batch_size=32,
                                     early_stopping=5, X_val=X_val, y_val=y_val):
                train_losses.append(metrics["train_loss"])
                val_losses.append(metrics["val_loss"])
            ```
        """
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=lr_reduction_factor)
        val_losses = []
        with tqdm(total=epochs, desc="Training", unit="epoch") as pbar:
            for epoch in range(epochs):
                epoch_loss = 0
                self.train()
                shuffled_indices = np.arange(len(X))
                np.random.shuffle(shuffled_indices)  # Shuffle indices
                X = X[shuffled_indices]
                y = y[shuffled_indices]
                for i in range(0, len(X), batch_size):
                    optimizer.zero_grad()
                    batch_X = X[i:i + batch_size].to(self.device)
                    batch_y = y[i:i + batch_size].to(self.device)
                    outputs = self(batch_X).squeeze()
                    loss = criterion(outputs, batch_y)
                    epoch_loss += loss.item()
                    loss.backward()
                train_loss = epoch_loss / (len(X) / batch_size)
                epoch_loss = 0
                if X_val is not None or y_val is not None:
                    self.eval()
                    shuffled_indices_val = np.arange(len(X_val))
                    np.random.shuffle(shuffled_indices_val)  # Shuffle indices
                    X_val = X_val[shuffled_indices_val]
                    y_val = y_val[shuffled_indices_val]
                    for i in range(0, len(X_val), batch_size):
                        batch_X = X_val[i:i + batch_size].to(self.device)
                        batch_y = y_val[i:i + batch_size].to(self.device)
                        outputs = self(batch_X).squeeze()
                        loss = criterion(outputs, batch_y)
                        epoch_loss += loss.item()
                optimizer.step()
                val_loss = 0.0 if X_val is None else epoch_loss / (len(X_val) / batch_size)
                val_losses.append(val_loss)
                pbar.set_postfix(train_loss=train_loss, val_loss=val_loss, lr=optimizer.param_groups[0]['lr'],
                                 mean_pred=outputs.mean().item(),
                                 mean_actual=batch_y.mean().item())
                yield {"train_loss": train_loss,
                       "val_loss": val_loss,
                       "epoch": epoch + 1}
                # Step the scheduler with the validation loss
                scheduler.step(val_loss)
                if early_stopping:
                    early_stopping_epochs = 10 if type(early_stopping) is bool else early_stopping
                    indices = -early_stopping_epochs if len(val_losses) > 10 else -len(val_losses)
                    diffs = np.array(val_losses[indices:]) - val_loss
                    if np.all(diffs < 0):
                        logger.info("Stopping training early "
                                    "since validation loss has not improved in the last 10 epochs.")
                        break
                pbar.update(1)

    def test(self, X, y):
        """ Evaluate forecasting for test data X on groundtruth y. Criterion returned is the Mean Squared Error."""
        self.eval()
        criterion = nn.MSELoss()
        with torch.no_grad():
            outputs = self(X)
            loss = criterion(outputs, y)
        return loss.item()

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))


def get_predictions(model, sequence_input):
    return model.predict_all_treatment(sequence_input)
