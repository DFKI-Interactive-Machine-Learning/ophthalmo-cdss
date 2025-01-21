# This file is part of Ophthalmo-CDSS.
# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# See https://creativecommons.org/licenses/by-nc-sa/4.0/ for details.

import pandas as pd
import numpy as np
import os
import torch
from logging import getLogger
from collections import defaultdict
from src.data.preprocess import preprocess_visit_details_of_patient_for_model, split_X_y_into_sequences
from src.data.database_access import get_patients, get_joined_vol_and_patient_details
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from typing import Tuple

logger = getLogger(__name__)


def load_X_y_from_file(file_path, to_device=None):
    """ Load the data from the file specified by file_path. The data is returned as X_train, y_train, X_test, y_test
        numpy arrays. If to_device is given, the data is converted to torch tensors and moved to the device specified
        by to_device. to_device can be "cpu" or "cuda"."""
    logger.info(f"Loading data from file {file_path}")
    npzfile = np.load(file_path)
    X_train = npzfile["X_train"]
    y_train = npzfile["y_train"]
    X_test = npzfile["X_test"]
    y_test = npzfile["y_test"]
    if to_device is None:
        return X_train, y_train, X_test, y_test
    else:
        return torch.from_numpy(X_train).float().to(to_device), torch.from_numpy(y_train).float().to(to_device), \
            torch.from_numpy(X_test).float().to(to_device), torch.from_numpy(y_test).float().to(to_device)


def load_usecase_as_X_y(usecase,
                        target_metric,
                        target_months,
                        train_perc=0.8,
                        sequence_length=12,
                        interpolator=None,
                        save_to_file: str = None,
                        save_dfs_to_dir: str = None,
                        overwrite=False,
                        limit: int = -1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ Load the data for a specific usecase as X and y. The target column is removed from the X dataframe
        and returned as y. The data is split into train and test data. The train data is returned as X_train
        and y_train, the test data as X_test and y_test. The data is split into sequences of length sequence_length.
        If save_to_file is not None, the data is saved to the file specified by save_to_file.
        If the file exists, the data is loaded from the file instead of the database. This overrides the params.
        :param usecase: The use case for which the data should be loaded. Can also be a list of usecases.
        :param target_metric: The target metric that should be forecasted.
        :param target_months: The number of months in the future the target metric should be forecasted.
        :param train_perc: The percentage of patients that should be used for training.
        :param sequence_length: The length of the sequences that the data is split into.
        :param interpolator: The interpolator that is used to interpolate the data.
        :param save_to_file: The file to which the data is saved. If None, the data is not saved.
        :param save_dfs_to_dir: The directory to which the preprocessed dataframes are saved for reusage.
            If None, the dataframes are not saved.
        :param overwrite: If True, the data is saved to the file even if it already exists.
        :param limit: The number of patients that should be loaded. If -1, all patients are loaded.
        :return: The train and test data as X_train, y_train, X_test, y_test numpy arrays."""
    save_to_file = save_to_file if save_to_file.endswith(".npz") else save_to_file + ".npz"
    if save_to_file is not None and os.path.exists(save_to_file) and not overwrite:
        raise FileExistsError(f"File {save_to_file} already exists and overwrite is set to False.")
    elif save_to_file is not None:
        logger.info(f"Loading data from database")
        os.makedirs(os.path.dirname(save_to_file), exist_ok=True)
    patients = get_patients(filter_usecase=usecase)
    if limit > 0:
        patients = patients.head(limit)
    patient_ids = patients["id"].tolist()
    test_ids = np.random.choice(patient_ids, int(len(patient_ids) * (1.0 - train_perc)), replace=False)
    X_train_list = []
    y_train_list = []
    X_test_list = []
    y_test_list = []
    uniques = defaultdict(set)
    skipped = 0
    finished = 0
    with logging_redirect_tqdm():
        with tqdm(desc="Patients loaded", total=len(patients) * 2) as pbar:
            for i, patient_data in patients.iterrows():
                for side in ["left", "right"]:
                    try:
                        df_filename = f"{interpolator.__name__()}_{patient_data['id']}_{side}_{target_metric}_{target_months}.csv"
                        df_filepath = os.path.join(save_dfs_to_dir, df_filename)
                        if save_dfs_to_dir is not None and os.path.exists(df_filepath):
                            preprocessed_df = pd.read_csv(df_filepath)
                        else:
                            joined_df = get_joined_vol_and_patient_details(patient_data["id"], side)
                            preprocessed_df = preprocess_visit_details_of_patient_for_model(joined_df,
                                                                                            target_metric=target_metric,
                                                                                            target_months=target_months,
                                                                                            side=side,
                                                                                            birthday=pd.to_datetime(
                                                                                                patient_data[
                                                                                                    "birthday"]),
                                                                                            gender=patient_data[
                                                                                                "gender"],
                                                                                            interpolator=interpolator)
                            if save_dfs_to_dir is not None and not preprocessed_df.empty:
                                os.makedirs(save_dfs_to_dir, exist_ok=True)
                                preprocessed_df.to_csv(df_filepath)
                        if preprocessed_df.empty:
                            logger.warning(
                                f"Preprocessed dataframe for patient {patient_data['id']} is empty. Skipping.")

                            skipped += 1
                            pbar.set_postfix(finished=finished, skipped=skipped)
                            pbar.update(1)
                            continue
                        X, y = load_dataframe_as_X_y(preprocessed_df,
                                                     f"Expected effect on {target_metric} in {target_months} months")
                        if sequence_length > 1:
                            X, y = split_X_y_into_sequences(X, y, sequence_length)
                        if np.isnan(X).any() or np.isnan(y).any():
                            logger.error(f"NaN values in patient {patient_data['id']} on {side} side! Skipping.")
                            skipped += 1
                            pbar.set_postfix(finished=finished, skipped=skipped)
                            pbar.update(1)
                            continue
                        if np.all(y == 0):
                            logger.error(f"All target values are 0 for patient {patient_data['id']} on {side} side!")
                        if patient_data["id"] not in test_ids:
                            X_train_list += X
                            y_train_list += y
                        else:
                            X_test_list += X
                            y_test_list += y
                        finished += 1
                    except Exception as e:
                        logger.error(f"Could not load patient {patient_data['id']} because of error {e}")
                        skipped += 1
                    pbar.set_postfix(finished=finished, skipped=skipped)
                    pbar.update(1)
    logger.info(f"Loaded {len(patient_ids) - skipped} patients from use case {usecase}\t"
                f"Skipped {skipped} patients because of errors.")
    X_train = np.array(X_train_list)
    y_train = np.array(y_train_list)
    X_test = np.array(X_test_list)
    y_test = np.array(y_test_list)
    for i in uniques.keys():
        # Encode the categorical values here, because the LabelEncoder cannot be used in the
        # preprocess_visit_details_of_patient_for_model function or else we would get different labels.
        X_train[:, :, i] = LabelEncoder().fit(list(uniques[i])).transform(X_train[:, :, i])
        X_test[:, :, i] = LabelEncoder().fit(list(uniques[i])).transform(X_test[:, :, i])
    logger.info(f"Train samples: {X_train.shape[0]}\tTrain features: {X_train.shape[-1]}\t")
    logger.info(f"Test samples: {X_test.shape[0]}\tTest features: {X_test.shape[-1]}\t")
    if np.any(np.isnan(X_train)) or np.any(np.isnan(y_train)) or np.any(np.isnan(X_test)) or np.any(np.isnan(y_test)):
        variables = ["X_train", "y_train", "X_test", "y_test"]
        nan_vars = []
        for var in variables:
            if np.any(np.isnan(eval(var))):
                nan_vars.append(var)
        logger.critical(f"NaN values in {nan_vars}! This will cause problems with the ML algorithm!")
    if save_to_file is not None:
        logger.info(f"Saving data to file {save_to_file}.")
        np.savez(save_to_file, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    return X_train, y_train, X_test, y_test


def load_dataframe_as_X_y(df, target_columns, as_tensor=False):
    """ Load the data as X and y. The target column is removed from the X dataframe and returned as y."""
    try:
        X = df.drop(columns=target_columns, inplace=False)
        y = df[target_columns]
    except KeyError:
        logger.error(f"Could not find target column {target_columns} in dataframe. Available columns are {df.columns}")
        raise
    if as_tensor:
        return torch.from_numpy(X.values).float(), torch.from_numpy(y.values).float()
    else:
        return np.array(X), np.array(y)


def load_and_scale_data_from_path(filepath, device="cpu"):
    """ Load the data from the file specified by filepath and scale it using a StandardScaler. The data is returned
        as X_train, y_train, X_test, y_test torch tensors. The data is scaled to have a mean of 0 and a standard
        deviation of 1. The data is moved to the device specified by device."""
    X_train, y_train, X_test, y_test = load_X_y_from_file(filepath,
                                                          to_device="cpu")
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()

    x_scaler = StandardScaler()  # Initialize the StandardScaler
    y_scaler = StandardScaler()  # Initialize the StandardScaler

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

    X_train_scaled_tensor = torch.from_numpy(X_train_scaled).float().to(device)
    y_train_scaled_tensor = torch.from_numpy(y_train_scaled).float().to(device)
    X_test_scaled_tensor = torch.from_numpy(X_test_scaled).float().to(device)
    y_test_scaled_tensor = torch.from_numpy(y_test_scaled).float().to(device)
    return X_train_scaled_tensor, y_train_scaled_tensor, X_test_scaled_tensor, y_test_scaled_tensor
