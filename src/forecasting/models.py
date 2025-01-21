# This file is part of Ophthalmo-CDSS.
# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# See https://creativecommons.org/licenses/by-nc-sa/4.0/ for details.

from os.path import join
import pandas as pd
import numpy as np
import torch
import os
import config
from logging import getLogger
from config import PROGNOSIS_WEIGHTS, IvomDrugs
from src.forecasting.architectures import *
from src.data.preprocess import preprocess_visit_details_of_patient_for_model
from collections import defaultdict
import pickle


def load_scaler(scaler_load_path):
    with open(scaler_load_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return scaler


logger = getLogger(__name__)


class PrognosisModel:

    def __init__(self):
        self.model = None
        self.device = None

    @staticmethod
    def prepare_df(joined_df, side, usecase, birthday, gender):
        input_features = preprocess_visit_details_of_patient_for_model(joined_df,
                                                                       target_metric=None,
                                                                       target_months=0,
                                                                       birthday=birthday,
                                                                       gender=gender,
                                                                       without_targets=True,
                                                                       default_usecase=usecase,
                                                                       side=side)
        last_12_visits = input_features.iloc[-12:]
        return last_12_visits

    def predict_all_treatment(self, df):
        if not self.model:
            raise NotImplementedError("This model is not yet implemented.")
        preds = {}
        for treatment in IvomDrugs.keys():
            df[config.DatabaseKeys.drug].iloc[-1] = IvomDrugs[treatment]
            X = torch.from_numpy(df.to_numpy()).float().unsqueeze(0).to(self.device)
            logger.debug(f"Sending input of shape {X.shape} to device {self.device}")
            pred = self.model(X).detach().cpu().numpy()
            preds[treatment] = pred.squeeze()
        return pd.DataFrame(preds).transpose()


class MultipleGenericModel(PrognosisModel):
    """ This class is used to load the best model for a given folder_name.
        The folder must be under src/forecasting/weights. The folder must contain the results of the training in a csv.
        This class automatically finds the weights for the best model and the weights. If you trained a model with
        the train script
    """

    def __init__(self, folder_name):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.folder_name = folder_name
        self.folder_path = join(PROGNOSIS_WEIGHTS, folder_name)
        self.result_files = [f for f in os.listdir(self.folder_path) if f.endswith(".csv")]
        self.weights_dir = [f for f in os.listdir(self.folder_path) if os.path.isdir(join(self.folder_path, f))]
        self.possible_months = [int(f.split("_")[-1].split("months")[0]) for f in self.result_files]
        self.results_paths = {months: join(self.folder_path, f"train_results_{months}.csv")
                              for months in self.possible_months}
        self.weights_dir_paths = {}
        for months in self.possible_months:
            for directory in self.weights_dir:
                if f"{months}months" in directory:
                    self.weights_dir_paths[months] = join(self.folder_path, directory)

    def load_model(self, months):
        """Automatically load the best model for the given time target in months."""
        results_csv = os.path.join(self.folder_path, f"train_results_{months}months.csv")
        results_df = pd.read_csv(results_csv)
        best_model = results_df.iloc[results_df["Best loss"].idxmin()]
        model = AdvancedLSTMRegressor(35,
                                      out_features=1,
                                      hidden_size=int(best_model["Hidden size"]),
                                      num_layers=int(best_model["Number of Layers"]),
                                      dropout_rate=float(best_model["Dropout Rate"]),
                                      device=self.device)
        load_path = os.path.join(self.weights_dir_paths[months],
                                 f"AdvancedLSTMRegressor_"
                                 f"hidden_{best_model['Hidden size']}_"
                                 f"layers_{best_model['Number of Layers']}_"
                                 f"lr_{best_model['Learning Rate']}_"
                                 f"redf_{best_model['LR Reduction Factor']}_"
                                 f"batch_{best_model['Batch Size']}_"
                                 f"loss_L1Loss_"
                                 f"drop_{best_model['Dropout Rate']}.pt")
        model.load_state_dict(torch.load(load_path, map_location=self.device))
        return model

    def load_scalers(self, months):
        x_scaler = load_scaler(join(PROGNOSIS_WEIGHTS,
                                    f"SmoothedInterpolator_90_{self.folder_name}_{months}months_x_scaler.pkl"))
        y_scaler = load_scaler(join(PROGNOSIS_WEIGHTS,
                                    f"SmoothedInterpolator_90_{self.folder_name}_{months}months_y_scaler.pkl"))
        return x_scaler, y_scaler

    def predict_all_treatment(self, df):
        preds = defaultdict(list)
        for months in self.possible_months:
            model = self.load_model(months)
            for treatment in IvomDrugs.keys():
                df.iloc[-1, 1] = IvomDrugs[treatment]  # Set the treatment for the current visit
                X = df.values
                x_scaler, y_scaler = self.load_scalers(months)
                X_scaled = x_scaler.transform(X)
                X = torch.from_numpy(X_scaled).float().unsqueeze(0).to(self.device)
                logger.debug(f"Sending input of shape {X.shape} to device {self.device}")
                pred = model(X).detach().cpu().numpy()
                preds[treatment].append(np.clip(y_scaler.inverse_transform(pred).squeeze(), 0, None))
        return pd.DataFrame(preds, index=[f"{i} months" for i in self.possible_months]).transpose()


class DemoForecastingModel(PrognosisModel):
    """ Class for the demo showcase. This class is used to show the functionality of the forecasting model in the
        demo. It returns random predictions for the treatments."""

    def __init__(self):
        super().__init__()

    def predict_all_treatment(self, df):
        preds = defaultdict(list)
        last_VA = df.iloc[-1, 17]
        for month in [1, 3, 6, 9, 12]:
            for treatment in IvomDrugs.keys():
                if month == 1:
                    preds[treatment].append(
                        max(
                            last_VA + np.random.randint(
                                -100 if treatment == "Control" else 50,
                                0 if treatment == "Control" else 50
                            ) / 100
                            , 0
                        )
                    )
                else:
                    preds[treatment].append(
                        max(
                            preds[treatment][-1] + np.random.randint(
                                -50 if treatment == "Control" else 50,
                                0 if treatment == "Control" else 50
                            ) / 100
                            , 0
                        )
                    )
                preds[treatment].append(np.random.rand(1, 1))
        return pd.DataFrame(preds, index=[f"{i} months" for i in [1, 3, 6, 9, 12]]).transpose()
