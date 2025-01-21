# This file is part of Ophthalmo-CDSS.
# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# See https://creativecommons.org/licenses/by-nc-sa/4.0/ for details.

import random
from unittest import TestCase
from config import ROOT, UseCases
from streamlit.testing.v1 import AppTest
from logging import basicConfig
from src.data.database_access import get_patients, get_visit_dates_of_patient
import os
import warnings

warnings.simplefilter(action="ignore")
basicConfig(level="CRITICAL")


class TestGeneral(TestCase):
    path = os.path.join(ROOT, "Home.py")
    at = AppTest.from_file(path)

    def test_run(self):
        self.at.run(timeout=60)
        assert not self.at.exception, "Exception raised while running Home.py!"
        print("Home.py ran successfully!")


class TestDifferentConfigurations(TestCase):
    path = os.path.join(ROOT, "Home.py")
    at = AppTest.from_file(path)
    configurations = []
    for insti in ["AZM", "AKS"]:
        for usecase in list(UseCases.keys()):
            patients = get_patients(usecase)
            patients = patients[patients["id"].str.contains(insti.lower())]
            try:
                random_patient = patients.sample(3)
            except ValueError:
                random_patient = patients
            for i, (_, patient) in enumerate(random_patient.iterrows()):
                side = random.choice(["Left", "Right"])
                visit_dates = get_visit_dates_of_patient(patient["id"], side=side)
                configurations.append((insti,
                                       usecase,
                                       patient["id"],
                                       side,
                                       random.choice(visit_dates),
                                       ["IRSLO", "VOL Slices", "3D"][i]
                                       ))

    def test_configs(self):
        print(f"Testing {len(self.configurations)} different configurations.")
        for i, config in enumerate(self.configurations):
            institute, usecase, patient, side, date, view = config
            print(f"Testing configuration {i + 1}/{len(self.configurations)}:"
                        f"\n - Institute: {institute}"
                        f"\n - Usecase: {usecase}"
                        f"\n - Patient: {patient}"
                        f"\n - Side: {side}"
                        f"\n - Date: {date}"
                        f"\n - View: {view}")
            self.at.run(timeout=60)
            self.at.sidebar.radio[0].set_value(institute)
            self.at.sidebar.selectbox[0].set_value(usecase)
            self.at.run(timeout=60)
            self.at.sidebar.selectbox[1].set_value(patient)
            self.at.sidebar.radio[1].set_value(side)
            self.at.radio[0].set_value(view)
            #self.at.sidebar.selectbox[2].set_value(date)
            self.at.run(timeout=60)
            assert not self.at.exception, "Exception raised while running configuration!"
            print(f"Configuration {i + 1} ran successfully!")

