# This file is part of Ophthalmo-CDSS.
# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# See https://creativecommons.org/licenses/by-nc-sa/4.0/ for details.

import os
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import config
import src.visualisation as viz
from logging import getLogger
from visual_components.util import *
from src.recommendation import check_series


logger = getLogger(__name__)


def TopBar():
    # Health and meta data
    patient = get_session_state("patient")
    visit_data_full_both_df = get_session_state("visit_data_full_both_df")
    joined_df = get_session_state("joined_df")
    ivom_timeline = get_session_state("ivom_timeline")
    date_picker = get_session_state("date_picker")
    age = try_get_value(patient, "age")
    gender = try_get_value(patient, "gender")
    smoker = try_get_value(visit_data_full_both_df, "Finding of tobacco smoking behavior")
    weight = try_get_value(visit_data_full_both_df, "Body weight")
    height = try_get_value(visit_data_full_both_df, "Body height")
    blood_pressure = try_get_value(visit_data_full_both_df, "Blood pressure")
    diastolic, systolic = blood_pressure.split("/") if blood_pressure != "--" else ("--", "--")

    systolic = int(systolic) if systolic != "--" else "--"
    systolic = systolic * 10 if systolic != "--" and systolic < 20 else "--"
    diastolic = int(diastolic) if diastolic != "--" else "--"
    diastolic = diastolic * 10 if diastolic != "--" and diastolic < 20 else "--"
    if weight != "--" and height != "--":
        bmi = weight / (height / 100) ** 2
    else:
        bmi = "--"
    # Displays the top bar
    top_bar = st.columns([1, 1, 1, 2, 2, 5, 2])
    with top_bar[0]:
        st.image(Image.open(os.path.join(config.ICONS, "DFKI-Logo.png")), width=75)
        st.image(Image.open(os.path.join(config.ICONS, "OAI-logo.png")), use_column_width=True)
    with top_bar[1]:
        st.markdown(f"Age: {age}")
        st.markdown(f"Gender: {gender}")
        diagnoses = joined_df["Diagnosis - Use case"].unique()
        st.info(f"Diagnosed: {' & '.join([d for d in diagnoses if not pd.isna(d)]) if len(diagnoses) > 0 else '--'}")
    with top_bar[2]:
        if bmi == "--":
            st.markdown(f"Weight: {weight}")
            st.markdown(f"Height: {height}")
            st.markdown(f"BMI: {bmi}")
        else:
            st.markdown(f"Weight: :{'green' if bmi < 25 else ('orange' if bmi < 30 else 'red')}[{weight:.2f}kg]")
            st.markdown(f"Height: {height:.2f}cm")
            st.markdown(f"BMI: :{'green' if bmi < 25 else ('orange' if bmi < 30 else 'red')}[{bmi:.2f}]")
    with top_bar[3]:
        st.markdown(f"Smoking Status: :{'red' if smoker.lower() == 'smoker' else 'green'}[{smoker}]")
        systolic_color = "red" if systolic != "--" and (systolic > 140 or systolic < 90) else "green"
        diastolic_color = "red" if diastolic != "--" and (diastolic > 90 or diastolic < 60) else "green"
        st.markdown(f"Blood Pressure: :{systolic_color}[{systolic}] / :{diastolic_color}[{diastolic}]")
    with top_bar[4]:
        # st.markdown(f"Treatment Status on {str(date_picker)[:-8]}:")
        sequence = get_session_state("model_input")
        in_series, series_ivom, n_ivom = check_series(sequence)
        series_ivom_str = list(config.IvomDrugs.keys())[int(series_ivom)]
        days_since_last_treatment = sequence["Days since last treatment"].iloc[0]
        if in_series:
            st.success(f"### {n_ivom}/3 IVOMs \n Series with {series_ivom_str}.\n Days since last treatment: {days_since_last_treatment}.")
        else:
            if days_since_last_treatment == -1:
                st.info(f"### Not in series.\n Patient has not been treated yet.")
            else:
                st.info(f"### Not in series.\n Days since last treatment: {days_since_last_treatment}.")
    with top_bar[-2]:
        if get_session_state("compare_date", False):
            dates = [date_picker, st.session_state["compare_date"]]
        else:
            dates = [date_picker]
        st.plotly_chart(viz.get_ivom_timeline(ivom_timeline, dates), use_container_width=True)
    with top_bar[-1]:
        try:
            counts_df = pd.DataFrame({
                "Drug": ivom_timeline[config.DatabaseKeys.drug].unique(),
                "Count": ivom_timeline[config.DatabaseKeys.drug].value_counts()})

            def color(val):
                if val in config.med_color_map.keys():
                    color = config.med_color_map[val]
                else:
                    color = "rgba(0, 0, 0, 0)"
                return f'background-color: {color}'

            st.dataframe(counts_df.style.applymap(color), hide_index=True,
                         use_container_width=True)
        except KeyError:
            st.info("No IVOMs given yet.")

