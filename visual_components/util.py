# This file is part of Ophthalmo-CDSS.
# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# See https://creativecommons.org/licenses/by-nc-sa/4.0/ for details.

import os
from typing import Union, List, Literal

import streamlit as st
import numpy as np
import pandas as pd
from logging import getLogger

import src.visualisation as viz
from src.data import database_access as db
from src.util import timing

logger = getLogger(__name__)


def add_to_session_state(key, value):
    """Adds a value to the session state."""
    if key not in st.session_state:
        st.session_state[key] = value


def update_session_state(key, value):
    """Updates a value in the session state."""
    if key in st.session_state:
        st.session_state[key] = value
    else:
        add_to_session_state(key, value)


def get_session_state(key, default=False):
    try:
        return st.session_state[key]
    except KeyError:
        return default


def reset_session_state():
    """Resets the session state."""
    st.session_state.clear()


def get_eye_side(sequence):
    try:
        col = sequence["Left eye - Distance visual acuity"]
        return "Left eye"
    except KeyError:
        return "Right eye"


def prepare_for_visit_diff(series: pd.Series):
    for i in range(len(series)):
        if type(series[i]) is set:
            if len(series[i]) == 0:
                series[i] = ""
            else:
                data = np.array(list(series[i]))
                locations = data[:, 0]
                qualifiers = data[:, 1]
                to_remove = []
                for j, (location, qualifier) in enumerate(list(zip(locations, qualifiers))):
                    if location == "General":
                        if np.any(locations != "General"):
                            if np.logical_and(locations != "General", qualifiers == qualifier).any():
                                to_remove.append(j)
                            elif qualifier != "":
                                locations[j] = ""
                            else:
                                to_remove.append(j)
                        else:
                            locations[j] = "Yes"
                    elif location == "General" and qualifier != "":
                        locations[j] = ""
                locations = np.delete(locations, to_remove)
                qualifiers = np.delete(qualifiers, to_remove)
                series[i] = "| ".join([f"{location} {qualifier}" for location, qualifier in zip(locations, qualifiers)])
    return series


def clear_cache():
    """Clears the cache of the dashboard."""
    st.cache_resource.clear()
    st.cache_data.clear()
    logger.info("Cleared cache.")


def try_get_value(df, key, default="--"):
    """Tries to get a value from a DataFrame and returns a default value if it fails."""
    try:
        return df[key].dropna().values[0]
    except Exception as e:
        return default


def get_query_param_index_safe(value_list, key):
    """Safely get a query parameter."""
    try:
        return value_list.index(st.query_params[key])
    except Exception:
        return 0


def reset_query():
    """Resets the query parameters."""
    st.query_params.clear()
    st.query_params["institute"] = get_session_state("institute")
    st.query_params["usecase"] = get_session_state("usecase")
    st.query_params["patient_id"] = get_session_state("patient_id")
    st.query_params["side"] = get_session_state("side")
    st.query_params["date"] = get_session_state("date_picker")
    st.query_params["vol"] = get_session_state("main_date")
    logger.info("Query parameters reset.")


@timing
@st.cache_data
def get_patients_cached(filter_usecase: Union[str, List[str], None] = None,
                        institute: Literal["All", "AKS", "AZM"] = "All") -> pd.DataFrame:
    """
        Query all rows in the patients table
        :param institute: The institute which patient data should be from. Can be either 'All', 'AKS', 'AZM'.
        :param filter_usecase: Which usecase to filter for. Can be either 'AMD', 'DR', a list of strings or None.
        :return: List of dictionaries containing dictionaries of patient data
        """
    df = db.get_patients(filter_usecase)
    if institute != "All":
        df = df[df["id"].str.contains(institute.lower())]
    return df


@timing
@st.cache_data
def get_visit_data_cached(patient_id, side="left"):
    df = db.get_visit_details_of_patient(patient_id, side)
    # Convert 'visdate' column to datetime type
    df['visdate'] = pd.to_datetime(df.index)
    return df


@timing
@st.cache_data
def get_medicamentation_per_side_cached(patient_id, side):
    # df = pd.DataFrame(db.patient_get_visit_data(patient_id, "left", limit=None))
    return db.get_medicamentation_per_side(patient_id, side)


@timing
@st.cache_data
def get_vol_info_cached(patient_id, eye_side=None, min_slices=0):
    return db.get_vol_infos(patient_id, eye_side, min_slices=min_slices)


@timing
@st.cache_data
def download_vol_cached(file_uri, show_case=0):
    return db.download_vol(file_uri, show_case)


@timing
@st.cache_data
def check_credentials():
    return False


@timing
@st.cache_resource
def get_3D_plot(file_uri, _vol):
    return viz.get_3D_plot(_vol)
