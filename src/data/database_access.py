import io
import logging
import sqlite3
import sys

from typing import Union, List
from contextlib import contextmanager
from datetime import datetime
from sqlite3 import Error

import pandas as pd
import numpy as np

import config
import os
from config import DATABASE, UseCases
from collections import defaultdict

from src.data.preprocess import preprocess_query_result
from src.oct import OCT
from src.util import timing

logger = logging.getLogger(__name__)


def dictionarify(columns, rows):
    """Converts the given columns and rows to a list of dictionaries."""
    data = []
    for row in rows:
        d = dict(zip(columns, row))
        data.append(d)
    return data


def get_eye_restriction(side, prefix=""):
    """Computes the SQL restriction for the given eye side and prefix."""
    if side and side.lower() in ['right', 'right side', 'right eye']:
        return f" and {prefix + '.' if prefix else ''}eye_side  IN ('Right eye', 'Both eyes') "
    elif side and side.lower() in ['left', 'left side', 'left eye']:
        return f" and {prefix + '.' if prefix else ''}eye_side IN ('Left eye', 'Both eyes') "
    else:
        return ""


def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        logger.debug("Connection to DB: ok")
    except Error as e:
        logger.error(e)
    return conn


@contextmanager
def open_db_connection(file=DATABASE, commit=False):
    connection = sqlite3.connect(file)
    cursor = connection.cursor()
    try:
        yield cursor
    except Exception as err:
        error, = err.args
        sys.stderr.write(error)
        connection.rollback()
        raise err
    else:
        if commit:
            cursor.close()
            connection.commit()
        else:
            cursor.close()
            connection.rollback()
    finally:
        cursor.close()
        connection.close()


def query_database(query: str):
    """Queries the database with the given SQLite query string."""
    # logger.warning(f"Executing Query: {query}")
    with open_db_connection() as cur:
        cur.execute(query)
        columns = [d[0] for d in cur.description]
        rows = cur.fetchall()
    if not columns or not rows:
        logger.warning(f"Query returned no results: {query}")
    return columns, rows


@timing
def get_patients(filter_usecase: Union[str, List[str], None] = None) -> pd.DataFrame:
    """
        Query all rows in the patients table
        :param filter_usecase: Which usecase to filter for. Can be either 'AMD', 'DR', a list of strings or None.
        :return: List of dictionaries containing dictionaries of patient data
    """
    if filter_usecase is None or isinstance(filter_usecase, list):
        # Filter is either None or a list of filters.
        # If none use all possible use cases, else use the given list
        usecases = UseCases if filter_usecase is None else filter_usecase
        return pd.concat([get_patients(usecase) for usecase in usecases], ignore_index=True)

    filter_usecase = UseCases[filter_usecase]

    columns, rows = query_database(f"SELECT * "
                                   f"FROM patients "
                                   f"WHERE use_case={filter_usecase} "
                                   f"AND birthday IS NOT NULL "
                                   f"AND gender IS NOT NULL "
                                   f"order by id")
    data = defaultdict(list)
    for row in rows:
        d = dict(zip(columns, row))
        if 'birthday' in d and d['birthday']:
            dob = datetime.strptime(d['birthday'], '%Y-%m-%d')
            age_secs = datetime.now() - dob
            d['age'] = divmod(age_secs.total_seconds(), 31536000)[0]
        else:
            d['age'] = None
        for key in columns + ['age']:
            data[key].append(d[key])
    df = pd.DataFrame.from_dict(data)
    return df


def get_patient_data(patient_id):
    """ Get the data of a patient with the given ID."""
    columns, rows = query_database(f"SELECT * "
                                   f"FROM patients "
                                   f"WHERE id='{patient_id}' ")
    data = defaultdict(list)
    for row in rows:
        d = dict(zip(columns, row))
        for key in columns:
            data[key].append(d[key])
    df = pd.DataFrame.from_dict(data)
    return df


def get_medicamentation_per_side(patient_id, side, limit=-1):
    """Returns the medicamentation of the patient with the given side."""
    logger.info("Getting medicamentation of patient %s with side %s", patient_id, side)
    columns, rows = query_database(f" SELECT strftime('%Y-%m-%d', date_iso) AS timestamp, "
                                   f" x.param_description_en, l.param_value, x.python_type "
                                   f" FROM patient_labels l, xploit_parameters x  "
                                   f" WHERE x.xploit_param_id = 1710669398 "
                                   f" and timestamp IN "
                                   f"               (SELECT strftime('%Y-%m-%d', date_iso) timestamp"
                                   f"               FROM patient_labels "
                                   f"               WHERE patient_id='{patient_id}' "
                                   f"               {get_eye_restriction(side)} "
                                   f"               GROUP BY timestamp "
                                   f"               ORDER BY timestamp DESC "
                                   f"               LIMIT {limit}) "
                                   f" and x.xploit_param_id = l.param_id "
                                   f" and l.patient_id='{patient_id}' "
                                   f" {get_eye_restriction(side)} "
                                   f" order by timestamp DESC ")

    df = preprocess_query_result(columns, rows, "Physical findings of Eye Narrative - ")
    # Convert 'visdate' column to datetime type
    df[config.DatabaseKeys.visit_date] = pd.to_datetime(df.index)
    return df


@timing
def get_visit_details_of_patient(patient_id, side, limit=-1):
    """ Joins the patient_labels and xploit_parameters tables to get the visit details of the patient with
        the given side. Returns a DataFrame with the visit details order by descending timestamp."""
    logger.info("Getting visit details of patient %s with side %s", patient_id, side)
    columns, rows = query_database(f" SELECT strftime('%Y-%m-%d', date_iso) AS timestamp, "
                                   f" x.param_description_en, l.param_value, x.python_type, x.unit "
                                   f" FROM patient_labels l, xploit_parameters x  "
                                   f" WHERE x.xploit_param_id = l.param_id "
                                   f" and l.patient_id ='{patient_id}' "
                                   f" {get_eye_restriction(side)} "
                                   f" and timestamp IN "
                                   f"               (SELECT strftime('%Y-%m-%d', date_iso) timestamp"
                                   f"               FROM patient_labels "
                                   f"               WHERE patient_id='{patient_id}' "
                                   f"               {get_eye_restriction(side)} "
                                   f"               GROUP BY timestamp "
                                   f"               ORDER BY timestamp DESC "
                                   f"               LIMIT {limit}) "
                                   f" order by timestamp DESC ")
    df = preprocess_query_result(columns, rows, "Physical findings of Eye Narrative - ")
    return df


def get_visit_dates_of_patient(patient_id, side, limit=-1):
    """ Get all visit dates of the patient with the given side."""
    columns, rows = query_database(f"SELECT distinct(date_iso) timestamp "
                                   f" FROM patient_labels l"
                                   f" WHERE l.patient_id='{patient_id}' "
                                   f" {get_eye_restriction(side)} "
                                   f" order by timestamp DESC"
                                   f" limit {limit} ")
    return [pd.to_datetime(row[0].split("+")[0]) for row in rows]


def get_vol_infos(patient_id, eye_side=None, include_angio=False,
                  set_index="date_of_visit", min_slices=0, with_fluids_only=False):
    """ Get the OCT information of a patient.
    :param patient_id: The ID of the patient.
    :param eye_side: The side of the eye. Can be 'left', 'right', 'left side', 'right side', 'left eye', 'right eye'.
    :param include_angio: If set to True, the query will include angio images.
    :param set_index: The column to set as index in the returned DataFrame.
    :param min_slices: The minimum number of slices the VOL must have.
    :param with_fluids_only: If set to True, only the VOLs with fluids will be returned.
    :returns vol_df: The DataFrame with the VOL information."""
    eye_restriction = get_eye_restriction(eye_side)
    angio_restriction = " AND o.file_uri LIKE '%Angio%' " if include_angio else ""
    columns, rows = query_database(f"SELECT slices, strftime('%Y-%m-%d', date_of_visit) AS date_of_visit"              
                                   f", eye_side, f.* "
                                   f" FROM oct_files o, fluids f, visits v"
                                   f" WHERE o.visit_id=v.id "
                                   f" and o.file_uri = f.file_uri"
                                   f" and o.patient_id='{patient_id}'"
                                   f" and o.slices >= {min_slices}"
                                   + angio_restriction
                                   + eye_restriction
                                   + f" ORDER by date_of_visit DESC "
                                   )
    vol_info = defaultdict(list)
    for entry in dictionarify(columns, rows):
        for k, v in entry.items():
            vol_info[k].append(v)
    vol_df = pd.DataFrame(vol_info)
    if vol_df.empty:
        logger.warning(f"No VOLs (with min {min_slices}) found for patient {patient_id} and side {eye_side}.")
        return vol_df
    vol_df['date_of_visit'] = vol_df['date_of_visit'].astype('datetime64[ns]')
    for column in vol_df.columns:
        if column not in ['file_uri', 'date_of_visit', 'eye_side']:
            vol_df[column] = vol_df[column].astype('float')
    if set_index:
        return vol_df.set_index(set_index)
    else:
        return vol_df


def download_vol(file_uri, show_case=0, remove_after=False):
    """ Gets the VOL from the give file_uri.
    :param file_uri: The file_uri of the VOL to download. This is the file_uri from Xpl0it, not an actual file path
    :param show_case: If set to 1 or 2, the showcase VOLs will be used instead of the given file_uri.
    :param remove_after: If set to True, the downloaded VOL will be removed after loading. Deprecated as all VOLs
        are now downloaded since Xpl0it is down. Will not be used.
    :returns VOL: The loaded VOL object. Instance of class src.vol.VOL."""
    if show_case:
        logger.info(f"Using showcase VOL {OCT_TEST_1_FILE if show_case == 1 else OCT_TEST_2_FILE}")
        vol_file_save_path = OCT_TEST_1_FILE if show_case == 1 else OCT_TEST_2_FILE
    else:
        vol_file_save_path = os.path.join(config.VOL_DATA, file_uri)
        # download if vol file does not exist
        if not os.path.exists(vol_file_save_path):
            raise FileNotFoundError(f"File {vol_file_save_path} has not been downloaded from Xpl0it.")
    logging.info(f"Loading VOL from {vol_file_save_path}.")
    with io.FileIO(vol_file_save_path) as buffer:
        # buffer.read(vol_file_save_path)
        buffer.seek(0)
        vol = OCT(buffer)
    return vol


def get_joined_vol_and_patient_details(patient_id, side):
    """ Get the joined meta data, OCT and patient details of the patient with the given side."""

    def replace_empty(x):
        if x.empty:
            return np.nan
        return x.values[0]

    patient_data = get_patient_data(patient_id)
    details_df = get_visit_details_of_patient(patient_id, side).sort_index(
        ascending=False)
    details_df['date'] = pd.to_datetime(details_df.index).date
    # Identify numerical and categorical columns
    numerical_cols = details_df.select_dtypes(include='number').columns
    categorical_cols = details_df.select_dtypes(exclude='number').columns
    agg_funcs = {col: 'mean' for col in numerical_cols}
    agg_funcs.update({col: lambda x: pd.Series.mode(x) for col in categorical_cols if col != 'date'})
    details_df = details_df.groupby('date').agg(agg_funcs).reset_index()
    vol_df = get_vol_infos(patient_id,
                           side.capitalize() + " eye",
                           with_fluids_only=True,
                           include_angio=False,
                           )
    if vol_df.empty:
        return pd.DataFrame()
    vol_df['date'] = pd.to_datetime(vol_df.index).date
    idx = vol_df.groupby('date')['slices'].idxmax()
    vol_df = vol_df.loc[idx]
    vol_df = vol_df.map(lambda x: replace_empty(x))
    details_df = details_df.map(lambda x: replace_empty(x))
    merged_df = pd.merge(details_df, vol_df, how="left", on="date").set_index("date")
    birthday = pd.to_datetime(patient_data['birthday'].values[0])
    dates = pd.to_datetime(merged_df.index)
    merged_df['Age'] = ((dates - birthday).days / 365.25).astype('int64')
    merged_df['Gender'] = patient_data['gender'].values[0]
    merged_df['Use case'] = patient_data['use_case'].values[0]
    merged_df.columns = [col.replace("Right eye - ", "").replace("Left eye - ", "") for col in merged_df.columns]
    return merged_df
