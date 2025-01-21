# This file is part of Ophthalmo-CDSS.
# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# See https://creativecommons.org/licenses/by-nc-sa/4.0/ for details.

from collections import defaultdict
from logging import getLogger
from typing import Union, Callable, Tuple, Dict, List

from sklearn.preprocessing import LabelEncoder

import config
from config import EyeStructures, Qualifiers, EyeSymptomes, UseCases, IvomDrugs
from src.data.interpolation import Interpolator, SmoothedInterpolator, ConstantInterpolator

import pandas as pd
import numpy as np

from src.util import timing

logger = getLogger(__name__)


def preprocess_query_result(columns, rows, param_prefix):
    """
    Preprocesses the parameters from the database. This includes removing the prefix from the parameter name,
    converting the value to the correct type and removing the unit from the value. The unit is returned separately.
    :param columns: The column names of the query result.
    :param rows: The rows of the query result.
    :param param_prefix: The prefix of the parameter that should be removed.
    :return: The preprocessed parameter name, value and unit.
    """
    logger.info(f"Preprocessing {len(rows)} rows with {len(columns)} columns.")
    data = defaultdict(dict)
    dtypes = {}
    for row in rows:
        for d, d1 in preprocess_row(columns, row, param_prefix):
            data[d['timestamp']][d1[0]] = d1[1]
            if d1[0] not in dtypes:
                dtypes[d1[0]] = d1[2]
            else:
                if not dtypes[d1[0]] == d1[2]:
                    dtypes[d1[0]] = 'object'
    df = pd.DataFrame.from_dict(data).transpose().infer_objects(copy=False)
    return df


def preprocess_row(columns, row, param_prefix):
    """
    Preprocesses the parameters from the database. This includes removing the prefix from the parameter name,
    converting the value to the correct type and removing the unit from the value. The unit is returned separately.
    :param columns: The column names of the query result.
    :param row: The row of the query result.
    :param param_prefix: The prefix of the image parameter.
    :return: The preprocessed parameter name, value and unit.
    """
    len_prefix = len(param_prefix)
    d = dict(zip(columns, row))
    if d['param_description_en'].startswith(param_prefix):
        desc = d['param_description_en'][len_prefix:]
    else:
        desc = d['param_description_en']
    type_start = desc.find("(")
    type_end = desc.find(")")
    if type_start > -1 and type_end > -1:
        if desc[type_start + 1:type_end].lower() == 'datetime':
            k = desc  # Do not remove type if it says datetime. This is needed for Intraocular pressure finding as they
            # have the same description but different types, and you cannot cast a datetime value to a float.
        else:
            # Else remove the type indicator from the key as it is unnecessary.
            k = desc[:type_start - 1]
    else:
        # If there is no type indicator, just use the description as key.
        k = desc
    k = k.lstrip().rstrip()  # Remove leading and trailing whitespaces
    # check type of parameter
    if d['python_type'] == 'bool':
        if d['param_value'] == '1':
            yield d, (k, True, 'bool')
        else:
            yield d, (k, False, 'bool')
    elif d['python_type'] == 'datetime':
        yield d, (k, d['param_value'][:10], 'datetime64[D]')
    elif d['python_type'] == 'float':
        if "Distance visual acuity" in k:
            value = float(d['param_value'])
            if config.va_unit == "logmar" and 'unit' in d and d['unit'].lower() == "decimal":
                value = -np.log(value)
            elif config.va_unit == "decimal" and 'unit' in d and d['unit'].lower() == "logmar":
                value = np.exp(-value)
            value = min(value, 3.0)
            value = max(value, -0.3)
            yield d, (k, value, 'float32')
        else:
            yield d, (k, float(d['param_value']), 'float32')
    elif d['python_type'] == 'str':
        if "Intraocular pressure finding" in k:
            if "Both eyes" in k:
                left, right = d['param_value'].split(" / ")
                k_left = k.replace("Both eyes", "Left eye")
                k_right = k.replace("Both eyes", "Right eye")
                yield d, (k_left, float(left), 'float32')
                yield d, (k_right, float(right), 'float32')
            else:
                try:
                    yield d, (k, float(d['param_value']), 'float32')
                except ValueError:
                    logger.warning(f"Could not convert {d['param_value']} to float for {k}")
        else:
            try:
                yield d, (k, float(d['param_value']), 'float32')
            except ValueError:
                yield d, (k, d['param_value'], 'object')  # in pandas this is a string
    else:
        yield d, (k, d['param_value'], d['python_type'])


def merge_multiple_columns_by_key(df: pd.DataFrame,
                                  merge_tups: List[
                                      Tuple[Union[str, List[str]], str, Callable[[pd.DataFrame], pd.Series]]],
                                  remove_merged_columns=True,
                                  default_value=None) -> pd.DataFrame:
    """ For every key given merges all columns that contain the key into one new column named after new_keys using
        the given merge functions.
        :param default_value: The default value to fill the new column with if no column with the key is found.
        :param df: The dataframe to merge.
        :type df: pd.DataFrame
        :param merge_tups: A list of tuples containing the keys to merge, the new key and the merge function.
        :type merge_tups: list[tuple[Union[str, list[str]], str, Callable[[pd.DataFrame], pd.Series]]]
        :param remove_merged_columns: If True, removes the merged columns from the dataframe.
        :type remove_merged_columns: bool
        :return: The merged dataframe.
    """
    df = df.copy()
    merged_indices = set()
    for keys, new_key, merge_function in merge_tups:
        if new_key is None and isinstance(keys, str):
            new_key = keys
        elif new_key is None and isinstance(keys, list):
            raise ValueError(f"New key must be provided if keys is a list. Given keys: {keys} New key: {new_key}")
        if merge_function is None:
            logger.debug(f"No merge function was provided for {keys} using default OR function.")
            merge_function = MergeFunctions.merge_bools_with_OR
        df, merged = merge_columns_by_key(df, keys, merge_function=merge_function,
                                          new_key=new_key, default_value=default_value)
        merged_indices.update(merged)
    if remove_merged_columns:
        df = df.drop(columns=df.iloc[:, list(merged_indices)].columns, inplace=False)
    return df


def add_columns_with_default_values_if_not_present(df: pd.DataFrame, names_to_values: Dict[str, str]):
    """ Adds a column to the dataframe with the given column name and fills it with the given default value."""
    df = df.copy()
    for column_name, default_value in names_to_values.items():
        if column_name not in df.columns:
            df[column_name] = default_value
        elif df[column_name].isna().values.all():
            df[column_name] = default_value
    return df


def interpolate_and_pad_df(df: pd.DataFrame, interpolator: Interpolator = None) \
        -> Tuple[pd.DataFrame, Dict[str, Callable]]:
    """ Interpolate and pad the dataframe. This function interpolates columns depending on their data type.
        Numerical columns are interpolated using a smoothed interpolator or the given interpolator. If there are not
        enough values to interpolate, a constant interpolator is used. If there are no values and the column represents
        the volume or number of fluids or PED, a constant value of 0.0 is used. If there are no values and the column
        represents the total retinal thickness, a constant value of 600.0 is used.
        Object columns are interpolated using forward fill first followed by backward fill
        (this also pads the dataframe on the bounds)."""
    df = df.copy().sort_index(ascending=True)  # Sort by date. Important for interpolation.
    logger.info(f"Interpolating and padding dataframe with {len(df)} rows and {len(df.columns)} columns.")
    interpolators = {}  # For each column, store the interpolator function, such that you can later sample new values
    for column, dtype in zip(df.columns, df.dtypes):
        if dtype in ["float32", "float64"] and not (column == "Use case"):
            column_df = df[column]
            dates = column_df.index
            values = column_df
            non_nan_values = values[~np.isnan(values)]  # Remove nan values
            non_nan_dates = dates[~np.isnan(values)]  # Remove dates that correspond to nan values in column
            if len(non_nan_values) >= 2:
                if interpolator is None:
                    interpolator = SmoothedInterpolator(moving_average_window=90)
                else:
                    interpolator = interpolator.new()
                logger.debug(f"Interpolating {column} of type float with {interpolator.__name__}. ")
                if "Distance visual acuity" in column:
                    interpolator.set_value_range(0., 2.)
                interpolator.set_fit_data(non_nan_dates, non_nan_values)
                interpolators[column] = interpolator
                preds = interpolator(list(dates))
            elif len(non_nan_values) == 1:
                logger.debug(f"Could not interpolate {column} with {interpolator.__name__}. "
                             f"Using constant interpolation.")
                interpolator_other = ConstantInterpolator(non_nan_dates, non_nan_values)
                interpolators[column] = interpolator_other
                preds = interpolator_other(dates)
            elif column in ["v_fluid", "n_fluid", "v_ped", "n_ped"]:
                interpolator_other = ConstantInterpolator([], [0.0])
                interpolators[column] = interpolator_other
                preds = interpolator_other(dates)
            elif column == "total_thickness":
                interpolator_other = ConstantInterpolator([], [600.0])
                interpolators[column] = interpolator_other
                preds = interpolator_other(dates)
            else:
                logger.error(f"Could not interpolate {column} because there are no non nan values.")
                raise ValueError(f"Could not interpolate {column} because there are no non nan values.")
            df[column] = preds
        elif dtype == 'object' or column == "Use case":
            if column == config.DatabaseKeys.drug:
                logger.debug(f"Filling {column} of type object/string with value 'Control'.")
                df[column] = df[column].fillna("Control")
            else:
                logger.debug(f"Interpolating {column} of type object/string with method ffill and bfill.")
                df[column] = df[column].interpolate(method='ffill', axis=0).interpolate(method='bfill', axis=0)
        else:
            logger.debug(f"Skipping {column} of type {dtype}.")
            pass
    return df, interpolators


def get_indices_containing_substring(string_list: List[str], substrings: Union[str, List[str]]):
    """ Returns a list of all indices of a list of strings that contain at least one substring from
        a list of substrings. You can also provide a single string as a substring."""
    returns = []
    for i, s in enumerate(string_list):
        if type(substrings) is list:
            try:
                if any([substring.lower() in s.lower() for substring in substrings]):
                    returns.append(i)
            except AttributeError:
                logger.debug(f"Substrings contains non string type: {substrings}!")
        try:
            if substrings.lower() in s.lower():
                returns.append(i)
        except AttributeError:
            logger.debug(f"Could not convert {s} to string")
    return returns


def merge_columns_by_key(df: pd.DataFrame, key: Union[str, List[str]],
                         merge_function: Callable[[pd.DataFrame], pd.Series],
                         new_key=None, default_value=None) -> Tuple[pd.DataFrame, List[int]]:
    """ Merges columns of a dataframe that contain a certain substring. The new column is named after the
        new_key parameter. If no new_key is provided, the key parameter is used. If there is only one column
        with the key, it is not merged. If there are no columns with the key, a new column with the new_key
        is created and filled with False.
        :param df: The dataframe to merge.
        :type df: pd.DataFrame
        :param key: The key(s) to merge. Can be a string or a list of strings.
        :type key: Union[str, list[str]]
        :param merge_function: The function to use for merging.
        :type merge_function: Callable[[pd.DataFrame], pd.Series]
        :param new_key: The name of the new column. If None is provided, the key parameter is used. Must be provided
            if key is a list.
        :type new_key: str
        :param default_value: The default value to fill the new column with if no column with the key is found.
        :type default_value: Any
        :return: The merged dataframe."""
    if type(key) is list and new_key is None:
        logger.error(f"Key is a list but no new_key was provided. Keys: {key}")
        raise ValueError(f"If key is a list, new_key must be provided. Keys: {key}")
    new_key = key if new_key is None else new_key
    merge_indices = get_indices_containing_substring(df.columns.tolist(), key)
    if len(merge_indices) > 1:
        logger.debug(f"Merging {merge_indices} to {new_key} with function {merge_function.__name__}.")
        merged_columns = df.iloc[:, merge_indices]
        if merge_function is None:
            raise Exception("No merge function was provided.")
        else:
            new_values = merge_function(merged_columns)
            if new_key in df.columns:
                # If the key is already in the dataframe, drop it and replace it with the new values.
                df = df.drop(columns=new_key, inplace=False)
            df[new_key] = new_values
    elif len(merge_indices) == 1:
        # There is only one column containing the key
        try:
            new_values = merge_function(df.iloc[:, merge_indices])  # Try to merge the column. Important if the
            # merge function changes the values (eg. from a list to a bool).
        except Exception as e:
            new_values = df.iloc[:, merge_indices]
        if new_key in df.columns:
            df = df.drop(columns=new_key, inplace=False)
        df[new_key] = new_values
    else:
        logger.debug(f"Could not merge {key} because there is no column with this name.")
        df[new_key] = default_value
    assert new_key in df.columns, f"Could not merge {key} to {new_key}"
    return df, merge_indices


def drop_all_columns(df: pd.DataFrame, keys):
    """ Drops all columns that contain a certain substring. """
    logger.debug(f"Dropping columns with keys {keys}")
    for key in keys:
        indices = get_indices_containing_substring(df.columns.tolist(), key)
        df = df.drop(columns=df.iloc[:, indices].columns, inplace=False)
    return df


class MergeFunctions:
    """ Class of static functions that contain different ways to merge annotations. Most are experimental and are
        not used in the final version."""

    @staticmethod
    def invert_bool_values_if_marked_with_no(df: pd.DataFrame):
        """ Function that inverts the boolean values of a dataframe if the column name contains the substring
            "- No" or " - Closed". """
        # Create a copy of the DataFrame to avoid modifying the original one
        new_df = df.copy()
        # Loop through each column in the DataFrame
        for column in df.columns:
            # Check if the column name contains the specified substrings
            if "- No" in column or " - Closed" in column:
                # Invert boolean values while leaving NaN values unchanged
                new_df[column] = new_df[column].apply(lambda x: not x if pd.notna(x) else x)
        return new_df

    @staticmethod
    def merge_bools_with_three_levels(df: pd.DataFrame):
        """ Merges boolean values with an OR function. """
        values = []
        inverted_df = MergeFunctions.invert_bool_values_if_marked_with_no(df)
        for index in df.index:
            absence_marked = False
            for column in df.columns:
                if "- No" in column or " - Closed" in column:
                    if not pd.isna(df.loc[index, column]):
                        absence_marked |= df.loc[index, column]
            if inverted_df.loc[index].any():
                # The lesion is marked present in at least one column
                values.append(1.0)
            elif absence_marked:
                # The lesion is marked as absent
                values.append(-1.0)
            else:
                # The lesion is marked neither present nor absent
                values.append(np.nan)
        return pd.Series(values, index=df.index)

    @staticmethod
    def merge_bools_with_OR(df: pd.DataFrame):
        """ Merges boolean values with an OR function. """
        df = MergeFunctions.invert_bool_values_if_marked_with_no(df)
        return df.any(axis="columns")

    @staticmethod
    def merge_bools_with_AND(df: pd.DataFrame):
        """ Merges boolean values with an AND function. """
        df = MergeFunctions.invert_bool_values_if_marked_with_no(df)
        return df.all(axis=1)

    @staticmethod
    def merge_to_structure_qualifier_tuple(df: pd.DataFrame):
        """ Merges the values of a dataframe to a set of tuples containing the structure and the qualifier."""
        df = MergeFunctions.invert_bool_values_if_marked_with_no(df)
        new_values = []
        for i in range(len(df)):
            row = df.iloc[i, :]
            row_values = set()
            for column in df.columns:
                if row[column]:
                    added = False
                    for structure in EyeStructures:
                        if structure.lower() in column.lower():
                            for qualifier in Qualifiers:
                                if qualifier.lower() in column.lower():
                                    row_values.add((structure, qualifier))
                                    added |= True
                            if not added:
                                row_values.add((structure, ""))
                                added |= True
                    for qualifier in Qualifiers:
                        if qualifier.lower() in column.lower():
                            row_values.add(("Yes", qualifier))
                            added |= True
                    if not added:
                        logger.debug(f"Could not find structure or qualifier for {column}.")
                        row_values.add(("Yes", ""))
            new_values.append(row_values)
        return pd.Series(new_values, index=df.index)


@timing
def preprocess_visit_details_of_patient_for_model(input_df,
                                                  target_metric: str = None,
                                                  target_months: int = None,
                                                  birthday=None,
                                                  gender=None,
                                                  without_targets=False,
                                                  default_usecase="AMD",
                                                  transform_to_categorical=True,
                                                  side="left",
                                                  interpolator=None):
    """ Apply preprocessing to the visit details of a patient. This includes interpolation and filling of
        missing numerical values, also merges categorical values that belong together. Also selects features
        that are relevant for the ML algorithm.
        :param input_df: The input dataframe with the visit details of a patient.
        :param target_metric: The target metric that should be forecasted.
        :param target_months: The number of months in the future the target metric should be forecasted.
        :param birthday: The birthday of the patient.
        :param gender: The gender of the patient.
        :param without_targets: If True, the target columns are not included in the output dataframe.
        :param default_usecase: The default use case for the patient.
        :param transform_to_categorical: If True, the categorical columns are transformed to numerical values.
        :param side: The side of the eye that is being treated.
        :param interpolator: The interpolator that is used to interpolate the data."""
    if not without_targets and (target_metric is None or target_months is None):
        raise ValueError("Target metric and target months must be specified if without_targets is False.")

    df = input_df.copy()
    defaults = {"Use case": default_usecase,
                "Gender": gender if gender is not None else "male",
                "Age": (pd.to_datetime(df.index, yearfirst=True) - pd.to_datetime(
                    birthday, yearfirst=True)).days / 365 if birthday is not None else 85.0,
                "Finding of tobacco smoking behavior": "Non-smoker",
                "Body weight": 80.0 if gender is None or gender == "male" else 70.0,
                "Body height": 170.0 if gender is None or gender == "male" else 160.0,
                "Intraocular pressure finding": 15.0,
                "v_fluids": 0.0, "n_fluids": 0.0, "v_ped": 0.0, "n_ped": 0.0,
                config.DatabaseKeys.drug: "Control",
                }

    # Merge the annotation columns to the new binary feature columns.
    merge_tups = [(keys, symptome, MergeFunctions.merge_to_structure_qualifier_tuple)
                  for symptome, keys in EyeSymptomes.items()]
    df = merge_multiple_columns_by_key(df, merge_tups, remove_merged_columns=False)

    # Rename side specific columns to general columns. Side information is stored in the "Treated side" column.
    df = df.rename(
        columns={f"{side.capitalize()} eye - Distance visual acuity": "Distance visual acuity",
                 f"{side.capitalize()} eye - Intraocular pressure finding": "Intraocular pressure finding"})

    # Add default values to the dataframe if they are not present.
    df = add_columns_with_default_values_if_not_present(df, defaults)

    # Drop all columns that are not relevant for the ML algorithm.
    selected_features = df[
        ['Use case',
         config.DatabaseKeys.drug,
         'Age', 'Gender', 'Finding of tobacco smoking behavior',
         'v_fluids', 'n_fluids', 'v_ped', 'n_ped', 'v_drusen', 'n_drusen',
         'ipl_thickness', 'opl_thickness', 'elm_thickness',
         'rpe_thickness', 'bm_thickness', 'choroidea_thickness',
         'total_thickness',
         "Distance visual acuity", "Intraocular pressure finding"]
        + list(EyeSymptomes.keys())]
    selected_features["BMI"] = df["Body weight"] / (df["Body height"] / 100) ** 2

    # Interpolate and pad the dataframe.
    try:
        selected_features, interpolators = interpolate_and_pad_df(selected_features, interpolator=interpolator)
    except ValueError as e:
        logger.error(f"Could not interpolate and pad dataframe. This happens when there are no numerical columns in "
                     f"the dataframe. Dropping all columns.")
        raise e
        return pd.DataFrame()
    logger.debug(f"Interpolators computed for {interpolators.keys()}")

    # Add date features and target variables to the dataframe.
    selected_features = add_date_features_and_target_variables(selected_features,
                                                               target_metric,
                                                               target_months,
                                                               only_date_features=without_targets,
                                                               interpolators=interpolators)
    if transform_to_categorical:
        selected_features = transform_categorical_cols_to_bool(selected_features)
    return selected_features


def add_tertiary_features_to_df(df: pd.DataFrame):
    merge_tups = [(keys, symptome, MergeFunctions.merge_bools_with_three_levels)
                  for symptome, keys in EyeSymptomes.items()]
    return merge_multiple_columns_by_key(df, merge_tups, remove_merged_columns=False, default_value=np.nan)


def add_time_features_to_df(df: pd.DataFrame):
    df = df.sort_index(ascending=True)  # Make sure the dataframe is sorted by date in ascending order.
    last_visit = df.index[0]
    first_visit = df.index[0]
    last_treatment = None
    first_treatment = None
    ivom_counter = 0
    df["Days since last visit"] = -1
    df["Days since last treatment"] = -1
    df["Days since first visit"] = 0
    df["Days since first treatment"] = 0
    df["Number of IVOMs"] = 0
    for vis_date in df.index:
        df.loc[vis_date, "Days since last visit"] = (vis_date - last_visit).days
        df.loc[vis_date, "Days since first visit"] = (vis_date - first_visit).days
        df.loc[vis_date, "Days since last treatment"] = (vis_date - last_treatment).days \
            if last_treatment is not None else -1
        df.loc[vis_date, "Days since first treatment"] = (vis_date - first_treatment).days \
            if first_treatment is not None else -1
        df.loc[vis_date, "Number of IVOMs"] = ivom_counter
        last_visit = vis_date
        if config.DatabaseKeys.drug in df.columns:
            visit_type = df.loc[vis_date, config.DatabaseKeys.drug]
            if not pd.isna(visit_type) and len(visit_type) > 0:
                if first_treatment is None:
                    first_treatment = vis_date
                    last_treatment = vis_date
                else:
                    last_treatment = vis_date
                ivom_counter += 1
    return df.sort_index(ascending=False)


def convert_drug_to_binary(x, drug):
    if isinstance(x, str):
        return x == drug
    elif isinstance(x, float) and np.isnan(x):
        return False
    else:
        try:
            for y in x:
                if y == drug:
                    return True
            return False
        except TypeError:
            return False


def split_drug_column_into_binary_features(df: pd.DataFrame):
    """ Splits the drug column into binary features. """
    df = df.copy()
    if config.DatabaseKeys.drug not in df.columns:
        df[config.DatabaseKeys.drug] = np.nan
    for drug in IvomDrugs:
        df[drug] = df[config.DatabaseKeys.drug].map(lambda x: convert_drug_to_binary(x, drug))
    return df


def transform_categorical_cols_to_bool(df):
    """ Transform the columns in the dataframe to categorical values. """
    for column in df.columns:
        if df[column].dtype in ["object", "category", "bool"]:
            if column == "Use case":
                df[column] = df[column].map(UseCases)
            elif column == config.DatabaseKeys.drug:
                df[column] = df[column].map(IvomDrugs)
            elif column in config.EyeStructures_to_index.keys():
                old_vals = df[column].tolist()
                new_vals = []
                for val in old_vals:
                    if len(val) > 0:
                        val = val.pop()
                        val = val[0]
                        if val in config.EyeStructures_to_index.keys():
                            new_vals.append(config.EyeStructures_to_index[val])
                        else:
                            new_vals.append(-1)
                    else:
                        new_vals.append(-1)
                df[column] = new_vals
            elif "Expected effect" not in column:
                df[column] = LabelEncoder().fit_transform(
                    df[column].astype("bool"))
    return df


def add_date_features_and_target_variables(df,
                                           target_key,
                                           target_months,
                                           only_date_features=False,
                                           interpolators=None):
    """ Add date features and target variables to the dataframe. Date features are the number of days since the first
        and last visit and treatment. Also the number of IVOMs is added. The target variable is the expected effect
        on the target key in target_months months. interpolators must be given to compute the target variable."""
    df = df.sort_index(ascending=True)  # Make sure the dataframe is sorted by date in ascending order.
    if not only_date_features:
        if "visual acuity" in target_key.lower():
            key = "Distance visual acuity"
        elif "intraocular" in target_key.lower():
            key = "Intraocular pressure finding"
        else:
            key = target_key
        interpolant = interpolators[key]
    else:
        key = target_key
    max_date = df.index[-1]
    last_visit = None
    first_visit = None
    last_treatment = None
    first_treatment = None
    ivom_counter = 0
    days_since_last_visit = []
    days_since_first_visit = []
    days_since_last_treatment = []
    days_since_first_treatment = []
    num_ivoms = []
    target_values = []
    stop_index = -1
    for i, vis_date in enumerate(df.index):
        if first_visit is None:
            first_visit = vis_date
        if not only_date_features:
            future_date = vis_date + pd.DateOffset(months=target_months)
            if future_date - max_date > pd.Timedelta(0):
                logger.warning(f"Future date {future_date} is larger than max date {max_date}. Current date {vis_date}"
                               f"Skipping this row.")
                stop_index = i
                break
            y = interpolant([future_date])
            target_values.append(y[0])
        if last_visit is None:
            days_since_first_visit.append(0)
            days_since_last_visit.append(-1)
        else:
            days_since_first_visit.append(abs((vis_date - first_visit).days))
            days_since_last_visit.append(abs((vis_date - last_visit).days))
        last_visit = vis_date
        visit_type = df.loc[vis_date, config.DatabaseKeys.drug]
        if visit_type != "Control":
            if first_treatment is None:
                days_since_first_treatment.append(0)
                days_since_last_treatment.append(0)
                first_treatment = vis_date
                last_treatment = vis_date
            else:
                days_since_first_treatment.append(abs((vis_date - first_treatment).days))
                days_since_last_treatment.append(abs((vis_date - last_treatment).days))
                last_treatment = vis_date
            ivom_counter += 1
        else:
            if first_treatment is None:
                days_since_first_treatment.append(-1)
                days_since_last_treatment.append(-1)
            else:
                days_since_first_treatment.append(abs((vis_date - first_treatment).days))
                days_since_last_treatment.append(abs((vis_date - last_treatment).days))
        num_ivoms.append(ivom_counter)
    if stop_index > 0:
        # stop_index is the index of the last valid value, we drop all rows after that.
        df = df.iloc[:stop_index]
    elif stop_index == 0:
        logger.error(f"Stop index is 0, meaning all values are invalid. This happens when the target time range is "
                     f"larger than the time range of the given data. Dropping all rows.")
        return pd.DataFrame()
    df["Days since last visit"] = days_since_last_visit
    df["Days since last treatment"] = days_since_last_treatment
    df["Days since first visit"] = days_since_first_visit
    df["Days since first treatment"] = days_since_first_treatment
    df["Number of IVOMs"] = num_ivoms
    if not only_date_features:
        if np.all(np.array(target_values == 0)):
            logger.error(f"All target values are 0 for target key {target_key} and target months {target_months}.")
        df[f"Expected effect on {target_key} in {target_months} months"] = target_values
        df.dtypes[f"Expected effect on {target_key} in {target_months} months"] = "float32"
    return df.sort_index(ascending=False)


def split_X_y_into_sequences(X, y, sequence_length):
    """ Split the X and y dataframes into sequences of length sequence_length. """
    X_list = []
    y_list = []
    for i in range(len(X) - sequence_length):
        X_list.append(X[i:i + sequence_length])
        y_list.append(y[i + sequence_length - 1])
    return X_list, y_list


def interpolate_df(df: pd.DataFrame, interpolator: Interpolator):
    """ Interpolate the dataframe using the given interpolator. """
    df = df.copy()
    for column in df.columns:
        if np.issubdtype(df[column].dtype, np.number):
            df[column] = interpolator(df[column])
        else:
            df[column] = df[column].interpolate(method='ffill', axis=0).interpolate(method='bfill', axis=0)
    return df
