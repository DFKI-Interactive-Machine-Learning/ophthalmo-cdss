# This file is part of Ophthalmo-CDSS.
# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# See https://creativecommons.org/licenses/by-nc-sa/4.0/ for details.

import streamlit as st
import config
import numpy as np
from src.util import get_recommendation_str
from visual_components.util import *


def try_get_value_at_index(sequence, column, index, default=None):
    try:
        return sequence[column].dropna().iloc[index]
    except IndexError:
        return default


def Infobox():
    rec, diff_tab, thickness = st.tabs(["Reasoning", "Visit Diff", "Mean Thickness (µm)"])
    with rec:
        Reasoning()
    with diff_tab:
        VisitDiff()
    with thickness:
        MeanThickness()


def Reasoning():
    sequence = get_session_state("joined_df")
    side = get_session_state("side")
    preds_visus = get_session_state("preds_visus")
    preds_volume = get_session_state("preds_fluids")
    preds_n_fluids = get_session_state("preds_n_fluids")
    visus_after = try_get_value_at_index(sequence, f'{side} eye - Distance visual acuity', 0, default=0)
    visus_before = try_get_value_at_index(sequence, f'{side} eye - Distance visual acuity', 1, default=0)
    num_fluids_after = try_get_value_at_index(sequence, 'n_fluids', 0, default=0)
    num_fluids_before = try_get_value_at_index(sequence, 'n_fluids', 1, default=0)
    volume_after = try_get_value_at_index(sequence, "v_fluids", 0, default=0) / 1_000_000
    volume_before = try_get_value_at_index(sequence, "v_fluids", 1, default=0) / 1_000_000
    volume_after_rounded = round(volume_after, 2)
    unit = "mL"
    if volume_after_rounded == 0.:
        volume_after *= 1_000
        volume_before *= 1_000
        unit = "μL"
    indicators, prognosis = st.columns([1, 1])
    with indicators:
        st.markdown(f"Indicators for recommendation:")

        st.markdown(f"- {get_recommendation_str('Visus', visus_before, visus_after)}")

        st.markdown(
            f"- {get_recommendation_str('Number of fluids', num_fluids_before, num_fluids_after, higher_is_better=False)}")

        st.markdown(
            f"- {get_recommendation_str('Total volume of fluids', volume_before, volume_after, higher_is_better=False, unit=unit)}")

    with prognosis:
        st.markdown(f"Expected change in 3 months without treatment:")

        st.markdown(
            f"- {get_recommendation_str('Visus', visus_after, preds_visus.iloc[config.IvomDrugs['Control'], 1])}")

        st.markdown(
            f"- {get_recommendation_str('Number of fluids', num_fluids_after, np.ceil(preds_n_fluids.iloc[config.IvomDrugs['Control'], 1]), higher_is_better=False)}")

        st.markdown(
            f"- {get_recommendation_str('Total volume of fluids', volume_after, preds_volume.iloc[config.IvomDrugs['Control'], 1], higher_is_better=False, unit=unit)}")


def VisitDiff():
    df = get_session_state("preprocessed_df")
    df = df.sort_index(ascending=False).drop(columns=["Use case",
                                                      "Treated side",
                                                      "BMI",
                                                      "Age",
                                                      "Gender",
                                                      config.DatabaseKeys.drug,
                                                      "Finding of tobacco smoking behavior",
                                                      "Days since last treatment",
                                                      "Days since first treatment",
                                                      "Days since first visit",
                                                      "Days since last visit",
                                                      "Distance visual acuity",
                                                      "Intraocular pressure finding",
                                                      "Number of IVOMs",
                                                      "total_thickness",
                                                      "v_fluids", "n_fluids",
                                                      "v_ped", "n_ped"], errors="ignore")
    latest_visit = df.iloc[0]
    previous_visit = df.iloc[1]
    remove = []
    unchanged = []
    for column, dtype, late, prev in zip(df.columns, df.dtypes, latest_visit, previous_visit):
        if late == prev and ((type(late) is set and len(late) == 0) or late is None):
            remove.append(column)
        elif column in config.EyeSymptomes.keys():
            if late == prev:
                unchanged.append("unchanged")
            elif type(late) is set and type(prev) is set:
                if len(late) == 0 and len(prev) > 0:
                    unchanged.append("resolved")
                elif len(late) > 0 and len(prev) == 0:
                    unchanged.append("emerged")
                elif len(late) < len(prev):
                    unchanged.append("decreased")
                elif len(late) > len(prev):
                    unchanged.append("increased")
                else:
                    unchanged.append("changed")
            else:
                unchanged.append("changed")
        elif dtype == np.dtype("float64"):
            if late == prev:
                unchanged.append("unchanged")
            elif late > prev:
                unchanged.append("increased")
            else:
                unchanged.append("decreased")
        else:
            if late == prev and late is not None:
                unchanged.append("unchanged")
            elif late is None or (type(late) is set and len(late) == 0):
                remove.append(column)
            else:
                unchanged.append("changed")

    latest_visit = latest_visit.drop(remove)
    previous_visit = previous_visit.drop(remove)
    diff_df = pd.DataFrame({df.index[1]: prepare_for_visit_diff(previous_visit),
                            "Changes": unchanged,
                            df.index[0]: prepare_for_visit_diff(latest_visit)})
    diff_df.replace("set()", None).dropna(axis=0, inplace=False)

    # Define a function to apply different styles based on the value
    def color_change(val):
        if val == 'unchanged':
            color = 'rgba(255, 255, 255, 0)'
        elif val == 'changed':
            color = 'rgba(0, 0, 200, 0.2)'
        elif val == 'increased':
            color = 'rgba(255, 88, 0, 0.2)'
        elif val == 'decreased':
            color = 'rgba(151, 255, 0, 0.2)'
        elif val == 'emerged':
            color = 'rgba(255, 0, 0, 0.2)'
        elif val == 'resolved':
            color = 'rgba(0, 255, 0, 0.2)'
        else:
            color = 'rgba(255, 255, 255, 0)'  # Default color for other cases
        return f'background-color: {color}'

    styled_diff = diff_df.style.applymap(color_change)
    st.dataframe(styled_diff, use_container_width=True)


def MeanThickness():
    vol_compare = get_session_state("compare_vol")
    vol = get_session_state("main_vol")

    def color_change_num(val):
        val = float(val)
        if abs(val) < 0.5:
            color = 'rgba(255, 255, 255, 0)'
        elif val < -100:
            color = 'rgba(255, 0, 0, 0.2)'
        elif val <= -0.5:
            color = 'rgba(255, 88, 0, 0.2)'
        elif val > 100:
            color = 'rgba(0, 255, 0, 0.2)'
        elif val >= 0.5:
            color = 'rgba(151, 255, 0, 0.2)'
        else:
            color = 'rgba(255, 255, 255, 0)'
        return f'background-color: {color}'

    thickness_vals = {}
    total_before = 0.
    total_after = 0.
    for index in range(1, 8):
        thickness_mean_before = vol_compare.get_mean_thickness(index)
        total_before += thickness_mean_before
        thickness_mean_after = vol.get_mean_thickness(index)
        total_after += thickness_mean_after
        thickness_vals[config.LAYERS_TO_STRING[index]] = [
            f"{thickness_mean_before:.1f}",
            f"{thickness_mean_after - thickness_mean_before:.1f}",
            f"{thickness_mean_after:.1f}"
        ]
    thickness_vals["Total"] = [f"{total_before:.1f}",
                               f"{total_after - total_before:.1f}",
                               f"{total_after:.1f}"]
    thickness_df = pd.DataFrame(thickness_vals,
                                index=[get_session_state('compare_date'), "Change", get_session_state('main_date')])
    if get_session_state("compare_date") == get_session_state("main_date"):
        st.warning("You have selected the same OCT for main and comparison. Please select different OCTs to see the "
                   "difference.")
        styled_diff = thickness_df.iloc[0].to_frame()
    else:
        styled_diff = thickness_df.T.style.applymap(color_change_num, subset=["Change"])
    st.dataframe(styled_diff, use_container_width=True)
