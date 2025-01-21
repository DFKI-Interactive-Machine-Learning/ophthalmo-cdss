# This file is part of Ophthalmo-CDSS.
# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# See https://creativecommons.org/licenses/by-nc-sa/4.0/ for details.

import streamlit as st
import src.visualisation as viz
import config
from visual_components.util import *


def HistoryGraphs():
    visus, fluids, peds, iop = st.tabs(["Visual Acuity", "Fluids", "PEDs", "Intraocular pressure"])
    side = get_session_state("side")
    with visus:
        visus_fig = st.empty()
    with fluids:
        fluids_fig = st.empty()
    with peds:
        peds_fig = st.empty()
        draw_fig(peds_fig, "v_ped", "PEDs in ml", 1_000_000)
    with iop:
        iop_fig = st.empty()
        draw_fig(iop_fig, side + " eye - Intraocular pressure finding",
                 "Intraocular Pressure in mmHg", 1)
    if get_session_state("show_in_charts"):
        preds_visus = get_session_state("preds_visus")
        preds_volume = get_session_state("preds_fluids")
        with visus_fig:
            draw_fig(visus_fig, side + " eye - " + config.DatabaseKeys.visual_acuity, "Visual acuity in decimal",
                     1., prognosis_df=preds_visus)
        with fluids_fig:
            draw_fig(fluids_fig, "v_fluids", "Fluid Volume in μl", 1_000, prognosis_df=preds_volume)
    else:
        draw_fig(visus_fig, side + " eye - " + config.DatabaseKeys.visual_acuity,
                 "Visual Acuity in decimal", 1)
        draw_fig(fluids_fig, "v_fluids", "Fluid Volume in μl", 1_000)
    return visus_fig, fluids_fig


def draw_fig(fig, metric, title, conversion_factor, prognosis_df=None):
    try:
        ivom_timeline = get_session_state("ivom_timeline")
        joined_df = get_session_state("joined_df")
        selected_drug = get_session_state("select_treatment", ["Control"])
        with fig:
            st.plotly_chart(
                viz.get_line_chart_for_metric(joined_df,
                                              ivom_timeline,
                                              metric,
                                              y_title=title,
                                              prognosis_df=prognosis_df,
                                              selected_drug=selected_drug,
                                              current_date=get_session_state("date_picker"),
                                              main_date=get_session_state("main_date"),
                                              compare_date=get_session_state("compare_date", None),
                                              conversion_factor=conversion_factor),
                use_container_width=True)
    except Exception as e:
        with fig:
            st.error(f"No data for {title} available.")
