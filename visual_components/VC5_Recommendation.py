# This file is part of Ophthalmo-CDSS.
# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# See https://creativecommons.org/licenses/by-nc-sa/4.0/ for details.

import config
import streamlit as st
from src.recommendation import recommendation_model
from visual_components.util import *


def Recommendation():
    rec, stuff = st.columns([3, 2])
    sequence_input = get_session_state("model_input").sort_index(ascending=False)
    treatment_type, drug, reasons = recommendation_model(sequence_input,
                                                         get_session_state("preds_visus"),
                                                         get_session_state("preds_fluids"),
                                                         get_session_state("date_picker"),
                                                         get_session_state("vol_info").index[0])
    with rec:
        if treatment_type == "Abort":
            st.error("### Abort treatment! \n Reasons:\n - " + "\n - ".join(reasons))
        elif treatment_type == "Should":
            st.success(f"### Treat with {drug}! \n Reasons:\n - " + "\n - ".join(reasons))
        elif treatment_type == "Could":
            st.warning(f"### Consider treating with {drug}! \n Reasons:\n - " + "\n - ".join(reasons))
        elif treatment_type == "OCT":
            st.warning("### Take new OCT! \n Reasons:\n - " + "\n - ".join(reasons))
        else:
            st.info("### No treatment needed! \n Reasons:\n - " + "\n - ".join(reasons))
        update_session_state("recommendation", drug)
        update_session_state("recommendation_reasons", reasons)
    with stuff:
        st.multiselect("Select treatment", options=config.IvomDrugs.keys(),
                       default=get_session_state("recommendation"),
                       key="select_treatment")
        st.toggle("Show forecasting in charts", value=True, key="show_in_charts")
