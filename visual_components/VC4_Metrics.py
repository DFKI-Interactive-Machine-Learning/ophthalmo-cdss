# This file is part of Ophthalmo-CDSS.
# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# See https://creativecommons.org/licenses/by-nc-sa/4.0/ for details.

import streamlit as st
from visual_components.util import *


def Metrics():
    """Shows the metrics of the patient. This function is cached, such that the metrics are not recomputed on every
        rerun of the script."""
    col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
    sequence = get_session_state("joined_df")
    side = get_session_state("side")
    with col1:
        try:
            visus_current = sequence[f'{side} eye - Distance visual acuity'].dropna().iloc[0]
            visus_before = sequence[f'{side} eye - Distance visual acuity'].dropna().iloc[1]
            st.metric("Visual Acuity", f"{visus_current:.2f}", f"{visus_current - visus_before:.2f}")
        except Exception:
            st.error("No visual acuity data available.")
    with col2:
        try:
            volume_after = sequence['v_fluids'].dropna().iloc[0] / 1_000_000
            volume_before = sequence['v_fluids'].dropna().iloc[1] / 1_000_000
            volume_after_rounded = round(volume_after, 2)
            unit = "ml"
            if volume_after_rounded == 0.:
                volume_after *= 1_000
                volume_before *= 1_000
                unit = "μl"
            st.metric("Vol. Fluid", f"{volume_after:.2f} {unit}", f"{volume_after - volume_before:.2f} {unit}",
                      delta_color="inverse")
        except Exception:
            st.error("No fluid data available.")
    with col3:
        try:
            volume_after = sequence['v_ped'].dropna().iloc[0] / 1_000_000
            volume_before = sequence['v_ped'].dropna().iloc[1] / 1_000_000
            volume_after_rounded = round(volume_after, 2)
            unit = "ml"
            if volume_after_rounded == 0.:
                volume_after *= 1_000
                volume_before *= 1_000
                unit = "μl"
            st.metric("Vol. PED", f"{volume_after:.2f} {unit}", f"{volume_after - volume_before:.2f} {unit}",
                      delta_color="inverse")
        except Exception:
            st.error("No PED data available.")
    with col4:
        try:
            volume_after = sequence[f'{side} eye - Intraocular pressure finding'].dropna().iloc[0]
            volume_before = sequence[f'{side} eye - Intraocular pressure finding'].dropna().iloc[1]
            st.metric("IOP", f"{volume_after:.2f} mmHg", f"{volume_after - volume_before:.2f} mmHg",
                      delta_color="inverse")
        except Exception:
            st.error("No IOP data available.")
