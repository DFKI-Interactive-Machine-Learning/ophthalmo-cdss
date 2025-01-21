# This file is part of Ophthalmo-CDSS.
# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# See https://creativecommons.org/licenses/by-nc-sa/4.0/ for details.

import streamlit as st
# Streamlit configs
st.set_page_config(page_title="Ophthalmo-CDSS",
                   layout="wide")
from visual_components.VC0_Sidebar import Sidebar
from visual_components.VC1_Topbar import TopBar
from visual_components.VC21_Tools import Tools
from visual_components.VC2_VOLViewer import VOLViewer
from visual_components.VC3_HistoryGraphs import HistoryGraphs, draw_fig
from visual_components.VC4_Metrics import Metrics
from visual_components.VC5_Recommendation import Recommendation
from visual_components.VC6_Infobox import Infobox
