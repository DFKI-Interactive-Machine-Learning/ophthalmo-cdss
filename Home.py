# This file is part of Ophthalmo-CDSS.
# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# See https://creativecommons.org/licenses/by-nc-sa/4.0/ for details.


from logging import getLogger
import logging
import config
import warnings
logging.basicConfig(level=config.LoggingConfig().level, format="%(levelname)-8s | %(name)-30s | %(message)s ")
logger = getLogger("Dashboard")
logger.setLevel(logging.INFO)
logger.info("Dashboard opened.")

# Must be imported after logging configuration
from visual_components import *
from visual_components.util import *



# Remove whitespace from the top of the page and sidebar
warnings.simplefilter(action="ignore")

# Sidebar
try:
    Sidebar()
except Exception as e:
    st.error("# Running the Ophthalmo-CDSS without a proper database "
             "following the scheme of `./data/demo.sqlite` will not work.")
    st.info("We will be working on a more general version of the CDSS in 2025. Please stay tuned! :)")


# Dashboard
# Removes padding from the top of the page and sides. Must be called after the sidebar or else sidebar will also have
# padding removed.
st.markdown("""
        <style>
               .block-container {
                    padding-top: 4rem;
                    padding-bottom: 0rem;
                    padding-left: 1rem;
                    padding-right: 2rem;
                }
        section[data-testid="stSidebar"] {
            width: 400px !important; # Set the width to your desired value
        }
        </style>
        """, unsafe_allow_html=True)

try:
    TopBar()
except Exception as e:
    st.info("This is where the top bar would be.")
st.divider()
tools, vol_view, data_col = st.columns([1, 4, 4])
with tools:
    try:
        Tools()
    except Exception as e:
        st.info("This is where the toolbar for the OCT viewer would be.")
with vol_view:
    try:
        VOLViewer()
    except Exception as e:
        st.info("This is where the OCTViewer would be.")
with data_col:
    try:
        visus_fig, preds_fig = HistoryGraphs()
    except Exception as e:
        pass
    try:
        Metrics()
    except Exception as e:
        st.info("This is where the Metrics would be.")
    st.divider()
    try:
        Recommendation()
    except Exception as e:
        st.info("This is where the Recommendation would be.")
    draw_fig(visus_fig,
             get_session_state("side") + " eye - Distance visual acuity", "Visual Acuity in decimal", 1,
             get_session_state("preds_visus") if get_session_state("show_in_charts") else None)
    draw_fig(preds_fig, "v_fluids", "Fluid Volume in μl", 1_000,
             get_session_state("preds_fluids") if get_session_state("show_in_charts") else None)
    try:
        Infobox()
    except Exception as e:
        st.info("This is where the Infobox would be.")

css = '''
<style>
section.main > div:has(~ footer ) {
    padding-bottom: 5px;
}
</style>
'''
st.markdown(css, unsafe_allow_html=True)
