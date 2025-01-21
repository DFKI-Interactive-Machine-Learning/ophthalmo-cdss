# This file is part of Ophthalmo-CDSS.
# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# See https://creativecommons.org/licenses/by-nc-sa/4.0/ for details.

from visual_components.util import *
from src.util import add_img_mask
import config
from visual_components.util import download_vol_cached, check_credentials


def Tools():
    vol_info = get_session_state("vol_info")
    prep_date = st.selectbox("Select Main VOL",
                             options=vol_info.index,
                             index=get_query_param_index_safe(vol_info.index.tolist(), "vol"),
                             help="Select which VOL to display.",
                             key="main_date")
    reset_query()
    file_uri = vol_info[vol_info.index == prep_date]["file_uri"].values[0]
    vol = download_vol_cached(file_uri, 0 if check_credentials() else 1)
    update_session_state("main_vol", vol)
    if "slice_index" not in st.session_state:
        st.session_state["slice_index"] = vol.wholefile["header"]["numBscan"] // 2
    show_type = st.radio("Show", options=["IRSLO", "VOL Slices", "3D"], horizontal=True, key="show_type")
    st.divider()
    compare_to = st.selectbox("Select Compare VOL",
                              options=vol_info.index,
                              index=1 if len(vol_info) > 1 else 0,
                              help="Select which VOL to compare to.",
                              key="compare_date")
    compare = st.toggle("Compare to other VOL",
                        help="Compare slices of different VOLs", disabled=len(vol_info) == 1 or show_type == "3D",
                        key="compare")
    compare_uri = vol_info[vol_info.index == compare_to]["file_uri"].values[0]
    vol_compare = download_vol_cached(compare_uri, 0 if check_credentials() else 2)
    update_session_state("compare_vol", vol_compare)
    alignment_possible = len(vol.oct) == len(vol_compare.oct) and compare
    st.toggle("Align VOLs",
                      disabled=not alignment_possible,
                      value=True,
                      help="Align the second VOL with the first for maximum overlap. Can lead to different scale "
                           "of VOLs.",
                      key="align")

    st.divider()
    segment = st.radio("Mode", options=["None", "Segment", "Thickness Map"], key="segment") if show_type == "IRSLO" else \
        st.toggle("Segment Slice", help="Segment the slice using YNet.",
                  disabled=show_type in ["3D"], key="segment")
    if show_type == "IRSLO" and segment == "Thickness Map":
        layers = ["Total retinal thickness"] + [
            config.LAYERS_TO_STRING[i] for i in vol.thickness_maps.keys()
        ]
        layer = st.selectbox("Select Layer", options=layers, index=0, key="layer")
        update_session_state("layer_index", layers.index(layer))
    mask_transparency = st.slider("Mask Transparency", value=0.2, min_value=0., max_value=1.0, step=0.1,
                                  disabled=not segment, key="mask_transparency")
    st.divider()
    irslo_small = st.empty()
    irslo_small.plotly_chart(viz.show_image(add_img_mask(vol.irslo, vol.get_IRSLO_segmentation(
        highlight_slice=st.session_state["slice_index"] - 1), mask_transparency), height=200), use_container_width=True,
                             )