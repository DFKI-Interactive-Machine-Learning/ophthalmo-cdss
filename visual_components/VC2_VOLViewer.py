# This file is part of Ophthalmo-CDSS.
# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# See https://creativecommons.org/licenses/by-nc-sa/4.0/ for details.

import config
from src.util import add_img_mask
from src.alignment import compute_matches, compute_transformation, align_slices
import plotly.graph_objects as go
import cv2 as cv
from streamlit_image_comparison import image_comparison
from visual_components.util import *
import time


@st.cache_data
def getTransformationMatrixForVOLs(_vol1, _vol2, vol1_date, vol2_date, n_matches):
    """ Computes the transformation matrix for the two best matching slices of two VOLs. The transformation matrix is
        used to align the slices of the second VOL with the first VOL. The transformation matrix is computed by
        comparing the two VOLs slice by slice and finding the best matching slices. The transformation matrix is then
        computed by aligning the best matching slices. This transformation matrix can then be used to align any
        two slices of the two VOLs. """
    mean_matches = []
    for i in range(len(_vol1.oct)):
        matches, _, _ = compute_matches(
            _vol1.oct[i].astype(np.uint8),
            _vol2.oct[i].astype(np.uint8),
        )
        mean_matches.append(np.mean([x.distance for x in matches[:n_matches]]))
    best_match = np.argmax(mean_matches)
    return compute_transformation(_vol1.oct[best_match].astype(np.uint8),
                                  _vol2.oct[best_match].astype(np.uint8),
                                  n_matches)


def findIRSLOAlignment(current, compare):
    """Computes the alignment of two IRSLO."""
    transform_matrix = compute_transformation(current, compare, n_matches=500)
    if np.sum(transform_matrix[:, :2]) >= 1.75:
        # Alignment Sanity Check: If the scaling coefficients get too small, this leads to very bad
        # alignments, where the second VOL is just super small. In this case do not align at all.
        return align_slices(current, compare, transform_matrix)
    else:
        return compare


def CompareSlider(current, compare, current_date, compare_date):
    cv.imwrite(os.path.join(config.TEMP_DATA, st.connection.__name__ + "current.png"),
               current)
    cv.imwrite(os.path.join(config.TEMP_DATA, st.connection.__name__ + "compare.png"),
               compare)
    image_comparison(os.path.join(config.TEMP_DATA, st.connection.__name__ + "current.png"),
                     os.path.join(config.TEMP_DATA, st.connection.__name__ + "compare.png"),
                     label1=current_date,
                     label2=compare_date,
                     width=700)


def IRSLO():
    """Shows the IRSLO image with the selected options. This function is cached, such that the IRSLO image is not
        reloaded on every rerun of the script."""
    vol = get_session_state("main_vol")
    main_date = get_session_state("main_date")
    vol_compare = get_session_state("compare_vol")
    compare_date = get_session_state("compare_date")
    segment = get_session_state("segment", "None")
    compare = get_session_state("compare", False)
    align = get_session_state("align", False)
    mask_transparency = get_session_state("mask_transparency", 0.2)
    slice_index = get_session_state("slice_index")
    layer_index = get_session_state("layer_index", 0)
    if segment == "Segment":
        current_irslo = add_img_mask(vol.irslo, vol.get_IRSLO_segmentation(highlight_slice=slice_index),
                                     mask_transparency)
    elif segment == "Thickness Map":
        fig = viz.show_image(vol.irslo)
        min_x, min_y = min(vol.grid[:][0]), min(vol.grid[:][1])
        vol_thickness = vol.get_thickness_map(layer_index)
        if not compare:
            z = pd.DataFrame(vol_thickness).interpolate(method="from_derivatives", axis=0,
                                                                             limit_area='inside').values
        else:
            vol_compare_thickness = vol_compare.get_thickness_map(layer_index)

            z_vol = pd.DataFrame(vol_thickness).interpolate(method="from_derivatives", axis=0,
                                                            limit_area='inside').values
            z_vol_compare = pd.DataFrame(vol_compare_thickness).interpolate(method="from_derivatives", axis=0,
                                                                            limit_area='inside').values
            z_vol_compare_resized = cv.resize(z_vol_compare,
                                              (vol_thickness.shape[1], vol_thickness.shape[0]))
            z = z_vol - z_vol_compare_resized

        fig.add_trace(go.Heatmap(z=z, x0=min_x, y0=min_y, zmid=0,
                                 showscale=True, connectgaps=True, zsmooth="best",
                                 opacity=mask_transparency,
                                 colorscale='IceFire' if compare else 'Viridis',
                                 colorbar=dict(title='Thickness [µm]'),
                                 hoverinfo='z',
                                 hovertemplate='Difference in Thickness: %{z:.2f}µm' if
                                 compare else 'Thickness: %{z:.2f}µm'))
        st.plotly_chart(fig, use_container_width=True)
        if compare:
            mean_vol = vol.get_mean_thickness(layer_index)
            mean_vol_compare = vol_compare.get_mean_thickness(layer_index)
            difference = mean_vol - mean_vol_compare
            if difference > 0:
                st.success(f"Mean {config.LAYERS_TO_STRING[layer_index]} Thickness increased "
                           f"from {mean_vol_compare:.2f} µm to {mean_vol:.2f} µm.")
            elif difference < 0:
                st.warning(f"Mean {config.LAYERS_TO_STRING[layer_index]} Thickness decreased "
                           f"from {mean_vol_compare:.2f} µm to {mean_vol:.2f} µm.")
            else:
                st.info(f"Mean {config.LAYERS_TO_STRING[layer_index]} Thickness did not change. "
                        f"Currently at {mean_vol:.2f} µm.")
        else:
            st.info(f"Mean {config.LAYERS_TO_STRING[layer_index]} "
                    f"Thickness: {vol.get_mean_thickness(layer_index):.2f} µm")
        return
    else:
        current_irslo = vol.irslo
    if compare:
        if segment == "Segment":
            compare_irslo = add_img_mask(vol_compare.irslo,
                                         vol_compare.get_IRSLO_segmentation(highlight_slice=slice_index),
                                         mask_transparency)
        else:
            compare_irslo = vol_compare.irslo
        CompareSlider(current_irslo,
                      findIRSLOAlignment(current_irslo, compare_irslo) if align else compare_irslo,
                      main_date,
                      compare_date)
    else:
        st.plotly_chart(viz.show_image(current_irslo),
                        use_container_width=True)


def Slices():
    """Shows the OCT slices of the VOL. This function is cached, such that the slices are not reloaded on every rerun of
        the script."""
    vol = get_session_state("main_vol")
    main_date = get_session_state("main_date")
    vol_compare = get_session_state("compare_vol")
    compare_date = get_session_state("compare_date")
    segment = get_session_state("segment", "None")
    compare = get_session_state("compare", False)
    align = get_session_state("align", False)
    mask_transparency = get_session_state("mask_transparency", 0.2)
    slice_index = get_session_state("slice_index")
    og_current_slice = (vol.oct[slice_index - 1]).astype(np.uint8)  # always keep original slice in memory
    if segment:
        current_slice = add_img_mask(og_current_slice, vol.get_as_rgb_mask(slice_index - 1), mask_transparency)
    else:
        current_slice = og_current_slice
    if compare:
        if len(vol.oct) == len(vol_compare.oct):
            transformation_matrix = getTransformationMatrixForVOLs(vol, vol_compare,
                                                                   main_date, compare_date,
                                                                   100)

        current_slice_pil = np.flip(current_slice, axis=[0, 1])
        compare_slice = (vol_compare.oct[slice_index - 1]).astype(np.uint8)

        if segment:
            seg_mask = vol_compare.get_as_rgb_mask(slice_index - 1)
            compare_slice = cv.cvtColor(compare_slice, cv.COLOR_GRAY2RGB)
            if mask_transparency < 1.0:
                compare_slice = cv.addWeighted(compare_slice, 1.0, seg_mask, mask_transparency, 0.0)
            else:
                compare_slice = seg_mask

        if align:
            compare_slice = align_slices(og_current_slice, compare_slice, transformation_matrix)
        compare_slice_pil = np.flip(compare_slice, axis=[0, 1])
        CompareSlider(current_slice_pil, compare_slice_pil, main_date, compare_date)
        if align and compare:
            if not vol.is_followup(vol_compare.fileHeader["ReferenceID"]) \
                    or vol_compare.is_followup(vol.fileHeader["ReferenceID"]):
                st.warning("VOLs are not Follow-Up. Attempting alignment!")
            st.warning("Alignment can rescale Slices and distort ratios!")
    else:
        st.plotly_chart(viz.show_image(current_slice, vol.masks[slice_index - 1], segment),
                        use_container_width=True)


def Graph3D():
    """Shows the 3D plot of the VOL. This function is cached, such that the 3D plot is not reloaded on every rerun of
        the script."""
    vol = get_session_state("main_vol")
    main_date = get_session_state("main_date")
    slice_index = get_session_state("slice_index")
    fig = get_3D_plot(main_date, vol)
    st.plotly_chart(viz.add_image_to_fig3D(fig, vol.oct[slice_index - 1], vol.grid[slice_index - 1][1]))
    st.caption("You can move the slice in the plot by using the slider above. Layers and Lesions can be switched on "
               "and off using the legend above.")


def SliderAndButtons():
    buttons = st.columns([1, 5, 1])
    num_scans = get_session_state("main_vol").wholefile["header"]["numBscan"]
    with buttons[1]:
        slice_index = st.slider("Slice: ", min_value=1, max_value=num_scans,
                                step=1,
                                label_visibility="collapsed",
                                key="slice_index")
    with buttons[0]:
        st.button("Prev", on_click=lambda: st.session_state.update(slice_index=slice_index - 1),
                  use_container_width=True, disabled=slice_index <= 1)
    with buttons[2]:
        st.button("Next", on_click=lambda: st.session_state.update(slice_index=slice_index + 1),
                  use_container_width=True, disabled=slice_index == num_scans)
        return slice_index


@st.dialog("Feedback")
def Feedback():
    st.write("What did you have an issue with?")
    st.checkbox("Segmentation is incorrect.")
    st.checkbox("Recommendation is incorrect.")
    st.checkbox("Other (Please describe below).")
    st.write("Please describe the issue.")
    st.text_area("")
    if st.button("Submit"):
        with st.status("Submitting..."):
            st.write("Submitting feedback.")
            time.sleep(1)
            st.write("Transferring OCT to database.")
            time.sleep(3)
            st.write("Feedback submitted!")
        st.success("Thank you for your feedback.")




def VOLViewer():
    SliderAndButtons()
    show_type = get_session_state("show_type", "IRSLO")
    if show_type == "IRSLO":
        IRSLO()
    elif show_type == "3D":
        Graph3D()
    else:
        Slices()
    if st.button("Report Issue", help="Report an issue with the VOL Viewer."):
        Feedback()
