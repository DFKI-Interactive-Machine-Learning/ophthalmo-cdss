# This file is part of Ophthalmo-CDSS.
# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# See https://creativecommons.org/licenses/by-nc-sa/4.0/ for details.

import config
from visual_components.util import *
from src.forecasting import get_predictions
from src.data.interpolation import SmoothedInterpolator
from src.data.preprocess import preprocess_visit_details_of_patient_for_model, transform_categorical_cols_to_bool
from src.forecasting.models import MultipleGenericModel
from visual_components.util import get_patients_cached, get_visit_data_cached, get_medicamentation_per_side_cached, \
    get_vol_info_cached

logger = getLogger(__name__)


def Sidebar():
    institute = st.sidebar.radio("Institute",
                                 options=["All", "AKS", "AZM"],
                                 horizontal=True,
                                 index=get_query_param_index_safe(["All", "AKS", "AZM"], "institute"),
                                 disabled=True,
                                 key="institute")
    usecase = st.sidebar.selectbox("Select use case",
                                   ["All"] + list(config.UseCases.keys()),
                                   index=get_query_param_index_safe(["All"] + list(config.UseCases.keys()), "usecase"),
                                   key="usecase")
    # patients_df = get_patients_cached(list(config.UseCases.keys()) if usecase == "All" else usecase,
    #                                   institute=institute)
    # patients_df["anonymous_id"] = patients_df["id"].apply(
    #     lambda x: f"Patient {patients_df.index[patients_df['id'] == x][0]}")
    patient_id = st.sidebar.selectbox(
        'Select patient',
        ["Demo patient"],
        # index=get_query_param_index_safe(patients_df['id'].tolist(), "patient_id"),
        key="patient_id")
    # patient_id = patients_df[patients_df["anonymous_id"] == patient_id]["id"].values[0]
    # patient = patients_df[patients_df["id"] == patient_id].reset_index()
    # update_session_state("patient", patient)
    side = st.sidebar.radio("Select Eye Side",
                            options=["Left", "Right"],
                            index=get_query_param_index_safe(["Left", "Right"], "side"),
                            horizontal=True,
                            label_visibility="visible",
                            key="side")

    visit_dates = ["2023-01-01", "2023-02-01", "2023-03-01", "2023-12-01", "2024-01-01", "2024-02-02", "2024-11-01",
                   "2024-12-01"]  # db.get_visit_dates_of_patient(patient_id, side=side)
    visit_dates = [pd.to_datetime(str(date)[:10]) for date in visit_dates]
    visit_dates = pd.DataFrame(visit_dates, columns=["date"]).drop_duplicates(keep="first")["date"].tolist()
    date_picker = st.sidebar.selectbox('Select visit date',
                                       visit_dates,
                                       index=get_query_param_index_safe(visit_dates, "date"),
                                       disabled=True,
                                       key="date_picker")
    logger.info(f"Selection:")
    logger.info(f" - Institute:  {institute}")
    logger.info(f" - Use case:   {usecase}")
    logger.info(f" - Patient ID: {patient_id}")
    logger.info(f" - Side:       {side}")
    logger.info(f" - Date:       {date_picker}")

    # Visit data
    visit_data_full_both_df = get_visit_data_cached(patient_id, "any")
    visit_data_full_both_df.index = pd.to_datetime(visit_data_full_both_df.index)
    visit_data_full_side_df = get_visit_data_cached(patient_id, side)
    visit_data_full_side_df.index = pd.to_datetime(visit_data_full_side_df.index)
    visit_data_side_df = visit_data_full_side_df[visit_data_full_side_df.index <= date_picker]
    visit_data_both_df = visit_data_full_both_df[visit_data_full_both_df.index <= date_picker]
    update_session_state("visit_data_both_df", visit_data_both_df)

    # VOL data
    vol_info = get_vol_info_cached(patient_id, "Left eye" if side == "Left" else "Right eye")
    vol_info.index = pd.to_datetime(vol_info.index)
    if vol_info.empty:
        st.error(f"# The patient {patient_id} has no OCT data for the selected side. Please select another patient or "
                 f"try another side.")
        raise RuntimeError(f"No OCT data for patient {patient_id} on side {side}.")
    vol_info = vol_info[pd.to_datetime(vol_info.index) <= date_picker]
    vol_info = vol_info.sort_values(by="slices", ascending=False)
    vol_info = vol_info[~vol_info.index.duplicated(keep='first')]
    vol_info = vol_info.sort_index(ascending=False)
    update_session_state("vol_info", vol_info)

    # Joined data frame
    joined_df = (pd.merge(visit_data_side_df, vol_info, how="left", left_index=True, right_index=True)
                 .sort_index(ascending=False))
    update_session_state("joined_df", joined_df)

    # IVOM timeline
    ivom_timeline = get_medicamentation_per_side_cached(patient_id, side)
    ivom_timeline.index = pd.to_datetime(ivom_timeline.index)
    ivom_timeline = ivom_timeline[ivom_timeline.index <= date_picker]
    update_session_state("ivom_timeline", ivom_timeline)

    # Preprocessed data for model input
    diagnoses = joined_df["Diagnosis - Use case"].unique()
    usecase_str = ' & '.join([d for d in diagnoses if not pd.isna(d)]) if len(diagnoses) > 0 else '--'
    preprocessed_df = preprocess_visit_details_of_patient_for_model(joined_df,
                                                                    side=side,
                                                                    default_usecase=usecase_str,
                                                                    birthday=patient["birthday"].values[0],
                                                                    gender=patient["gender"].values[0],
                                                                    without_targets=True,
                                                                    transform_to_categorical=False,
                                                                    interpolator=SmoothedInterpolator(
                                                                        moving_average_window=90))
    update_session_state("preprocessed_df", preprocessed_df)
    model_input = transform_categorical_cols_to_bool(preprocessed_df.iloc[:12])
    update_session_state("model_input", model_input)
    model_input = model_input[[
        "Use case", config.DatabaseKeys.drug, "Age", "Gender", "Finding of tobacco smoking behavior",
        "v_fluids", "n_fluids", "v_ped", "n_ped",
         "total_thickness", "Distance visual acuity",
        "Intraocular pressure finding", "edema", "bleeding", "aneurysm", "inflammation",
        "scar", "ischemia", "atrophy", "cataract", "neovascularization",
        "exudate", "cyst", "tumor", "hypertrophy",
        "detachment", "drusen", "hyaline_bodies", "BMI",
        "Days since last visit", "Days since last treatment",
        "Days since first visit", "Days since first treatment",
        "Number of IVOMs", "astigmatism"
    ]]  # FIXME: Add new prediction models, the features are wrong but it works for now

    # Predictions of the forecasting models
    # Makes sure that the models are only loaded once and not on every rerun of the script.
    if "visual_model" not in st.session_state:
        add_to_session_state("visual_model", MultipleGenericModel("visual_acuity"))
        add_to_session_state("fluid_model", MultipleGenericModel("v_fluids"))
        add_to_session_state("n_fluids_model", MultipleGenericModel("n_fluids"))
    try:
        preds_fluids = get_predictions(get_session_state("fluid_model"), model_input)
        preds_visus = get_predictions(get_session_state("visual_model"), model_input)
        preds_n_fluids = get_predictions(get_session_state("n_fluids_model"), model_input)
    except Exception as e:
        logger.error(f"Cannot predict for patient {patient_id} at date {date_picker}.")
        logger.error(e)
        preds_fluids = None
        preds_visus = None
        preds_n_fluids = None
    update_session_state("preds_fluids", preds_fluids)
    update_session_state("preds_visus", preds_visus)
    update_session_state("preds_n_fluids", preds_n_fluids)

    # Clear cache button
    st.sidebar.button("Clear Cache", on_click=clear_cache)
