# This file is part of Ophthalmo-CDSS.
# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# See https://creativecommons.org/licenses/by-nc-sa/4.0/ for details.

import config
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler



def scale_array(x):
    """Scales the input array using a standard scaler."""
    scaler = StandardScaler()
    return scaler.fit_transform(x)


def should_abort(sequence_input):
    """Returns whether the treatment should be aborted. If the IOP is below 10 or above 21 mmHg, the visual acuity is
        below 0.025 or the last treatment was less than 10 days ago, the function returns True. Otherwise, it returns
        False."""
    current_iop = sequence_input.iloc[0, 11]
    current_visus = sequence_input.iloc[0, 10]
    days_since_last = sequence_input["Days since last treatment"].iloc[0]
    reasons = []
    if current_iop < 10.:
        reasons.append(f"IOP below 10 mmHg.")
    if current_iop > 21.:
        reasons.append(f"IOP above 21 mmHg.")
    if current_visus < 0.025:
        reasons.append(f"Visus below 0.025.")
    if 0 <= days_since_last < 10:
        reasons.append(f"Less than 10 days since last treatment.")
    return (
            current_iop < 10. or
            current_iop > 21. or
            current_visus < 0.025 or
            0 <= days_since_last < 10
    ), reasons


def should_treat(sequence_input):
    """Returns whether a treatment should be given. If fluids are detected in the last three visits, exudates are
        annotated or cysts are annotated, the function returns True. Otherwise, it returns False."""
    recent_fluids = np.any(sequence_input["v_fluids"].iloc[0:3].values > 0.0)
    exudates = sequence_input["exudate"].iloc[0]
    cysts = sequence_input["cyst"].iloc[0]
    reasons = []
    if recent_fluids:
        reasons.append("Fluids in last three visits detected.")
    if exudates:
        reasons.append("Exudates annotated.")
    if cysts:
        reasons.append("Cysts annotated.")

    return (
            recent_fluids or
            # current_ped > 100 or
            # retinal_thickness > 300 or  # Assuming 300 microns as an abnormal thickness threshold
            # retinal_change > 150 or  # Assuming a retinal thickness change threshold
            # visual_acuity < 1.0 or  # Assuming a visual acuity threshold
            # visual_acuity_change > 0.25 or  # Assuming a visual acuity change threshold
            # neovascularization or
            exudates or
            # edema or
            cysts
    ), reasons


def could_treat(preds_fluids):
    """Returns whether a treatment could be beneficial. If the maximum predicted fluid volume is above 5.0, the
        function returns True. Otherwise, it returns False."""
    return preds_fluids[["1 months", "3 months"]].loc["Control"].max() > 5.0, ["No fluids detected, but forecasted."]


def get_weighted_mean(predictions, weights):
    return [np.dot(predictions.iloc[i, :len(weights)], weights) for i in range(len(predictions))]


def determine_best_treatment(preds_visus, preds_fluids):
    """Returns treatment for which best forecasting is expected. If Control is the best treatment, the second best
        treatment is returned."""
    weights = [0.4, 0.4, 0.1, 0.05, 0.05]
    preds_visus["weighted_mean"] = get_weighted_mean(preds_visus, weights)
    preds_fluids["weighted_mean"] = get_weighted_mean(preds_fluids, weights)
    best_visus_fluids = np.mean([-1 * preds_visus["weighted_mean"].values,
                                 preds_fluids["weighted_mean"].values], axis=0)
    sorted_mean = np.argsort(best_visus_fluids)
    best_treatment = preds_visus.index[sorted_mean[0]]
    if best_treatment == "Control":
        return preds_visus.index[sorted_mean[1]]
    else:
        return best_treatment


@st.cache_data
def recommendation_model(sequence_input, preds_visus, preds_fluids, current_date, last_vol_date):
    """Shows the recommendation for the patient. This function is cached, such that the recommendation is not recomputed
        on every rerun of the script."""
    in_series, series_ivom, used_ivoms = check_series(sequence_input)
    abort, abort_reasons = should_abort(sequence_input)

    if abort:
        return "Abort", "Control", abort_reasons  # Abort conditions

    if in_series:
        series_ivom_str = list(config.IvomDrugs.keys())[int(series_ivom)]
        return "Should", series_ivom_str, [f"{used_ivoms}/3 IVOMs of {series_ivom_str}"]   # In series conditions

    if current_date - last_vol_date > pd.Timedelta(days=30):
        return "OCT", "Control", [f"Last OCT was taken {(current_date - last_vol_date).days} days ago."]

    should, should_reasons = should_treat(sequence_input)
    could, could_reasons = could_treat(preds_fluids)
    if should or could:
        best_treatment = determine_best_treatment(preds_visus, preds_fluids)

        try:
            used_ivoms = st.session_state["ivom_timeline_data"][config.DatabaseKeys.drug].unique()
        except Exception:
            # If the session state is not available, we assume that no IVOMs have been used.
            used_ivoms = []
        if used_ivoms:
            best_already_used = preds_visus.loc[used_ivoms]["weighted_mean"].idxmin()
            already_used_mean = preds_visus.loc[best_already_used]["weighted_mean"]
            best_mean = preds_visus.loc[best_treatment]["weighted_mean"]
            diff = best_mean - already_used_mean
        else:
            diff = 0.0
            best_already_used = best_treatment
        if diff <= 0.1:
            selected_treatment = best_already_used
        else:
            selected_treatment = best_treatment
        if should:
            return "Should", selected_treatment, should_reasons # Should conditions
        elif (could and
              preds_fluids.loc[selected_treatment]["weighted_mean"] < preds_fluids.loc["Control"]["weighted_mean"]):
            return "Could", selected_treatment, could_reasons + [f"Less fluids expected for {selected_treatment}"]
        else:
            return "No indication", "Control", ["No fluids detected."]   # No indication for treatment
    else:
        return "No indication", "Control", ["No fluids detected."]  # No indication for treatment


def check_series(model_input):
    """Checks if a series of IVOMs has been given. Returns a boolean indicating whether a series has been given, the
        IVOM that is currently being given and the number of IVOMs given in the series."""
    current_time_since_IVOM = model_input["Days since last treatment"].iloc[0]
    current_IVOM = model_input[config.DatabaseKeys.drug].iloc[0]
    current_date = model_input.index[0]
    actual_ivoms = model_input.where(model_input[config.DatabaseKeys.drug] != 0).dropna().sort_index(ascending=False)
    if actual_ivoms.empty:
        # No treatment given in the last 12 visits
        return False, 0, 0
    actual_ivoms.loc[current_date, config.DatabaseKeys.drug] = current_IVOM
    actual_ivoms.loc[current_date, "Days since last treatment"] = current_time_since_IVOM
    actual_ivoms = actual_ivoms[[config.DatabaseKeys.drug, "Days since last treatment"]].sort_index(ascending=False)
    ivoms = actual_ivoms[config.DatabaseKeys.drug].values
    days_since_last_IVOM = actual_ivoms["Days since last treatment"].values
    if np.all(ivoms[1:4] != 0.0) and np.all(days_since_last_IVOM[0:3] < 45) and len(np.unique(ivoms[1:4])) == 1:
        # Full series given
        return False, 0, 0
    elif np.all(ivoms[1:3] != 0.0) and np.all(days_since_last_IVOM[0:2] < 45) and len(np.unique(ivoms[1:3])) == 1:
        # 2 of 3 given
        return True, ivoms[1], 2
    elif ivoms[1] != 0.0 and days_since_last_IVOM[0] < 45:
        # 1 of 3 given
        return True, ivoms[1], 1
    else:
        return False, 0, 0
