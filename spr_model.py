import pandas as pd
import numpy as np


# ============================================================
# LOAD SPR DATASET
# ============================================================

SPR_FILE = "Au_BP_WS2_SPR_dataset_ML_ready_RI_1.33_to_1.42.csv"

spr_df = pd.read_csv(SPR_FILE)


# ============================================================
# EXTRACT SPR CURVE DATA
# ============================================================

reflectance_cols = [
    col for col in spr_df.columns
    if col.startswith("R_")
]

RI_values = spr_df["RI"].values.astype(float)

SPR_curves = spr_df[
    reflectance_cols
].values.astype(float)

angles = np.array([
    float(
        col.replace("R_", "")
           .replace("_deg", "")
    )
    for col in reflectance_cols
])


# ============================================================
# GENERATE SPR CURVE
# ============================================================

def generate_spr_curve(ri):

    ri = float(ri)

    # Interpolate every SPR curve point
    predicted_curve = np.array([
        np.interp(
            ri,
            RI_values,
            SPR_curves[:, i]
        )
        for i in range(SPR_curves.shape[1])
    ])

    # Find minimum of predicted curve
    min_index = np.argmin(
        predicted_curve
    )

    curve_min_angle = float(
        angles[min_index]
    )

    curve_rmin = float(
        predicted_curve[min_index]
    )

    # Interpolate the experimentally/
    # COMSOL-derived resonance angle
    valid = spr_df[
        spr_df["SPR_Angle_deg"].notna()
    ]

    predicted_angle = float(
        np.interp(
            ri,
            valid["RI"].values.astype(float),
            valid["SPR_Angle_deg"].values.astype(float)
        )
    )

    return (
        predicted_curve,
        predicted_angle,
        curve_min_angle,
        curve_rmin
    )


# ============================================================
# FINAL USER SPR PREDICTION FUNCTION
# ============================================================

def predict_user_spr(ri):

    try:
        ri = float(ri)

    except (ValueError, TypeError):

        return {
            "success": False,
            "message": (
                "Invalid RI. Please enter "
                "a numerical value."
            )
        }

    # --------------------------------------------------------
    # VALIDATED RANGE
    # --------------------------------------------------------

    if ri < 1.33 or ri > 1.40:

        return {
            "success": False,
            "ri": ri,
            "status": "Outside validated range",
            "message": (
                f"RI = {ri:.4f} is outside the "
                "validated SPR range of 1.33–1.40."
            ),
            "angles": None,
            "curve": None,
            "spr_angle": None,
            "curve_min_angle": None,
            "rmin": None
        }

    # --------------------------------------------------------
    # PREDICT FULL SPR CURVE
    # --------------------------------------------------------

    (
        curve,
        predicted_angle,
        curve_min_angle,
        curve_rmin
    ) = generate_spr_curve(ri)

    # --------------------------------------------------------
    # RETURN RESULT
    # --------------------------------------------------------

    return {
        "success": True,
        "ri": ri,
        "status": "Within validated range",
        "message": (
            "SPR prediction generated successfully."
        ),
        "angles": angles,
        "curve": curve,
        "spr_angle": predicted_angle,
        "curve_min_angle": curve_min_angle,
        "rmin": curve_rmin
    }


# ============================================================
# HEALTHY REFERENCE
# ============================================================

def get_healthy_reference():

    # Healthy reference selected from
    # the lower-RI region used in our project
    healthy_ri = 1.35

    return predict_user_spr(
        healthy_ri
    )
