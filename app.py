import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageStat

from spr_model import predict_user_spr, get_healthy_reference


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Skin Cancer Multimodal Analysis",
    page_icon="🔬",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 19px;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 26px;
        font-weight: 600;
        margin-top: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🔬 Skin Cancer Multimodal Analysis'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Deep Learning + Surface Plasmon Resonance'
    '</div>',
    unsafe_allow_html=True
)

st.info(
    "Research prototype combining skin-image classification "
    "with refractive-index-based SPR analysis."
)


# ============================================================
# LOAD CNN MODEL
# ============================================================

@st.cache_resource
def load_cnn_model():

    model = tf.keras.models.load_model(
        "skin_cancer_cnn_91_75.keras"
    )

    return model


try:

    cnn_model = load_cnn_model()

except Exception as e:

    st.error(
        "Unable to load the trained CNN model."
    )

    st.code(str(e))

    st.stop()


# ============================================================
# IMAGE QUALITY CHECK
# ============================================================

def check_image_quality(image):

    image = image.convert("RGB")

    width, height = image.size

    if width < 100 or height < 100:

        return False, "Image resolution is too low."

    stat = ImageStat.Stat(image)

    brightness = sum(stat.mean) / 3

    if brightness < 15:

        return False, "Image is too dark."

    if brightness > 245:

        return False, "Image is too bright."

    return True, "Image quality check passed."


# ============================================================
# MULTIMODAL FUSION
# ============================================================

def calculate_multimodal_score(
    cnn_malignant_probability,
    user_ri,
    healthy_ri=1.35
):
    """
    Research-prototype multimodal fusion.

    CNN = primary image-based evidence.
    RI = optical contribution from SPR pathway.

    The RI contribution is centered around the healthy
    reference RI.

    RI below healthy reference:
        decreases the multimodal score.

    RI above healthy reference:
        increases the multimodal score.

    IMPORTANT:
    This is a research-prototype score and NOT a clinically
    validated cancer probability.
    """

    # --------------------------------------------------------
    # RI NORMALIZATION
    # --------------------------------------------------------

    validated_min_ri = 1.33
    validated_max_ri = 1.40

    # Normalize RI from 0 to 1
    ri_normalized = (
        (user_ri - validated_min_ri)
        /
        (validated_max_ri - validated_min_ri)
    )

    ri_normalized = float(
        np.clip(
            ri_normalized,
            0.0,
            1.0
        )
    )

    # --------------------------------------------------------
    # CENTER RI CONTRIBUTION AROUND HEALTHY RI
    # --------------------------------------------------------

    healthy_normalized = (
        (healthy_ri - validated_min_ri)
        /
        (validated_max_ri - validated_min_ri)
    )

    # Difference from healthy reference
    ri_difference = (
        ri_normalized
        -
        healthy_normalized
    )

    # --------------------------------------------------------
    # RI EFFECT
    #
    # Maximum RI effect = ±15 percentage points.
    #
    # This keeps the CNN as the primary contributor while
    # allowing RI to increase/decrease the final score.
    # --------------------------------------------------------

    ri_effect = 0.30 * ri_difference

    # --------------------------------------------------------
    # FINAL MULTIMODAL SCORE
    # --------------------------------------------------------

    multimodal_score = (
        cnn_malignant_probability
        +
        ri_effect
    )

    multimodal_score = float(
        np.clip(
            multimodal_score,
            0.0,
            1.0
        )
    )

    return (
        multimodal_score,
        ri_normalized,
        ri_effect
    )


# ============================================================
# INPUT SECTION
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Patient Inputs'
    '</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)


# ============================================================
# SKIN IMAGE INPUT
# ============================================================

with col1:

    st.subheader("🖼️ Skin Image")

    uploaded_image = st.file_uploader(
        "Upload a skin lesion image",
        type=["jpg", "jpeg", "png"],
        help="Upload a JPG, JPEG, or PNG skin image."
    )

    image = None

    if uploaded_image is not None:

        try:

            image = Image.open(
                uploaded_image
            ).convert("RGB")

            st.image(
                image,
                caption="Uploaded Skin Image",
                use_container_width=True
            )

            quality_ok, quality_message = (
                check_image_quality(image)
            )

            if quality_ok:

                st.success(
                    quality_message
                )

            else:

                st.warning(
                    quality_message
                )

        except Exception as e:

            st.error(
                "Unable to read the uploaded image."
            )

            st.code(str(e))


# ============================================================
# RI INPUT
# ============================================================

with col2:

    st.subheader("🔬 Refractive Index")

    user_ri = st.number_input(
        "Enter measured refractive index (RI)",
        min_value=1.30,
        max_value=1.45,
        value=1.375,
        step=0.001,
        format="%.3f"
    )

    st.caption(
        "Validated SPR range: RI = 1.33–1.42"
    )

    st.caption(
        "Healthy reference: RI = 1.33-1.35"
    )


# ============================================================
# ANALYZE BUTTON
# ============================================================

st.markdown("")

analyze = st.button(
    "🔍 ANALYZE",
    type="primary",
    use_container_width=True
)


# ============================================================
# MAIN ANALYSIS
# ============================================================

if analyze:

    # ========================================================
    # CHECK IMAGE
    # ========================================================

    if image is None:

        st.warning(
            "Please upload a skin image before analysis."
        )

        st.stop()


    # ========================================================
    # CNN ANALYSIS
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '🧬 CNN Skin Image Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    with st.spinner(
        "Analyzing skin image..."
    ):

        try:

            # ------------------------------------------------
            # CONVERT IMAGE TO NUMPY
            # ------------------------------------------------

            img_array = np.array(
                image,
                dtype=np.float32
            )

            # ------------------------------------------------
            # RESIZE TO CNN INPUT SIZE
            # ------------------------------------------------

            img_resized = tf.image.resize(
                img_array,
                (224, 224)
            )

            # ------------------------------------------------
            # ADD BATCH DIMENSION
            # ------------------------------------------------

            img_input = tf.expand_dims(
                img_resized,
                axis=0
            )

            # ------------------------------------------------
            # IMPORTANT
            #
            # DO NOT USE /255 HERE.
            #
            # The trained model already contains its
            # preprocessing layers.
            # ------------------------------------------------

           prediction = cnn_model.predict(
    img_input,
    verbose=0
)

raw_prediction = float(
    np.asarray(prediction).reshape(-1)[0]
)

st.write("DEBUG — Raw CNN output:", raw_prediction)

prediction = cnn_model.predict(
    img_input,
    verbose=0
)

raw_prediction = float(
    np.asarray(prediction).reshape(-1)[0]
)

st.write("DEBUG — Raw CNN output:", raw_prediction)

malignant_probability = raw_prediction
benign_probability = 1.0 - raw_prediction
        except Exception as e:

            st.error(
                "CNN prediction failed."
            )

            st.code(str(e))

            st.stop()


    # ========================================================
    # CNN CLASSIFICATION
    # ========================================================

    if malignant_probability >= 0.50:

        predicted_class = "Malignant"

    else:

        predicted_class = "Benign"


    # ========================================================
    # CNN CONFIDENCE
    # ========================================================

    cnn_confidence = max(
        malignant_probability,
        benign_probability
    )

    if cnn_confidence >= 0.80:

        cnn_confidence_level = "High"

    elif cnn_confidence >= 0.60:

        cnn_confidence_level = "Moderate"

    else:

        cnn_confidence_level = "Low"


    # ========================================================
    # CNN RESULTS
    # ========================================================

    cnn_col1, cnn_col2, cnn_col3, cnn_col4 = (
        st.columns(4)
    )

    with cnn_col1:

        st.metric(
            "CNN Prediction",
            predicted_class
        )

    with cnn_col2:

        st.metric(
            "Malignant Probability",
            f"{malignant_probability * 100:.2f}%"
        )

    with cnn_col3:

        st.metric(
            "Benign Probability",
            f"{benign_probability * 100:.2f}%"
        )

    with cnn_col4:

        st.metric(
            "CNN Confidence",
            cnn_confidence_level
        )


    if cnn_confidence < 0.60:

        st.warning(
            "The CNN output is close to the decision boundary "
            "and should be treated as uncertain."
        )

    elif cnn_confidence < 0.80:

        st.info(
            "The CNN shows moderate confidence."
        )

    else:

        st.success(
            "The CNN shows high model confidence."
        )


    st.caption(
        "CNN model: Fine-tuned EfficientNet-based classifier. "
        
    )


    # ========================================================
    # SPR ANALYSIS
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '🔬 SPR Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    with st.spinner(
        "Generating full SPR response..."
    ):

        try:

            spr_result = predict_user_spr(
                user_ri
            )

        except Exception as e:

            st.error(
                "SPR prediction failed."
            )

            st.code(str(e))

            st.stop()


    # ========================================================
    # RI RANGE CHECK
    # ========================================================

    if not spr_result["success"]:

        st.warning(
            spr_result["message"]
        )

        st.info(
            "Please enter an RI between 1.33 and 1.42 "
            "for the validated SPR model."
        )

        st.stop()


    # ========================================================
    # HEALTHY REFERENCE
    # ========================================================

    try:

        healthy_result = (
            get_healthy_reference()
        )

    except Exception as e:

        st.error(
            "Unable to generate healthy reference SPR curve."
        )

        st.code(str(e))

        st.stop()


    # ========================================================
    # SPR METRICS
    # ========================================================

    angular_shift = (
        spr_result["spr_angle"]
        -
        healthy_result["spr_angle"]
    )


    spr_col1, spr_col2, spr_col3, spr_col4 = (
        st.columns(4)
    )

    with spr_col1:

        st.metric(
            "User RI",
            f"{spr_result['ri']:.3f}"
        )

    with spr_col2:

        st.metric(
            "User SPR Angle",
            f"{spr_result['spr_angle']:.2f}°"
        )

    with spr_col3:

        st.metric(
            "User Rmin",
            f"{spr_result['rmin']:.4f}"
        )

    with spr_col4:

        st.metric(
            "Angular Shift",
            f"{angular_shift:.2f}°"
        )


    # ========================================================
    # FULL SPR CURVE
    # ========================================================

    st.subheader(
        "Healthy Skin vs User Skin SPR Response"
    )

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    # Healthy curve
    ax.plot(
        healthy_result["angles"],
        healthy_result["curve"],
        linewidth=2.5,
        label=(
            f"Healthy Skin "
            f"(RI = {healthy_result['ri']:.3f})"
        )
    )

    # User curve
    ax.plot(
        spr_result["angles"],
        spr_result["curve"],
        linewidth=2.5,
        linestyle="--",
        label=(
            f"User Skin "
            f"(RI = {spr_result['ri']:.3f})"
        )
    )

    # Healthy resonance
    ax.axvline(
        healthy_result["spr_angle"],
        linestyle=":",
        linewidth=1.8,
        label=(
            f"Healthy SPR angle = "
            f"{healthy_result['spr_angle']:.2f}°"
        )
    )

    # User resonance
    ax.axvline(
        spr_result["spr_angle"],
        linestyle=":",
        linewidth=1.8,
        label=(
            f"User SPR angle = "
            f"{spr_result['spr_angle']:.2f}°"
        )
    )

    ax.set_xlabel(
        "Incident Angle (degrees)",
        fontsize=12
    )

    ax.set_ylabel(
        "Reflectance",
        fontsize=12
    )

    ax.set_title(
        "Full SPR Response Curve",
        fontsize=15
    )

    ax.grid(
        True,
        alpha=0.3
    )

    ax.legend()

    fig.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


    # ========================================================
    # SPR INTERPRETATION
    # ========================================================

    st.subheader(
        "SPR Result"
    )

    if angular_shift > 0:

        st.info(
            f"The predicted user SPR resonance is shifted by "
            f"{angular_shift:.2f}° relative to the healthy "
            f"reference."
        )

    elif angular_shift < 0:

        st.info(
            f"The predicted user SPR resonance is shifted by "
            f"{abs(angular_shift):.2f}° toward lower angles "
            f"relative to the healthy reference."
        )

    else:

        st.info(
            "The predicted user SPR resonance is approximately "
            "the same as the healthy reference."
        )


    # ========================================================
    # MULTIMODAL FUSION
    # ========================================================

    (
        multimodal_score,
        ri_normalized,
        ri_effect
    ) = calculate_multimodal_score(
        malignant_probability,
        spr_result["ri"],
        healthy_result["ri"]
    )


    # ========================================================
    # MULTIMODAL CLASSIFICATION
    # ========================================================

    if multimodal_score >= 0.50:

        multimodal_class = "Malignant"

    else:

        multimodal_class = "Benign"


    # ========================================================
    # MULTIMODAL CONFIDENCE
    # ========================================================

    multimodal_confidence = max(
        multimodal_score,
        1.0 - multimodal_score
    )

    if multimodal_confidence >= 0.80:

        multimodal_confidence_level = "High"

    elif multimodal_confidence >= 0.60:

        multimodal_confidence_level = "Moderate"

    else:

        multimodal_confidence_level = "Low"


    # ========================================================
    # MULTIMODAL RESULT
    # ========================================================

    st.markdown("---")

    st.markdown(
        '<div class="section-title">'
        '🧬🔬 Multimodal Fusion Result'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "The final research score combines the CNN image "
        "output with the RI-based SPR contribution."
    )


    mm_col1, mm_col2, mm_col3, mm_col4 = (
        st.columns(4)
    )

    with mm_col1:

        st.metric(
            "Final Assessment",
            multimodal_class
        )

    with mm_col2:

        st.metric(
            "Multimodal Risk Score",
            f"{multimodal_score * 100:.2f}%"
        )

    with mm_col3:

        st.metric(
            "CNN Malignant Output",
            f"{malignant_probability * 100:.2f}%"
        )

    with mm_col4:

        if ri_effect >= 0:

            effect_text = (
                f"+{ri_effect * 100:.2f}%"
            )

        else:

            effect_text = (
                f"{ri_effect * 100:.2f}%"
            )

        st.metric(
            "RI Effect",
            effect_text
        )


    # ========================================================
    # RI CONTRIBUTION DETAILS
    # ========================================================

    st.write(
        "### 🔬 RI Contribution"
    )

    contribution_col1, contribution_col2, contribution_col3 = (
        st.columns(3)
    )

    with contribution_col1:

        st.write(
            f"**Healthy Reference RI:** "
            f"{healthy_result['ri']:.3f}"
        )

    with contribution_col2:

        st.write(
            f"**User RI:** "
            f"{spr_result['ri']:.3f}"
        )

    with contribution_col3:

        if ri_effect > 0:

            st.write(
                f"**RI effect:** "
                f"+{ri_effect * 100:.2f}%"
            )

        elif ri_effect < 0:

            st.write(
                f"**RI effect:** "
                f"{ri_effect * 100:.2f}%"
            )

        else:

            st.write(
                "**RI effect:** 0.00%"
            )


    # ========================================================
    # MULTIMODAL INTERPRETATION
    # ========================================================

    if ri_effect > 0:

        st.info(
            f"The entered RI is above the healthy reference "
            f"and increases the multimodal research score by "
            f"{ri_effect * 100:.2f} percentage points."
        )

    elif ri_effect < 0:

        st.info(
            f"The entered RI is below the healthy reference "
            f"and decreases the multimodal research score by "
            f"{abs(ri_effect) * 100:.2f} percentage points."
        )

    else:

        st.info(
            "The entered RI matches the healthy reference, "
            "so it produces no change to the CNN score."
        )


    # ========================================================
    # COMPLETE ANALYSIS SUMMARY
    # ========================================================

    st.markdown("---")

    st.markdown(
        '<div class="section-title">'
        '📋 Complete Analysis Summary'
        '</div>',
        unsafe_allow_html=True
    )

    summary_col1, summary_col2 = (
        st.columns(2)
    )


    # --------------------------------------------------------
    # CNN SUMMARY
    # --------------------------------------------------------

    with summary_col1:

        st.write("### 🧬 CNN Analysis")

        st.write(
            f"**Prediction:** "
            f"{predicted_class}"
        )

        st.write(
            f"**Malignant output:** "
            f"{malignant_probability * 100:.2f}%"
        )

        st.write(
            f"**Benign output:** "
            f"{benign_probability * 100:.2f}%"
        )

        st.write(
            f"**Confidence:** "
            f"{cnn_confidence_level}"
        )


    # --------------------------------------------------------
    # SPR SUMMARY
    # --------------------------------------------------------

    with summary_col2:

        st.write("### 🔬 SPR Analysis")

        st.write(
            f"**User RI:** "
            f"{spr_result['ri']:.3f}"
        )

        st.write(
            f"**Resonance angle:** "
            f"{spr_result['spr_angle']:.2f}°"
        )

        st.write(
            f"**Minimum reflectance:** "
            f"{spr_result['rmin']:.4f}"
        )

        st.write(
            f"**Angular shift:** "
            f"{angular_shift:.2f}°"
        )


    # ========================================================
    # FINAL RESULT
    # ========================================================

    st.markdown("---")

    st.subheader(
        "🎯 Final Multimodal Assessment"
    )

    final_col1, final_col2 = st.columns(2)

    with final_col1:

        st.metric(
            "Final Result",
            multimodal_class
        )

    with final_col2:

        st.metric(
            "Multimodal Risk Score",
            f"{multimodal_score * 100:.2f}%"
        )


    # ========================================================
    # RESEARCH INTERPRETATION
    # ========================================================

    st.markdown("---")

    st.subheader(
        "Research Interpretation"
    )

    st.write(
        "The CNN provides image-based classification, while "
        "the SPR pathway provides optical information based "
        "on the measured refractive index. The RI is used as "
        "an additional optical contribution in the proposed "
        "multimodal research score."
    )

    st.warning(
        "The multimodal score is a research-prototype fusion "
        "score and is not a clinically validated cancer "
        "probability. This system should not be used as a "
        "medical diagnosis."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "⚠️ Research prototype only. This system is not a medical "
    "diagnostic device and should not be used to make clinical "
    "decisions. Results must be interpreted by qualified "
    "healthcare professionals."
)
