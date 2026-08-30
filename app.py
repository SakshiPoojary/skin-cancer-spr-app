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
    'CNN + Clinical Features + Surface Plasmon Resonance'
    '</div>',
    unsafe_allow_html=True
)

st.info(
    "Research prototype combining image-based CNN analysis, "
    "lesion characteristics, and RI-based SPR analysis."
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
# QUESTIONNAIRE SCORE
# ============================================================

def calculate_questionnaire_score(
    change_recently,
    irregular_border,
    multiple_colors
):
    """
    Research-prototype lesion feature score.

    Each positive feature contributes equally.

    0 positive answers = 0.00
    1 positive answer  = 0.33
    2 positive answers = 0.67
    3 positive answers = 1.00
    """

    score = (
        int(change_recently)
        +
        int(irregular_border)
        +
        int(multiple_colors)
    ) / 3.0

    return float(score)


# ============================================================
# MULTIMODAL FUSION
# ============================================================

def calculate_multimodal_score(
    cnn_malignant_probability,
    questionnaire_score,
    user_ri,
    healthy_ri=1.35
):
    """
    Research-prototype multimodal fusion.

    CNN contribution       = 60%
    Questionnaire          = 20%
    SPR/RI contribution    = 20%

    IMPORTANT:
    This is a research-prototype score and NOT a clinically
    validated cancer probability.
    """

    # --------------------------------------------------------
    # VALIDATED RI RANGE
    # --------------------------------------------------------

    min_ri = 1.33
    max_ri = 1.40

    # --------------------------------------------------------
    # NORMALIZE RI
    # --------------------------------------------------------

    ri_normalized = (
        (user_ri - min_ri)
        /
        (max_ri - min_ri)
    )

    ri_normalized = float(
        np.clip(
            ri_normalized,
            0.0,
            1.0
        )
    )

    # --------------------------------------------------------
    # HEALTHY REFERENCE
    # --------------------------------------------------------

    healthy_normalized = (
        (healthy_ri - min_ri)
        /
        (max_ri - min_ri)
    )

    # --------------------------------------------------------
    # RI DIFFERENCE
    # --------------------------------------------------------

    ri_difference = (
        ri_normalized
        -
        healthy_normalized
    )

    # --------------------------------------------------------
    # RI EFFECT
    #
    # Maximum effect is approximately ±10 percentage points.
    # --------------------------------------------------------

    ri_effect = 0.20 * ri_difference

    # --------------------------------------------------------
    # QUESTIONNAIRE EFFECT
    #
    # Questionnaire score is 0 to 1.
    #
    # Center it around 0.5 so that:
    #
    # 0 positive features → decreases score
    # 1-2 features        → moderate effect
    # 3 features          → increases score
    # --------------------------------------------------------

    questionnaire_effect = (
        0.20 *
        (
            questionnaire_score
            -
            0.50
        )
    )

    # --------------------------------------------------------
    # CNN PRIMARY CONTRIBUTION
    # --------------------------------------------------------

    cnn_contribution = (
        0.60 *
        cnn_malignant_probability
    )

    # --------------------------------------------------------
    # FINAL SCORE
    # --------------------------------------------------------

    multimodal_score = (
        cnn_contribution
        +
        0.20 * questionnaire_score
        +
        ri_effect
    )

    # Keep score between 0 and 1
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
        ri_effect,
        questionnaire_effect
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


# ============================================================
# IMAGE + RI
# ============================================================

col1, col2 = st.columns(2)


# ============================================================
# IMAGE INPUT
# ============================================================

with col1:

    st.subheader("🖼️ Skin Image")

    uploaded_image = st.file_uploader(
        "Upload a skin lesion image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        help="Upload a clear skin lesion image."
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
        "Validated SPR range: RI = 1.33–1.40"
    )

    st.caption(
        "Healthy reference RI = 1.35"
    )


# ============================================================
# LESION QUESTIONS
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">'
    '🩺 Lesion Characteristics'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "Answer the following questions based on the appearance "
    "or recent history of the lesion."
)


q1 = st.radio(
    "1. Has the lesion changed in size, shape, or color recently?",
    [
        "No",
        "Yes"
    ],
    horizontal=True
)


q2 = st.radio(
    "2. Does the lesion have an irregular or uneven border?",
    [
        "No",
        "Yes"
    ],
    horizontal=True
)


q3 = st.radio(
    "3. Does the lesion have multiple or uneven colors?",
    [
        "No",
        "Yes"
    ],
    horizontal=True
)


# Convert answers to Boolean values
change_recently = (
    q1 == "Yes"
)

irregular_border = (
    q2 == "Yes"
)

multiple_colors = (
    q3 == "Yes"
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
    # IMAGE CHECK
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

            # Convert image to NumPy
            img_array = np.array(
                image,
                dtype=np.float32
            )

            # Resize
            img_resized = tf.image.resize(
                img_array,
                (224, 224)
            )

            # Add batch dimension
            img_input = tf.expand_dims(
                img_resized,
                axis=0
            )

            # IMPORTANT:
            # Do NOT divide by 255.
            # The trained model already contains
            # preprocessing layers.

            prediction = cnn_model.predict(
                img_input,
                verbose=0
            )

            # Raw model output
            raw_prediction = float(
                np.asarray(prediction)
                .reshape(-1)[0]
            )

            raw_prediction = float(
                np.clip(
                    raw_prediction,
                    0.0,
                    1.0
                )
            )

            # Class mapping confirmed during training:
            # Benign = 0
            # Malignant = 1

            malignant_probability = (
                raw_prediction
            )

            benign_probability = (
                1.0 -
                malignant_probability
            )

        except Exception as e:

            st.error(
                "CNN prediction failed."
            )

            st.code(
                str(e)
            )

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


    st.caption(
        "EfficientNet-based CNN | "
        "Test accuracy: 91.75%"
    )


    # ========================================================
    # LESION QUESTIONNAIRE RESULT
    # ========================================================

    questionnaire_score = (
        calculate_questionnaire_score(
            change_recently,
            irregular_border,
            multiple_colors
        )
    )


    st.markdown("---")

    st.subheader(
        "🩺 Lesion Feature Analysis"
    )


    feature_col1, feature_col2, feature_col3, feature_col4 = (
        st.columns(4)
    )


    with feature_col1:

        st.metric(
            "Recent Change",
            "Yes" if change_recently else "No"
        )


    with feature_col2:

        st.metric(
            "Irregular Border",
            "Yes" if irregular_border else "No"
        )


    with feature_col3:

        st.metric(
            "Multiple Colors",
            "Yes" if multiple_colors else "No"
        )


    with feature_col4:

        st.metric(
            "Feature Score",
            f"{questionnaire_score * 100:.0f}%"
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

            st.code(
                str(e)
            )

            st.stop()


    # ========================================================
    # RI VALIDATION
    # ========================================================

    if not spr_result["success"]:

        st.warning(
            spr_result["message"]
        )

        st.info(
            "Please enter an RI between 1.33 and 1.40 "
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

        st.code(
            str(e)
        )

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
    # SPR CURVE
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
        ri_effect,
        questionnaire_effect
    ) = calculate_multimodal_score(
        malignant_probability,
        questionnaire_score,
        spr_result["ri"],
        healthy_result["ri"]
    )


    # ========================================================
    # FINAL MULTIMODAL CLASSIFICATION
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
        1.0 -
        multimodal_score
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
        '🧬 + 🩺 + 🔬 Multimodal Fusion Result'
        '</div>',
        unsafe_allow_html=True
    )


    st.write(
        "The final research score combines three information "
        "sources: CNN image evidence, lesion characteristics, "
        "and RI-based SPR information."
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
            "CNN Output",
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
    # CONTRIBUTION BREAKDOWN
    # ========================================================

    st.subheader(
        "📊 Multimodal Contribution"
    )


    contribution_col1, contribution_col2, contribution_col3 = (
        st.columns(3)
    )


    with contribution_col1:

        st.write(
            f"**CNN contribution:** "
            f"{0.60 * malignant_probability * 100:.2f}%"
        )


    with contribution_col2:

        st.write(
            f"**Questionnaire contribution:** "
            f"{0.20 * questionnaire_score * 100:.2f}%"
        )


    with contribution_col3:

        if ri_effect >= 0:

            st.write(
                f"**SPR/RI effect:** "
                f"+{ri_effect * 100:.2f}%"
            )

        else:

            st.write(
                f"**SPR/RI effect:** "
                f"{ri_effect * 100:.2f}%"
            )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    st.markdown("---")

    st.subheader(
        "🎯 Final Multimodal Assessment"
    )


    final_col1, final_col2 = (
        st.columns(2)
    )


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
        "The proposed system combines image-based CNN "
        "features, user-reported lesion characteristics, "
        "and optical SPR information. These modalities "
        "provide complementary evidence for the proposed "
        "multimodal assessment."
    )


    st.warning(
        "The multimodal score is a research-prototype "
        "fusion score and is not a clinically validated "
        "cancer probability. It should not be used as a "
        "medical diagnosis."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "⚠️ Research prototype only. Not intended to replace "
    "clinical diagnosis or professional medical evaluation."
)
