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
    page_title="Multimodal Skin Lesion Analysis",
    page_icon="🔬",
    layout="wide"
)


# ============================================================
# CUSTOM CSS — INTERFACE ONLY
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- GLOBAL ---------- */

    .stApp {
        background-color: #f7f9fc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1450px;
    }

    /* ---------- HEADER ---------- */

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 750;
        color: #172033;
        margin-bottom: 4px;
        letter-spacing: -0.8px;
    }

    .subtitle {
        text-align: center;
        font-size: 17px;
        color: #657084;
        margin-bottom: 24px;
    }

    .header-line {
        width: 70px;
        height: 4px;
        margin: 0 auto 28px auto;
        border-radius: 10px;
        background: #3b82f6;
    }

    /* ---------- SECTION HEADERS ---------- */

    .section-title {
        font-size: 25px;
        font-weight: 700;
        color: #172033;
        margin-top: 26px;
        margin-bottom: 14px;
        padding-left: 12px;
        border-left: 4px solid #3b82f6;
    }

    .section-description {
        color: #657084;
        font-size: 14px;
        margin-top: -5px;
        margin-bottom: 18px;
    }

    /* ---------- CARDS ---------- */

    .info-card {
        background: #ffffff;
        border: 1px solid #e3e8ef;
        border-radius: 14px;
        padding: 20px 22px;
        margin-bottom: 18px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }

    .card-title {
        font-size: 18px;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 5px;
    }

    .card-text {
        font-size: 13px;
        color: #6b7280;
        line-height: 1.55;
    }

    /* ---------- METRICS ---------- */

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e3e8ef;
        border-radius: 12px;
        padding: 14px 16px;
        box-shadow: 0 2px 7px rgba(15, 23, 42, 0.04);
    }

    div[data-testid="stMetricLabel"] {
        color: #667085;
        font-size: 13px;
    }

    div[data-testid="stMetricValue"] {
        color: #172033;
        font-weight: 700;
    }

    /* ---------- INPUTS ---------- */

    div[data-testid="stFileUploader"] {
        background: #ffffff;
        border-radius: 12px;
    }

    div[data-baseweb="input"] {
        border-radius: 9px;
    }

    /* ---------- BUTTON ---------- */

    div.stButton > button {
        width: 100%;
        height: 50px;
        border-radius: 10px;
        font-size: 16px;
        font-weight: 700;
        border: none;
    }

    /* ---------- RADIO ---------- */

    div[role="radiogroup"] {
        background: #ffffff;
        border: 1px solid #e3e8ef;
        padding: 10px 14px;
        border-radius: 10px;
    }

    /* ---------- ALERTS ---------- */

    div[data-testid="stAlert"] {
        border-radius: 10px;
    }

    /* ---------- IMAGE ---------- */

    img {
        border-radius: 12px;
    }

    /* ---------- FOOTER ---------- */

    .footer {
        text-align: center;
        color: #8a94a6;
        font-size: 12px;
        padding-top: 10px;
    }

    /* ---------- RESULT BANNER ---------- */

    .result-banner {
        background: #ffffff;
        border: 1px solid #dfe5ec;
        border-radius: 14px;
        padding: 20px 24px;
        margin: 15px 0 20px 0;
        text-align: center;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
    }

    .result-label {
        font-size: 13px;
        color: #667085;
        margin-bottom: 5px;
    }

    .result-value {
        font-size: 30px;
        font-weight: 750;
        color: #172033;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🔬 Multimodal Skin Lesion Analysis</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'CNN-based Image Analysis&nbsp;&nbsp;•&nbsp;&nbsp;'
    'Lesion Characteristics&nbsp;&nbsp;•&nbsp;&nbsp;'
    'Surface Plasmon Resonance'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="header-line"></div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="info-card">
        <div class="card-title">Research Prototype</div>
        <div class="card-text">
            An interactive framework combining skin-lesion image analysis,
            lesion characteristics, and refractive-index-based SPR analysis.
        </div>
    </div>
    """,
    unsafe_allow_html=True
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

    min_ri = 1.33
    max_ri = 1.40

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

    healthy_normalized = (
        (healthy_ri - min_ri)
        /
        (max_ri - min_ri)
    )

    ri_difference = (
        ri_normalized
        -
        healthy_normalized
    )

    ri_effect = 0.20 * ri_difference

    questionnaire_effect = (
        0.20 *
        (
            questionnaire_score
            -
            0.50
        )
    )

    cnn_contribution = (
        0.60 *
        cnn_malignant_probability
    )

    multimodal_score = (
        cnn_contribution
        +
        0.20 * questionnaire_score
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
        ri_effect,
        questionnaire_effect
    )


# ============================================================
# INPUT SECTION
# ============================================================

st.markdown(
    '<div class="section-title">Patient Inputs</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Provide the skin-lesion image and refractive-index value required '
    'for the analysis.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# IMAGE + RI
# ============================================================

col1, col2 = st.columns(2, gap="large")


# ============================================================
# IMAGE INPUT
# ============================================================

with col1:

    st.markdown(
        """
        <div class="card-title">🖼️ Skin Lesion Image</div>
        <div class="card-text">
            Upload a clear image of the skin lesion for CNN-based analysis.
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_image = st.file_uploader(
        "Upload skin lesion image",
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

    st.markdown(
        """
        <div class="card-title">🔬 Refractive Index</div>
        <div class="card-text">
            Enter the refractive index used as the sensing condition
            for the SPR analysis.
        </div>
        """,
        unsafe_allow_html=True
    )

    user_ri = st.number_input(
        "Measured refractive index (RI)",
        min_value=1.30,
        max_value=1.45,
        value=1.375,
        step=0.001,
        format="%.3f"
    )

    st.info(
        "Validated SPR range: RI = 1.33–1.40"
    )

    st.caption(
        "Healthy reference RI = 1.35"
    )


# ============================================================
# LESION QUESTIONS
# ============================================================

st.markdown(
    '<div class="section-title">🩺 Lesion Characteristics</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Answer the following questions based on the appearance or '
    'recent history of the lesion.'
    '</div>',
    unsafe_allow_html=True
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
    "🔍  RUN MULTIMODAL ANALYSIS",
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
        '<div class="section-title">🧬 CNN Skin Image Analysis</div>',
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

            # Do NOT divide by 255.
            # The trained model already contains preprocessing layers.

            prediction = cnn_model.predict(
                img_input,
                verbose=0
            )

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

            # Class mapping:
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
        st.columns(4, gap="medium")
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


    st.markdown(
        '<div class="section-title">🩺 Lesion Feature Analysis</div>',
        unsafe_allow_html=True
    )

    feature_col1, feature_col2, feature_col3, feature_col4 = (
        st.columns(4, gap="medium")
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
        '<div class="section-title">🔬 SPR Analysis</div>',
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
        st.columns(4, gap="medium")
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

    st.markdown(
        '<div class="section-title">📈 SPR Response Comparison</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Healthy reference versus user-defined SPR response."
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
        fontsize=15,
        fontweight="600"
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

    st.markdown(
        '<div class="section-title">SPR Result</div>',
        unsafe_allow_html=True
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

    st.markdown(
        '<div class="section-title">'
        '🧬 + 🩺 + 🔬 Multimodal Fusion Result'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Combined analysis using CNN image evidence, lesion '
        'characteristics, and RI-based SPR information.'
        '</div>',
        unsafe_allow_html=True
    )


    mm_col1, mm_col2, mm_col3, mm_col4 = (
        st.columns(4, gap="medium")
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

    st.markdown(
        '<div class="section-title">📊 Multimodal Contribution</div>',
        unsafe_allow_html=True
    )


    contribution_col1, contribution_col2, contribution_col3 = (
        st.columns(3, gap="medium")
    )


    with contribution_col1:

        st.markdown(
            f"""
            <div class="info-card">
                <div class="card-title">CNN</div>
                <div class="card-text">
                    Contribution:
                    <strong>
                    {0.60 * malignant_probability * 100:.2f}%
                    </strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with contribution_col2:

        st.markdown(
            f"""
            <div class="info-card">
                <div class="card-title">Questionnaire</div>
                <div class="card-text">
                    Contribution:
                    <strong>
                    {0.20 * questionnaire_score * 100:.2f}%
                    </strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with contribution_col3:

        if ri_effect >= 0:

            spr_effect_text = (
                f"+{ri_effect * 100:.2f}%"
            )

        else:

            spr_effect_text = (
                f"{ri_effect * 100:.2f}%"
            )

        st.markdown(
            f"""
            <div class="info-card">
                <div class="card-title">SPR / RI</div>
                <div class="card-text">
                    Effect:
                    <strong>
                    {spr_effect_text}
                    </strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    st.markdown(
        '<div class="section-title">🎯 Final Multimodal Assessment</div>',
        unsafe_allow_html=True
    )


    final_col1, final_col2 = (
        st.columns(2, gap="medium")
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

    st.markdown(
        '<div class="section-title">Research Interpretation</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="info-card">
            <div class="card-text">
                The proposed system combines image-based CNN features,
                user-reported lesion characteristics, and optical SPR
                information. These modalities provide complementary
                evidence for the proposed multimodal assessment.
            </div>
        </div>
        """,
        unsafe_allow_html=True
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

st.markdown(
    """
    <div class="footer">
        Research prototype • Multimodal skin-lesion analysis
        <br>
        Not intended to replace clinical diagnosis or professional
        medical evaluation.
    </div>
    """,
    unsafe_allow_html=True
)
