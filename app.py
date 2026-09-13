import streamlit as st
import tensorflow as tfimport streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageStat

from spr_model import predict_user_spr, get_healthy_reference


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Skin Cancer Analysis",
    page_icon="🔬",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# INTERFACE ONLY
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --ink:#16263b; --slate:#53637a; --paper:#f4f6f9;
        --surface:#ffffff; --border:#dfe4eb; --navy:#16324a;
        --navy-deep:#0f2436; --bronze:#9c7a3c; --bronze-soft:#c9ac74;
    }

    html, body, [class*="css"] {
        font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif;
        -webkit-font-smoothing:antialiased;
        -moz-osx-font-smoothing:grayscale;
    }

    .stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"] {
        background-color:var(--paper) !important;
    }

    .block-container {
        padding-top:2.4rem;
        padding-bottom:3rem;
        max-width:1450px;
    }

    button:focus-visible,input:focus-visible,
    div[role="radiogroup"] label:focus-within {
        outline:2px solid var(--bronze);
        outline-offset:2px;
    }

    .main-title {
        text-align:center;
        font-family:'Fraunces',serif;
        font-size:58px;
        font-weight:600;
        color:var(--ink) !important;
        margin-bottom:10px;
        letter-spacing:-0.5px;
        line-height:1.15;
    }

    .subtitle {
        text-align:center;
        font-size:22px;
        font-weight:500;
        color:var(--slate) !important;
        margin-bottom:22px;
        line-height:1.6;
    }

    .header-line {
        width:96px;
        height:4px;
        margin:0 auto 32px auto;
        border-radius:10px;
        background:linear-gradient(90deg,var(--navy),var(--bronze));
    }

    .stApp p {
        font-size:19px !important;
        line-height:1.7 !important;
        color:var(--ink) !important;
    }

    .stApp label {
        font-size:19px !important;
        font-weight:500 !important;
        color:var(--ink) !important;
    }

    .stApp .stCaption,[data-testid="stCaptionContainer"] {
        font-size:17px !important;
        line-height:1.6 !important;
        color:var(--slate) !important;
    }

    .section-title {
        font-family:'Fraunces',serif;
        font-size:33px;
        font-weight:600;
        color:var(--ink) !important;
        margin-top:38px;
        margin-bottom:18px;
        padding-left:16px;
        border-left:4px solid var(--navy);
        line-height:1.3;
    }

    .section-description {
        color:var(--slate) !important;
        font-size:19px;
        line-height:1.7;
        margin-top:-6px;
        margin-bottom:24px;
    }

    .info-card {
        background:var(--surface) !important;
        border:1px solid var(--border);
        border-top:3px solid var(--bronze-soft);
        border-radius:12px;
        padding:25px 28px;
        margin-bottom:22px;
        box-shadow:0 2px 10px rgba(22,38,59,.05);
    }

    .card-title {
        font-family:'Fraunces',serif;
        font-size:23px;
        font-weight:600;
        color:var(--ink) !important;
        margin-bottom:10px;
        line-height:1.4;
    }

    .card-text {
        font-size:19px;
        color:var(--slate) !important;
        line-height:1.7;
    }

    .card-text strong { color:var(--ink) !important; }

    div[data-testid="stMetric"] {
        background:var(--surface) !important;
        border:1px solid var(--border);
        border-left:3px solid var(--navy);
        border-radius:10px;
        padding:20px 22px;
        min-height:122px;
        box-shadow:0 2px 8px rgba(22,38,59,.05);
    }

    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] * {
        color:var(--slate) !important;
        font-size:18px !important;
        font-weight:500 !important;
    }

    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] * {
        color:var(--ink) !important;
        font-size:33px !important;
        font-weight:700 !important;
    }

    div[data-testid="stFileUploader"] {
        background:var(--surface) !important;
        border:1px solid var(--border);
        border-radius:12px;
    }

    div[data-testid="stFileUploader"] section {
        background:var(--surface) !important;
    }

    div[data-testid="stFileUploader"] p,
    div[data-testid="stFileUploader"] span,
    div[data-testid="stFileUploader"] small {
        font-size:18px !important;
        color:var(--slate) !important;
    }

    div[data-baseweb="input"] {
        border-radius:9px;
        background-color:var(--surface) !important;
        border:1px solid var(--border) !important;
    }

    input {
        font-size:19px !important;
        color:var(--ink) !important;
        background-color:var(--surface) !important;
    }

    div[role="radiogroup"] {
        background:var(--surface) !important;
        border:1px solid var(--border);
        padding:15px 19px;
        border-radius:10px;
    }

    div[role="radiogroup"] label {
        font-size:19px !important;
        color:var(--ink) !important;
    }

    div.stButton > button {
        width:100%;
        height:58px;
        border-radius:9px;
        font-size:20px !important;
        font-weight:700;
        letter-spacing:.2px;
    }

    div.stButton > button[kind="primary"],
    button[data-testid="baseButton-primary"] {
        background-color:var(--navy) !important;
        border:none !important;
        color:#fff !important;
    }

    div.stButton > button[kind="primary"] *,
    button[data-testid="baseButton-primary"] * {
        color:#fff !important;
        opacity:1 !important;
    }

    div.stButton > button[kind="primary"]:hover,
    button[data-testid="baseButton-primary"]:hover {
        background-color:var(--navy-deep) !important;
    }

    div[data-testid="stAlert"] { border-radius:10px; }

    div[data-testid="stAlert"] p {
        font-size:18px !important;
        line-height:1.6 !important;
    }

    img { border-radius:10px; }

    .footer {
        text-align:center;
        color:var(--slate) !important;
        font-size:16px;
        line-height:1.7;
        padding-top:16px;
        border-top:1px solid var(--border);
        margin-top:6px;
    }

    /* ========================= DARK MODE ========================= */

    @media (prefers-color-scheme: dark) {
        :root {
            --ink:#f1f5f9;
            --slate:#c3ceda;
            --paper:#0f1720;
            --surface:#182330;
            --border:#334252;
            --navy:#4d8fc4;
            --navy-deep:#3978aa;
            --bronze:#d2ad68;
            --bronze-soft:#b99455;
        }

        html { color-scheme:dark; }

        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"] {
            background-color:#0f1720 !important;
        }

        .main-title,.section-title,.card-title {
            color:#f1f5f9 !important;
        }

        .subtitle,.section-description,.card-text {
            color:#c3ceda !important;
        }

        .stApp p { color:#d5dee8 !important; }
        .stApp label { color:#e5ebf2 !important; }

        .stApp .stCaption,
        [data-testid="stCaptionContainer"] {
            color:#aebdcb !important;
        }

        .info-card {
            background:#182330 !important;
            border-color:#334252 !important;
            border-top-color:#b99455 !important;
            box-shadow:0 4px 14px rgba(0,0,0,.25);
        }

        .card-text { color:#c3ceda !important; }
        .card-text strong { color:#f1f5f9 !important; }

        div[data-testid="stMetric"] {
            background:#182330 !important;
            border-color:#334252 !important;
            border-left-color:#4d8fc4 !important;
            box-shadow:0 4px 14px rgba(0,0,0,.25);
        }

        div[data-testid="stMetricLabel"],
        div[data-testid="stMetricLabel"] * {
            color:#b8c5d2 !important;
        }

        div[data-testid="stMetricValue"],
        div[data-testid="stMetricValue"] * {
            color:#f1f5f9 !important;
        }

        div[data-testid="stFileUploader"],
        div[data-testid="stFileUploader"] section {
            background:#182330 !important;
            border-color:#334252 !important;
        }

        div[data-testid="stFileUploader"] p,
        div[data-testid="stFileUploader"] span,
        div[data-testid="stFileUploader"] small {
            color:#c3ceda !important;
        }

        div[data-baseweb="input"] {
            background-color:#182330 !important;
            border-color:#46576a !important;
        }

        input {
            color:#f1f5f9 !important;
            background-color:#182330 !important;
        }

        div[role="radiogroup"] {
            background:#182330 !important;
            border-color:#334252 !important;
        }

        div[role="radiogroup"] label {
            color:#e5ebf2 !important;
        }

        div[data-testid="stAlert"] {
            background-color:#1b2b3b !important;
        }

        div[data-testid="stAlert"] p {
            color:#e1e8ef !important;
        }

        hr { border-color:#334252 !important; }

        .footer {
            color:#9eacbb !important;
            border-top-color:#334252 !important;
        }
    }

    /* Streamlit dark-theme attribute fallback */
    [data-theme="dark"] {
        color-scheme:dark;
    }

    [data-theme="dark"] .stApp,
    [data-theme="dark"] [data-testid="stAppViewContainer"],
    [data-theme="dark"] [data-testid="stMain"] {
        background-color:#0f1720 !important;
    }

    [data-theme="dark"] .main-title,
    [data-theme="dark"] .section-title,
    [data-theme="dark"] .card-title {
        color:#f1f5f9 !important;
    }

    [data-theme="dark"] .subtitle,
    [data-theme="dark"] .section-description,
    [data-theme="dark"] .card-text {
        color:#c3ceda !important;
    }

    [data-theme="dark"] .info-card,
    [data-theme="dark"] div[data-testid="stMetric"],
    [data-theme="dark"] div[data-testid="stFileUploader"],
    [data-theme="dark"] div[data-testid="stFileUploader"] section,
    [data-theme="dark"] div[role="radiogroup"] {
        background-color:#182330 !important;
        border-color:#334252 !important;
    }

    [data-theme="dark"] div[data-testid="stMetricValue"],
    [data-theme="dark"] div[data-testid="stMetricValue"] * {
        color:#f1f5f9 !important;
    }

    [data-theme="dark"] div[data-testid="stMetricLabel"],
    [data-theme="dark"] div[data-testid="stMetricLabel"] * {
        color:#b8c5d2 !important;
    }

    [data-theme="dark"] input {
        color:#f1f5f9 !important;
        background-color:#182330 !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🔬 Skin Cancer Analysis</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'CNN-Based Image Analysis&nbsp;&nbsp;•&nbsp;&nbsp;'
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
    '<div class="info-card">'
    '<div class="card-title">Research Prototype</div>'
    '<div class="card-text">An interactive framework combining '
    'skin-image analysis, lesion characteristics, and '
    'refractive-index-based Surface Plasmon Resonance analysis.'
    '</div>'
    '</div>',
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
    'Provide the skin-lesion image and refractive-index value '
    'required for the analysis.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# IMAGE + RI
# ============================================================

col1, col2 = st.columns(
    2,
    gap="large"
)


# ============================================================
# IMAGE INPUT
# ============================================================

with col1:

    st.markdown(
        '<div class="card-title">🖼️ Skin Lesion Image</div>'
        '<div class="card-text">Upload a clear image of the skin '
        'lesion for CNN-based analysis.</div>',
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

            # Displayed at a fixed, modest width rather than
            # stretched across the column, so the thumbnail
            # stays proportionate regardless of the source
            # image's resolution.
            st.image(
                image,
                caption="Uploaded Skin Image",
                width=280
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
        '<div class="card-title">🔬 Refractive Index</div>'
        '<div class="card-text">Enter the refractive index used as '
        'the sensing condition for the SPR analysis.</div>',
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
    'Answer the following questions based on the appearance '
    'or recent history of the lesion.'
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
        '<div class="section-title">'
        '🧬 CNN Skin Image Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    with st.spinner(
        "Analyzing skin image..."
    ):

        try:

            img_array = np.array(
                image,
                dtype=np.float32
            )

            img_resized = tf.image.resize(
                img_array,
                (224, 224)
            )

            img_input = tf.expand_dims(
                img_resized,
                axis=0
            )

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
        st.columns(
            4,
            gap="medium"
        )
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
        '<div class="section-title">'
        '🩺 Lesion Feature Analysis'
        '</div>',
        unsafe_allow_html=True
    )


    feature_col1, feature_col2, feature_col3, feature_col4 = (
        st.columns(
            4,
            gap="medium"
        )
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
        st.columns(
            4,
            gap="medium"
        )
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
        '<div class="section-title">'
        '📈 SPR Response Comparison'
        '</div>',
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
        fontsize=14
    )


    ax.set_ylabel(
        "Reflectance",
        fontsize=14
    )


    ax.set_title(
        "Full SPR Response Curve",
        fontsize=17,
        fontweight="600"
    )


    ax.tick_params(
        axis="both",
        labelsize=13
    )


    ax.grid(
        True,
        alpha=0.3
    )


    ax.legend(
        fontsize=12
    )


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
        '<div class="section-title">'
        'SPR Result'
        '</div>',
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
        st.columns(
            4,
            gap="medium"
        )
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
        '<div class="section-title">'
        '📊 Multimodal Contribution'
        '</div>',
        unsafe_allow_html=True
    )


    contribution_col1, contribution_col2, contribution_col3 = (
        st.columns(
            3,
            gap="medium"
        )
    )


    with contribution_col1:

        st.markdown(
            '<div class="info-card">'
            '<div class="card-title">CNN</div>'
            '<div class="card-text">Contribution: '
            f'<strong>{0.60 * malignant_probability * 100:.2f}%</strong>'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )


    with contribution_col2:

        st.markdown(
            '<div class="info-card">'
            '<div class="card-title">Questionnaire</div>'
            '<div class="card-text">Contribution: '
            f'<strong>{0.20 * questionnaire_score * 100:.2f}%</strong>'
            '</div>'
            '</div>',
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
            '<div class="info-card">'
            '<div class="card-title">SPR / RI</div>'
            '<div class="card-text">Effect: '
            f'<strong>{spr_effect_text}</strong>'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '🎯 Final Multimodal Assessment'
        '</div>',
        unsafe_allow_html=True
    )


    final_col1, final_col2 = (
        st.columns(
            2,
            gap="medium"
        )
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
        '<div class="section-title">'
        'Research Interpretation'
        '</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="info-card">'
        '<div class="card-text">The proposed system combines '
        'image-based CNN features, user-reported lesion '
        'characteristics, and optical SPR information. These '
        'modalities provide complementary evidence for the '
        'proposed multimodal assessment.</div>'
        '</div>',
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

st.markdown(
    '<div class="footer">'
    'Research prototype • Multimodal skin cancer analysis'
    '<br>'
    'Not intended to replace clinical diagnosis or '
    'professional medical evaluation.'
    '</div>',
    unsafe_allow_html=True
)
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageStat

from spr_model import predict_user_spr, get_healthy_reference


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Skin Cancer Analysis",
    page_icon="🔬",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# INTERFACE ONLY
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       FONTS

       Fraunces (a warm, editorial serif) carries the headline,
       Inter (a clean, highly legible grotesk) carries every
       other piece of text. The pairing is meant to read as a
       serious scientific instrument rather than a generic app.
       ======================================================== */

    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --ink: #16263b;         /* primary text / headings   */
        --slate: #53637a;       /* secondary / muted text    */
        --paper: #f4f6f9;       /* page background           */
        --surface: #ffffff;     /* card background           */
        --border: #dfe4eb;      /* hairline borders          */
        --navy: #16324a;        /* primary accent            */
        --navy-deep: #0f2436;   /* pressed / hover navy      */
        --bronze: #9c7a3c;      /* secondary accent          */
        --bronze-soft: #c9ac74; /* lighter bronze for tints  */
    }

    @media (prefers-reduced-motion: reduce) {
        * { transition: none !important; }
    }


    /* ========================================================
       GLOBAL
       ======================================================== */

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }

    /*
       This design is only tuned for a light palette. Rather than
       trying to also match Streamlit's separate dark theme (which
       swaps text/background pairings underneath these custom
       colors and breaks contrast), pin the whole app to the light
       theme regardless of the visitor's OS/browser preference or
       Streamlit's own theme setting.
    */
    html {
        color-scheme: light only;
    }

    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stHeader"],
    .main {
        background-color: var(--paper) !important;
    }

    .block-container {
        padding-top: 2.4rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    button:focus-visible,
    input:focus-visible,
    div[role="radiogroup"] label:focus-within {
        outline: 2px solid var(--bronze);
        outline-offset: 2px;
    }


    /* ========================================================
       MAIN TITLE
       ======================================================== */

    .main-title {
        text-align: center;
        font-family: 'Fraunces', serif;
        font-size: 58px;
        font-weight: 600;
        font-optical-sizing: auto;
        color: var(--ink);
        margin-bottom: 10px;
        letter-spacing: -0.5px;
        line-height: 1.15;
    }

    .subtitle {
        text-align: center;
        font-size: 22px;
        font-weight: 500;
        color: var(--slate);
        margin-bottom: 22px;
        line-height: 1.6;
    }

    .header-line {
        width: 96px;
        height: 4px;
        margin: 0 auto 32px auto;
        border-radius: 10px;
        background: linear-gradient(90deg, var(--navy), var(--bronze));
    }


    /* ========================================================
       NORMAL TEXT
       ======================================================== */

    .stApp p {
        font-size: 19px !important;
        line-height: 1.7 !important;
        color: #374151;
    }


    /* ========================================================
       INPUT LABELS
       ======================================================== */

    .stApp label {
        font-size: 19px !important;
        font-weight: 500 !important;
        color: #273142 !important;
    }


    /* ========================================================
       CAPTIONS
       ======================================================== */

    .stApp .stCaption,
    [data-testid="stCaptionContainer"] {
        font-size: 17px !important;
        line-height: 1.6 !important;
        color: var(--slate) !important;
    }


    /* ========================================================
       SECTION HEADINGS
       ======================================================== */

    .section-title {
        font-family: 'Fraunces', serif;
        font-size: 33px;
        font-weight: 600;
        color: var(--ink);
        margin-top: 38px;
        margin-bottom: 18px;
        padding-left: 16px;
        border-left: 4px solid var(--navy);
        line-height: 1.3;
    }

    .section-description {
        color: var(--slate);
        font-size: 19px;
        line-height: 1.7;
        margin-top: -6px;
        margin-bottom: 24px;
    }


    /* ========================================================
       INFORMATION CARDS

       A thin bronze rule along the top distinguishes these
       from the metric cards below (which carry a navy rule on
       the left) - two related but distinct structural cues
       rather than one repeated card style.
       ======================================================== */

    .info-card {
        background: var(--surface) !important;
        border: 1px solid var(--border);
        border-top: 3px solid var(--bronze-soft);
        border-radius: 12px;
        padding: 25px 28px;
        margin-bottom: 22px;
        box-shadow: 0 2px 10px rgba(22, 38, 59, 0.05);
    }

    .card-title {
        font-family: 'Fraunces', serif;
        font-size: 23px;
        font-weight: 600;
        color: var(--ink) !important;
        margin-bottom: 10px;
        line-height: 1.4;
    }

    .card-text {
        font-size: 19px;
        color: var(--slate) !important;
        line-height: 1.7;
    }

    .card-text strong {
        color: var(--ink) !important;
    }


    /* ========================================================
       METRIC CARDS
       ======================================================== */

    div[data-testid="stMetric"] {
        background: var(--surface) !important;
        border: 1px solid var(--border);
        border-left: 3px solid var(--navy);
        border-radius: 10px;
        padding: 20px 22px;
        min-height: 122px;
        box-shadow: 0 2px 8px rgba(22, 38, 59, 0.05);
    }

    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] * {
        color: var(--slate) !important;
        font-size: 18px !important;
        font-weight: 500 !important;
    }

    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] * {
        color: var(--ink) !important;
        font-size: 33px !important;
        font-weight: 700;
    }


    /* ========================================================
       FILE UPLOADER
       ======================================================== */

    div[data-testid="stFileUploader"] {
        background: var(--surface) !important;
        border: 1px solid var(--border);
        border-radius: 12px;
    }

    div[data-testid="stFileUploader"] section {
        background: var(--surface) !important;
    }

    div[data-testid="stFileUploader"] p,
    div[data-testid="stFileUploader"] span,
    div[data-testid="stFileUploader"] small {
        font-size: 18px !important;
        color: var(--slate) !important;
    }


    /* ========================================================
       NUMBER INPUT
       ======================================================== */

    div[data-baseweb="input"] {
        border-radius: 9px;
        background-color: var(--surface) !important;
        border: 1px solid var(--border) !important;
    }

    input {
        font-size: 19px !important;
        color: var(--ink) !important;
        background-color: var(--surface) !important;
    }


    /* ========================================================
       RADIO BUTTONS
       ======================================================== */

    div[role="radiogroup"] {
        background: var(--surface) !important;
        border: 1px solid var(--border);
        padding: 15px 19px;
        border-radius: 10px;
    }

    div[role="radiogroup"] label {
        font-size: 19px !important;
        color: #273142 !important;
    }


    /* ========================================================
       BUTTON
       ======================================================== */

    div.stButton > button {
        width: 100%;
        height: 58px;
        border-radius: 9px;
        font-size: 20px !important;
        font-weight: 700;
        letter-spacing: 0.2px;
        transition: background-color 0.15s ease, transform 0.05s ease;
    }

    div.stButton > button[kind="primary"],
    button[data-testid="baseButton-primary"] {
        background-color: var(--navy) !important;
        border: none !important;
        color: #ffffff !important;
    }

    /* Streamlit wraps the button label in its own inner element;
       force every descendant (label text, icon) to stay white so
       it can never inherit a theme-dependent, low-contrast color. */
    div.stButton > button[kind="primary"] *,
    button[data-testid="baseButton-primary"] * {
        color: #ffffff !important;
        opacity: 1 !important;
    }

    div.stButton > button[kind="primary"]:hover,
    button[data-testid="baseButton-primary"]:hover {
        background-color: var(--navy-deep) !important;
    }

    div.stButton > button[kind="primary"]:active,
    button[data-testid="baseButton-primary"]:active {
        transform: scale(0.99);
    }


    /* ========================================================
       ALERTS
       ======================================================== */

    div[data-testid="stAlert"] {
        border-radius: 10px;
    }

    div[data-testid="stAlert"] p {
        font-size: 18px !important;
        line-height: 1.6 !important;
    }


    /* ========================================================
       IMAGE

       Kept modest by default - the uploaded lesion photo is
       shown at a fixed, deliberately small width (see the
       st.image call) rather than stretched across the column.
       ======================================================== */

    img {
        border-radius: 10px;
    }


    /* ========================================================
       FOOTER
       ======================================================== */

    .footer {
        text-align: center;
        color: #7b8494;
        font-size: 16px;
        line-height: 1.7;
        padding-top: 16px;
        border-top: 1px solid var(--border);
        margin-top: 6px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🔬 Skin Cancer Analysis</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'CNN-Based Image Analysis&nbsp;&nbsp;•&nbsp;&nbsp;'
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
    '<div class="info-card">'
    '<div class="card-title">Research Prototype</div>'
    '<div class="card-text">An interactive framework combining '
    'skin-image analysis, lesion characteristics, and '
    'refractive-index-based Surface Plasmon Resonance analysis.'
    '</div>'
    '</div>',
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
    'Provide the skin-lesion image and refractive-index value '
    'required for the analysis.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# IMAGE + RI
# ============================================================

col1, col2 = st.columns(
    2,
    gap="large"
)


# ============================================================
# IMAGE INPUT
# ============================================================

with col1:

    st.markdown(
        '<div class="card-title">🖼️ Skin Lesion Image</div>'
        '<div class="card-text">Upload a clear image of the skin '
        'lesion for CNN-based analysis.</div>',
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

            # Displayed at a fixed, modest width rather than
            # stretched across the column, so the thumbnail
            # stays proportionate regardless of the source
            # image's resolution.
            st.image(
                image,
                caption="Uploaded Skin Image",
                width=280
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
        '<div class="card-title">🔬 Refractive Index</div>'
        '<div class="card-text">Enter the refractive index used as '
        'the sensing condition for the SPR analysis.</div>',
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
    'Answer the following questions based on the appearance '
    'or recent history of the lesion.'
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
        '<div class="section-title">'
        '🧬 CNN Skin Image Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    with st.spinner(
        "Analyzing skin image..."
    ):

        try:

            img_array = np.array(
                image,
                dtype=np.float32
            )

            img_resized = tf.image.resize(
                img_array,
                (224, 224)
            )

            img_input = tf.expand_dims(
                img_resized,
                axis=0
            )

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
        st.columns(
            4,
            gap="medium"
        )
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
        '<div class="section-title">'
        '🩺 Lesion Feature Analysis'
        '</div>',
        unsafe_allow_html=True
    )


    feature_col1, feature_col2, feature_col3, feature_col4 = (
        st.columns(
            4,
            gap="medium"
        )
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
        st.columns(
            4,
            gap="medium"
        )
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
        '<div class="section-title">'
        '📈 SPR Response Comparison'
        '</div>',
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
        fontsize=14
    )


    ax.set_ylabel(
        "Reflectance",
        fontsize=14
    )


    ax.set_title(
        "Full SPR Response Curve",
        fontsize=17,
        fontweight="600"
    )


    ax.tick_params(
        axis="both",
        labelsize=13
    )


    ax.grid(
        True,
        alpha=0.3
    )


    ax.legend(
        fontsize=12
    )


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
        '<div class="section-title">'
        'SPR Result'
        '</div>',
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
        st.columns(
            4,
            gap="medium"
        )
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
        '<div class="section-title">'
        '📊 Multimodal Contribution'
        '</div>',
        unsafe_allow_html=True
    )


    contribution_col1, contribution_col2, contribution_col3 = (
        st.columns(
            3,
            gap="medium"
        )
    )


    with contribution_col1:

        st.markdown(
            '<div class="info-card">'
            '<div class="card-title">CNN</div>'
            '<div class="card-text">Contribution: '
            f'<strong>{0.60 * malignant_probability * 100:.2f}%</strong>'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )


    with contribution_col2:

        st.markdown(
            '<div class="info-card">'
            '<div class="card-title">Questionnaire</div>'
            '<div class="card-text">Contribution: '
            f'<strong>{0.20 * questionnaire_score * 100:.2f}%</strong>'
            '</div>'
            '</div>',
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
            '<div class="info-card">'
            '<div class="card-title">SPR / RI</div>'
            '<div class="card-text">Effect: '
            f'<strong>{spr_effect_text}</strong>'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '🎯 Final Multimodal Assessment'
        '</div>',
        unsafe_allow_html=True
    )


    final_col1, final_col2 = (
        st.columns(
            2,
            gap="medium"
        )
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
        '<div class="section-title">'
        'Research Interpretation'
        '</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="info-card">'
        '<div class="card-text">The proposed system combines '
        'image-based CNN features, user-reported lesion '
        'characteristics, and optical SPR information. These '
        'modalities provide complementary evidence for the '
        'proposed multimodal assessment.</div>'
        '</div>',
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

st.markdown(
    '<div class="footer">'
    'Research prototype • Multimodal skin cancer analysis'
    '<br>'
    'Not intended to replace clinical diagnosis or '
    'professional medical evaluation.'
    '</div>',
    unsafe_allow_html=True
)
