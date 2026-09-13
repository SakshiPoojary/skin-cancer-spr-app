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

    /* ---------- MAIN TITLE ---------- */

    .main-title {
        text-align: center;
        font-size: 44px;
        font-weight: 750;
        color: #172033;
        margin-bottom: 6px;
        letter-spacing: -0.8px;
    }

    .subtitle {
        text-align: center;
        font-size: 19px;
        font-weight: 500;
        color: #596579;
        margin-bottom: 22px;
    }

    .header-line {
        width: 75px;
        height: 4px;
        margin: 0 auto 28px auto;
        border-radius: 10px;
        background: #3b82f6;
    }

    /* ---------- NORMAL TEXT ---------- */

    .stApp p {
        font-size: 16px;
        line-height: 1.6;
        color: #374151;
    }

    .stApp label {
        font-size: 16px !important;
        font-weight: 500;
        color: #273142 !important;
    }

    .stApp .stCaption {
        font-size: 14px !important;
    }

    /* ---------- SECTION HEADINGS ---------- */

    .section-title {
        font-size: 27px;
        font-weight: 700;
        color: #172033;
        margin-top: 30px;
        margin-bottom: 15px;
        padding-left: 13px;
        border-left: 4px solid #3b82f6;
        line-height: 1.3;
    }

    .section-description {
        color: #596579;
        font-size: 16px;
        line-height: 1.6;
        margin-top: -5px;
        margin-bottom: 20px;
    }

    /* ---------- CARD ---------- */

    .info-card {
        background: #ffffff;
        border: 1px solid #e1e6ee;
        border-radius: 14px;
        padding: 21px 24px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }

    .card-title {
        font-size: 20px;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 7px;
    }

    .card-text {
        font-size: 16px;
        color: #596579;
        line-height: 1.65;
    }

    /* ---------- METRICS ---------- */

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e1e6ee;
        border-radius: 12px;
        padding: 16px 18px;
        min-height: 105px;
        box-shadow: 0 2px 7px rgba(15, 23, 42, 0.04);
    }

    div[data-testid="stMetricLabel"] {
        color: #596579;
        font-size: 15px;
    }

    div[data-testid="stMetricValue"] {
        color: #172033;
        font-size: 25px;
        font-weight: 700;
    }

    /* ---------- FILE UPLOADER ---------- */

    div[data-testid="stFileUploader"] {
        background: #ffffff;
        border-radius: 12px;
    }

    /* ---------- INPUT ---------- */

    div[data-baseweb="input"] {
        border-radius: 9px;
    }

    input {
        font-size: 16px !important;
    }

    /* ---------- BUTTON ---------- */

    div.stButton > button {
        width: 100%;
        height: 52px;
        border-radius: 10px;
        font-size: 17px;
        font-weight: 700;
    }

    /* ---------- RADIO BUTTONS ---------- */

    div[role="radiogroup"] {
        background: #ffffff;
        border: 1px solid #e1e6ee;
        padding: 12px 16px;
        border-radius: 10px;
    }

    div[role="radiogroup"] label {
        font-size: 16px !important;
    }

    /* ---------- ALERTS ---------- */

    div[data-testid="stAlert"] {
        border-radius: 10px;
    }

    div[data-testid="stAlert"] p {
        font-size: 15px !important;
    }

    /* ---------- IMAGE ---------- */

    img {
        border-radius: 12px;
    }

    /* ---------- FOOTER ---------- */

    .footer {
        text-align: center;
        color: #7b8494;
        font-size: 14px;
        line-height: 1.6;
        padding-top: 12px;
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
    """
    <div class="info-card">
        <div class="card-title">Research Prototype</div>
        <div class="card-text">
            An interactive framework combining skin-image analysis,
            lesion characteristics, and refractive-index-based
            Surface Plasmon Resonance analysis.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
