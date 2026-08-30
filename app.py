import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

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
    '<div class="main-title">🔬 Skin Cancer Multimodal Analysis</div>',
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
# INPUT SECTION
# ============================================================

st.markdown(
    '<div class="section-title">Patient Inputs</div>',
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

        except Exception:

            st.error(
                "Unable to read the uploaded image."
            )


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
        "Healthy reference: RI = 1.35"
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
# ANALYSIS
# ============================================================

if analyze:

    # --------------------------------------------------------
    # CHECK IMAGE
    # --------------------------------------------------------

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

    with st.spinner("Analyzing skin image..."):

        try:

            # ----------------------------------------------
            # Resize to model input size
            # ----------------------------------------------

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

            # ----------------------------------------------
            # IMPORTANT:
            # Do NOT divide by 255 here.
            #
            # The trained CNN already contains:
            # Rescaling + Normalization layers.
            # ----------------------------------------------

            prediction = cnn_model.predict(
                img_input,
                verbose=0
            )

            malignant_probability = float(
                np.asarray(prediction).reshape(-1)[0]
            )

            benign_probability = (
                1.0 - malignant_probability
            )

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
    # CNN RESULT
    # ========================================================

    cnn_col1, cnn_col2, cnn_col3 = st.columns(3)

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


    st.caption(
        "CNN model: EfficientNet-based classifier. "
        "Test accuracy: 91.75%."
    )


    # ========================================================
    # SPR ANALYSIS
    # ========================================================

    st.markdown(
        '<div class="section-title">🔬 SPR Analysis</div>',
        unsafe_allow_html=True
    )

    with st.spinner("Generating full SPR response..."):

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
            "Please enter an RI between 1.33 and 1.40 "
            "for the validated SPR model."
        )

        st.stop()


    # ========================================================
    # HEALTHY REFERENCE
    # ========================================================

    try:

        healthy_result = get_healthy_reference()

    except Exception as e:

        st.error(
            "Unable to generate healthy reference SPR curve."
        )

        st.code(str(e))

        st.stop()


    # ========================================================
    # SPR METRICS
    # ========================================================

    spr_col1, spr_col2, spr_col3, spr_col4 = st.columns(4)

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

        angular_shift = (
            spr_result["spr_angle"]
            -
            healthy_result["spr_angle"]
        )

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

    # --------------------------------------------------------
    # HEALTHY CURVE
    # --------------------------------------------------------

    ax.plot(
        healthy_result["angles"],
        healthy_result["curve"],
        linewidth=2.5,
        label=(
            f"Healthy Skin "
            f"(RI = {healthy_result['ri']:.3f})"
        )
    )


    # --------------------------------------------------------
    # USER CURVE
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # HEALTHY RESONANCE
    # --------------------------------------------------------

    ax.axvline(
        healthy_result["spr_angle"],
        linestyle=":",
        linewidth=1.8,
        label=(
            f"Healthy SPR angle = "
            f"{healthy_result['spr_angle']:.2f}°"
        )
    )


    # --------------------------------------------------------
    # USER RESONANCE
    # --------------------------------------------------------

    ax.axvline(
        spr_result["spr_angle"],
        linestyle=":",
        linewidth=1.8,
        label=(
            f"User SPR angle = "
            f"{spr_result['spr_angle']:.2f}°"
        )
    )


    # --------------------------------------------------------
    # AXIS LABELS
    # --------------------------------------------------------

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
    # COMBINED SUMMARY
    # ========================================================

    st.markdown("---")

    st.markdown(
        '<div class="section-title">📋 Analysis Summary</div>',
        unsafe_allow_html=True
    )

    summary_col1, summary_col2 = st.columns(2)


    with summary_col1:

        st.write("### CNN")

        st.write(
            f"**Prediction:** {predicted_class}"
        )

        st.write(
            f"**Malignant probability:** "
            f"{malignant_probability * 100:.2f}%"
        )

        st.write(
            f"**Benign probability:** "
            f"{benign_probability * 100:.2f}%"
        )


    with summary_col2:

        st.write("### SPR")

        st.write(
            f"**User RI:** {spr_result['ri']:.3f}"
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
    # RESEARCH INTERPRETATION
    # ========================================================

    st.markdown("---")

    st.subheader(
        "Research Interpretation"
    )

    st.write(
        "The CNN provides an image-based classification result, "
        "while the SPR component provides an optical response "
        "based on the entered refractive index. These two "
        "measurements are presented together as complementary "
        "information in the proposed multimodal system."
    )


# ============================================================
# FOOTER / DISCLAIMER
# ============================================================

st.markdown("---")

st.caption(
    "⚠️ Research prototype only. This system is not a medical "
    "diagnostic device and should not be used to make clinical "
    "decisions. Results must be interpreted by qualified "
    "healthcare professionals."
)
st.markdown("---")
st.subheader("🧪 Multimodal Fusion Test")

test_ri = st.selectbox(
    "Test RI",
    [1.33, 1.35, 1.38, 1.40]
)

test_cnn = st.number_input(
    "Test CNN malignant probability",
    min_value=0.0,
    max_value=1.0,
    value=0.80,
    step=0.01
)

test_score, test_ri_score = calculate_multimodal_score(
    test_cnn,
    test_ri,
    1.35
)

st.write("CNN probability:", f"{test_cnn * 100:.2f}%")
st.write("RI optical score:", f"{test_ri_score * 100:.2f}%")
st.write("Multimodal score:", f"{test_score * 100:.2f}%")
