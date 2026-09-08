
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Explainable Heart Disease Risk Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 38px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 17px;
    color: #666;
    margin-bottom: 25px;
}

.section-title {
    font-size: 25px;
    font-weight: 650;
    margin-top: 20px;
}

.info-box {
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #ddd;
    margin-bottom: 15px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD SAVED PIPELINE
# ============================================================

@st.cache_resource
def load_pipeline():

    return joblib.load(
        "/content/heart_disease_xgboost_pipeline.pkl"
    )


pipeline = load_pipeline()

preprocessor = pipeline.named_steps["preprocessor"]

model = pipeline.named_steps["classifier"]

feature_names = preprocessor.get_feature_names_out()

# ============================================================
# HUMAN-READABLE FEATURE NAMES
# ============================================================

def make_readable_feature_name(feature):

    name = feature.replace("num__", "")
    name = name.replace("cat__", "")

    if name.startswith("sex_"):
        return "Sex"

    if name.startswith("cp_"):
        return "Chest Pain Type"

    if name.startswith("fbs_"):
        return "Fasting Blood Sugar"

    if name.startswith("restecg_"):
        return "Resting ECG"

    if name.startswith("exang_"):
        return "Exercise-Induced Angina"

    if name.startswith("slope_"):
        return "ST Segment Slope"

    if name.startswith("ca_"):
        return "Number of Major Vessels"

    if name.startswith("thal_"):
        return "Thalassemia Category"

    readable = {
        "age": "Age",
        "trestbps": "Resting Blood Pressure",
        "chol": "Serum Cholesterol",
        "thalach": "Maximum Heart Rate",
        "oldpeak": "ST Depression"
    }

    return readable.get(name, name)


readable_feature_names = [
    make_readable_feature_name(feature)
    for feature in feature_names
]


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">❤️ Explainable Heart Disease Risk Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'XGBoost-based clinical risk prediction with individual SHAP explanations'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR — ABOUT MODEL
# ============================================================

with st.sidebar:

    st.header("ℹ️ About the Model")

    st.write(
        "This application uses a trained XGBoost classifier "
        "for heart disease risk prediction."
    )

    st.markdown("### Machine Learning")

    st.write("**Model:** XGBoost")

    st.write("**Explainability:** SHAP")

    st.write("**Preprocessing:**")

    st.write(
        "• Median imputation for numerical features\n"
        "• Most-frequent imputation for categorical features\n"
        "• Standard scaling\n"
        "• One-hot encoding"
    )

    st.markdown("### Clinical Features")

    st.write(
        "The model uses 13 clinical variables including "
        "age, blood pressure, cholesterol, heart rate, "
        "chest pain type and other cardiovascular indicators."
    )

    st.divider()

    st.info(
        "This is a research/educational machine-learning "
        "prototype and not a medical diagnostic system."
    )


# ============================================================
# PATIENT INPUT
# ============================================================

st.markdown(
    '<div class="section-title">🩺 Patient Information</div>',
    unsafe_allow_html=True
)

st.write(
    "Enter the patient's clinical measurements below."
)


with st.form("patient_form"):

    # ========================================================
    # BASIC / NUMERICAL FEATURES
    # ========================================================

    st.subheader("Basic Measurements")

    col1, col2, col3 = st.columns(3)

    with col1:

        age = st.number_input(
            "Age (years)",
            min_value=1,
            max_value=120,
            value=55
        )

    with col2:

        trestbps = st.number_input(
            "Resting Blood Pressure (mmHg)",
            min_value=50,
            max_value=250,
            value=140
        )

    with col3:

        chol = st.number_input(
            "Serum Cholesterol (mg/dl)",
            min_value=50,
            max_value=700,
            value=250
        )


    col1, col2, col3 = st.columns(3)

    with col1:

        thalach = st.number_input(
            "Maximum Heart Rate",
            min_value=50,
            max_value=250,
            value=150
        )

    with col2:

        oldpeak = st.number_input(
            "ST Depression (Oldpeak)",
            min_value=0.0,
            max_value=10.0,
            value=1.5,
            step=0.1
        )

    with col3:

        sex = st.selectbox(
            "Sex",
            options=[0, 1],
            format_func=lambda x:
                "Female" if x == 0 else "Male"
        )


    # ========================================================
    # CLINICAL CATEGORICAL FEATURES
    # ========================================================

    st.subheader("Clinical Indicators")

    col1, col2 = st.columns(2)

    with col1:

        cp = st.selectbox(
            "Chest Pain Type",
            options=[0, 1, 2, 3],
            format_func=lambda x: {
                0: "Type 0",
                1: "Type 1",
                2: "Type 2",
                3: "Type 3"
            }[x]
        )

        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl",
            options=[0, 1],
            format_func=lambda x:
                "No" if x == 0 else "Yes"
        )

        restecg = st.selectbox(
            "Resting ECG Result",
            options=[0, 1, 2],
            format_func=lambda x: {
                0: "Normal",
                1: "ST-T wave abnormality",
                2: "Left ventricular hypertrophy"
            }[x]
        )

        exang = st.selectbox(
            "Exercise-Induced Angina",
            options=[0, 1],
            format_func=lambda x:
                "No" if x == 0 else "Yes"
        )


    with col2:

        slope = st.selectbox(
            "ST Segment Slope",
            options=[0, 1, 2],
            format_func=lambda x: {
                0: "Upsloping",
                1: "Flat",
                2: "Downsloping"
            }[x]
        )

        ca = st.selectbox(
            "Number of Major Vessels",
            options=[0, 1, 2, 3, 4]
        )

        thal = st.selectbox(
            "Thalassemia Category",
            options=[0, 1, 2, 3],
            format_func=lambda x: {
                0: "Category 0",
                1: "Category 1",
                2: "Category 2",
                3: "Category 3"
            }[x]
        )


    submitted = st.form_submit_button(
        "🔍 Calculate Risk"
    )


# ============================================================
# PREDICTION
# ============================================================

if submitted:

    # --------------------------------------------------------
    # Create original-format dataframe
    # --------------------------------------------------------

    input_data = pd.DataFrame({

        "age": [age],

        "trestbps": [trestbps],

        "chol": [chol],

        "thalach": [thalach],

        "oldpeak": [oldpeak],

        "sex": [sex],

        "cp": [cp],

        "fbs": [fbs],

        "restecg": [restecg],

        "exang": [exang],

        "slope": [slope],

        "ca": [ca],

        "thal": [thal]
    })


    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    X_processed = preprocessor.transform(input_data)


    if hasattr(X_processed, "toarray"):

        X_processed = X_processed.toarray()


    X_processed_df = pd.DataFrame(
        X_processed,
        columns=feature_names
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model.predict(
        X_processed_df
    )[0]


    probability = model.predict_proba(
        X_processed_df
    )[0][1]


    # ========================================================
    # PREDICTION CARD
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-title">📊 Risk Assessment</div>',
        unsafe_allow_html=True
    )


    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Predicted Risk Probability",
            f"{probability * 100:.2f}%"
        )


    with col2:

        if prediction == 1:

            st.error(
                "⚠️ Higher Predicted Risk"
            )

        else:

            st.success(
                "✅ Lower Predicted Risk"
            )


    # ========================================================
    # PROBABILITY BAR
    # ========================================================

    st.subheader("Risk Probability")

    st.progress(
        float(probability)
    )

    st.caption(
        f"Model probability: {probability * 100:.2f}%"
    )


    # ========================================================
    # RISK INTERPRETATION
    # ========================================================

    if probability < 0.30:

        st.success(
            "The model estimates a relatively low predicted "
            "probability of heart disease."
        )

    elif probability < 0.60:

        st.warning(
            "The model estimates an intermediate predicted "
            "probability of heart disease."
        )

    else:

        st.error(
            "The model estimates a relatively high predicted "
            "probability of heart disease."
        )


    # ========================================================
    # SHAP
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-title">🔎 Individual Explanation</div>',
        unsafe_allow_html=True
    )

    st.write(
        "SHAP (SHapley Additive exPlanations) estimates how "
        "individual model features contributed to this prediction."
    )


    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(
        X_processed_df
    )


    if isinstance(shap_values, list):

        individual_shap = shap_values[1][0]

    elif len(np.shape(shap_values)) == 3:

        individual_shap = shap_values[0, :, 1]

    else:

        individual_shap = shap_values[0]


    # ========================================================
    # SHAP TABLE
    # ========================================================

    shap_df = pd.DataFrame({

        "Feature": feature_names,

        "Processed Value": X_processed_df.iloc[0].values,

        "SHAP Value": individual_shap,

        "Importance": np.abs(individual_shap)
    })


    shap_df = shap_df.sort_values(
        "Importance",
        ascending=False
    )
   
    # ============================================================
# HUMAN-READABLE FEATURE NAMES
# ============================================================



    st.dataframe(
        shap_df[
            [
                "Feature",
                "SHAP Value"
            ]
        ].head(10),
        use_container_width=True
    )
    
    shap_df = pd.DataFrame({

   "Feature": readable_feature_names,

    "Processed Value": X_processed_df.iloc[0].values,

    "SHAP Value": individual_shap,

    "Importance": np.abs(individual_shap)
    })

    # ========================================================
    # SHAP CHART
    # ========================================================

    st.subheader("Feature Contribution Chart")


    top_shap = shap_df.head(10).sort_values(
        "SHAP Value"
    )


    fig, ax = plt.subplots(
        figsize=(9, 5)
    )


    ax.barh(
        top_shap["Feature"],
        top_shap["SHAP Value"]
    )


    ax.axvline(
        0,
        linewidth=1
    )


    ax.set_xlabel(
        "SHAP Value"
    )


    ax.set_title(
        "Individual Feature Contributions"
    )


    plt.tight_layout()


    st.pyplot(fig)


    # ========================================================
    # POSITIVE / NEGATIVE CONTRIBUTIONS
    # ========================================================

    st.subheader("🧠 Explanation Summary")


    positive_features = shap_df[
        shap_df["SHAP Value"] > 0
    ].head(5)


    negative_features = shap_df[
        shap_df["SHAP Value"] < 0
    ].sort_values(
        "SHAP Value"
    ).head(5)


    if len(positive_features) > 0:

        st.markdown(
            "**Factors pushing the prediction toward higher risk:**"
        )

        for _, row in positive_features.iterrows():

            st.write(
                f"• **{row['Feature']}** "
                f"(SHAP = {row['SHAP Value']:.3f})"
            )


    if len(negative_features) > 0:

        st.markdown(
            "**Factors pushing the prediction toward lower risk:**"
        )

        for _, row in negative_features.iterrows():

            st.write(
                f"• **{row['Feature']}** "
                f"(SHAP = {row['SHAP Value']:.3f})"
            )


    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.divider()

    st.warning(
        "⚠️ Research/educational use only. This model does not "
        "provide a medical diagnosis and should not replace "
        "professional clinical evaluation."
    )
