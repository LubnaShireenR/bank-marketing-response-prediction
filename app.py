import json
import streamlit as st
import pandas as pd
import joblib

# ----------------------------------
# PAGE CONFIGURATION
# ----------------------------------

st.set_page_config(
    page_title="Bank Marketing Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------
# CUSTOM CSS
# ----------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #F4F7FB;
}

.main-title {
    color: #142D4E;
    font-size: 36px;
    font-weight: 800;
}

.subtitle {
    color: #64748B;
    font-size: 16px;
}

.section-title {
    color: #17365D;
    font-size: 22px;
    font-weight: 700;
    margin-top: 15px;
}

div[data-testid="stForm"] {
    background-color: white;
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #DCE5EF;
}

div.stButton > button,
div[data-testid="stFormSubmitButton"] button {
    background-color: #17365D;
    color: white;
    border-radius: 8px;
    border: none;
    height: 48px;
    font-weight: 600;
}

div.stButton > button:hover,
div[data-testid="stFormSubmitButton"] button:hover {
    background-color: #28558A;
    color: white;
}

.result-card {
    background-color: white;
    padding: 25px;
    border-radius: 14px;
    border: 1px solid #DCE5EF;
    margin-top: 15px;
}

.footer {
    color: #64748B;
    text-align: center;
    font-size: 13px;
    padding: 25px;
}

</style>
""", unsafe_allow_html=True)


# ----------------------------------
# LOAD MODEL
# ----------------------------------


@st.cache_resource
def load_artifacts():

    # Load trained machine learning model
    model = joblib.load(
        "models/bank_marketing_model.pkl"
    )

    # Load preprocessing pipeline
    preprocessor = joblib.load(
        "models/bank_marketing_preprocessor.pkl"
    )

    # Load model evaluation metrics
    with open("models/model_metrics.json", "r") as file:
        metrics = json.load(file)

    return model, preprocessor, metrics


# Load all saved artifacts
model, preprocessor, model_metrics = load_artifacts()



# ----------------------------------
# SIDEBAR
# ----------------------------------

with st.sidebar:

    st.title("🏦 Bank Intelligence")

    st.markdown("---")

    st.subheader("About the Project")

    st.write(
        "An intelligent bank marketing prediction "
        "system developed using Machine Learning."
    )

    st.markdown("---")

    st.subheader("Technology Stack")

    st.write("🐍 Python")
    st.write("📊 Pandas & NumPy")
    st.write("🤖 Scikit-learn")
    st.write("🎨 Streamlit")

    st.markdown("---")

    st.subheader("Prediction Target")

    st.write(
        "Predict whether a customer will subscribe "
        "to a bank term deposit."
    )

    st.markdown("---")

    st.caption("Machine Learning Portfolio Project")


# ----------------------------------
# MAIN HEADER
# ----------------------------------

st.markdown(
    '<div class="main-title">Bank Marketing Response Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Customer Subscription Intelligence '
    '| Machine Learning Application</div>',
    unsafe_allow_html=True
)

st.markdown("---")
# MODEL PERFORMANCE DASHBOARD

st.markdown("### Model Performance")

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Accuracy",
    f"{model_metrics['accuracy'] * 100:.2f}%"
)

m2.metric(
    "Precision",
    f"{model_metrics['precision'] * 100:.2f}%"
)

m3.metric(
    "Recall",
    f"{model_metrics['recall'] * 100:.2f}%"
)

m4.metric(
    "F1 Score",
    f"{model_metrics['f1_score'] * 100:.2f}%"
)

st.caption(
    f"Selected Model: {model_metrics['model_name']} "
    f"| ROC-AUC: {model_metrics['roc_auc']:.4f}"
)

st.markdown("---")

st.info(
    "Enter customer information below to estimate "
    "their likelihood of subscribing to a bank term deposit."
)

# ----------------------------------
# CUSTOMER INPUT FORM
# ----------------------------------

st.markdown(
    '<div class="section-title">01 | Customer Information</div>',
    unsafe_allow_html=True
)

with st.form("customer_form"):

    col1, col2, col3 = st.columns(3)

    # CUSTOMER DETAILS

    with col1:

        st.markdown("#### Personal Details")

        age = st.number_input(
            "Age",
            min_value=17,
            max_value=100,
            value=35
        )

        job = st.selectbox(
            "Job",
            [
                "admin.", "blue-collar", "entrepreneur",
                "housemaid", "management", "retired",
                "self-employed", "services", "student",
                "technician", "unemployed", "unknown"
            ]
        )

        marital = st.selectbox(
            "Marital Status",
            ["married", "single", "divorced", "unknown"]
        )

        education = st.selectbox(
            "Education",
            [
                "basic.4y", "basic.6y", "basic.9y",
                "high.school", "illiterate",
                "professional.course",
                "university.degree", "unknown"
            ]
        )

        default = st.selectbox(
            "Credit in Default",
            ["no", "yes", "unknown"]
        )

        housing = st.selectbox(
            "Housing Loan",
            ["yes", "no", "unknown"]
        )

        loan = st.selectbox(
            "Personal Loan",
            ["yes", "no", "unknown"]
        )

    # CAMPAIGN DETAILS

    with col2:

        st.markdown("#### Campaign Details")

        contact = st.selectbox(
            "Contact Type",
            ["cellular", "telephone"]
        )

        month = st.selectbox(
            "Last Contact Month",
            [
                "mar", "apr", "may", "jun", "jul",
                "aug", "sep", "oct", "nov", "dec"
            ]
        )

        day_of_week = st.selectbox(
            "Last Contact Day",
            ["mon", "tue", "wed", "thu", "fri"]
        )

        campaign = st.number_input(
            "Campaign Contacts",
            min_value=1,
            value=1
        )

        pdays = st.number_input(
            "Days Since Previous Contact (999 = Never)",
            min_value=0,
            value=999
        )

        previous = st.number_input(
            "Previous Contacts",
            min_value=0,
            value=0
        )

        poutcome = st.selectbox(
            "Previous Campaign Outcome",
            ["failure", "nonexistent", "success"]
        )

    # ECONOMIC DETAILS

    with col3:

        st.markdown("#### Economic Indicators")

        emp_var_rate = st.number_input(
            "Employment Variation Rate",
            value=1.1
        )

        cons_price_idx = st.number_input(
            "Consumer Price Index",
            value=93.994
        )

        cons_conf_idx = st.number_input(
            "Consumer Confidence Index",
            value=-36.4
        )

        euribor3m = st.number_input(
            "Euribor 3 Month Rate",
            value=4.857
        )

        nr_employed = st.number_input(
            "Number of Employees",
            value=5191.0
        )

    st.markdown("---")

    submitted = st.form_submit_button(
        "🔍 Predict Customer Subscription",
        use_container_width=True
    )


# ----------------------------------
# PREDICTION
# ----------------------------------

if submitted:

    customer_data = pd.DataFrame([{

        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "month": month,
        "day_of_week": day_of_week,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome,
        "emp.var.rate": emp_var_rate,
        "cons.price.idx": cons_price_idx,
        "cons.conf.idx": cons_conf_idx,
        "euribor3m": euribor3m,
        "nr.employed": nr_employed

    }])

    processed_data = preprocessor.transform(
        customer_data
    )

    prediction = model.predict(processed_data)[0]

    probability = model.predict_proba(
        processed_data
    )[0][1] * 100

    # RESULT SECTION

    st.markdown("---")

    st.markdown(
        '<div class="section-title">02 | Prediction Results</div>',
        unsafe_allow_html=True
    )

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        st.markdown("#### Subscription Prediction")

        if prediction == 1:

            st.success(
                "Customer is predicted to subscribe!"
            )

        else:

            st.warning(
                "Customer is predicted not to subscribe."
            )

    with result_col2:

        st.markdown("#### Subscription Probability")

        st.metric(
            label="Estimated Probability",
            value=f"{probability:.2f}%"
        )

    st.progress(float(probability / 100))

    st.caption(
        "The classification uses the model's default "
        "0.5 probability threshold."
    )

    st.markdown(
        '<div class="result-card">'
        '<b>Business Insight</b><br><br>'
        'This prediction can support customer segmentation '
        'and help banks prioritize marketing campaigns. '
        'It is an estimate, not a guarantee of customer behaviour.'
        '</div>',
        unsafe_allow_html=True
    )


# ----------------------------------
# FOOTER
# ----------------------------------

st.markdown("---")

st.markdown(
    '<div class="footer">'
    'Bank Marketing Response Prediction System '
    '| Developed using Python, Machine Learning and Streamlit'
    '</div>',
    unsafe_allow_html=True
)
