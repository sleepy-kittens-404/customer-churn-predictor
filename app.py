import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Load model and data
@st.cache_resource
def load_model():
    return joblib.load('best_model.pkl')

model = load_model()

@st.cache_resource
def load_transformer():
    return joblib.load('transformer.pkl')

transformer = load_transformer()

st.title("Customer Churn Predictor")
st.write("Enter customer details to predict whether they will churn.")
st.sidebar.header("Customer Details")

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 30)
monthly_charges = st.sidebar.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=50.0,step=1.0)
total_charges = st.sidebar.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=600.0,step=1.0)
gender = st.sidebar.selectbox("Gender",["Male","Female"])
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment_method = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

online_security = st.sidebar.selectbox("Online Security", ["Yes", "No"])
tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
# We will not take input for the other features as it will be too much work for user
#Also shap values told us which features are most important for the model
senior_citizen = '0'
partner = 'No'
dependents = 'No'
phone_service = 'Yes'
multiple_lines = 'No'
online_backup = 'No'
device_protection = 'No'
streaming_tv = 'No'
streaming_movies = 'No'
tenure_scaled = (tenure - 32.83907611) / 24.63776656
monthly_scaled = (monthly_charges - 64.89881674) / 30.07248021
total_scaled = (total_charges - 2311.01193203) / 2263.31989595

# Manual OneHotEncoding
def ohe(value, categories):
    return [1 if value == cat else 0 for cat in categories]

encoded = (
    [tenure_scaled, monthly_scaled, total_scaled] +
    ohe(gender, ['Female', 'Male']) +
    ohe(str(senior_citizen), ['0', '1']) +
    ohe(partner, ['No', 'Yes']) +
    ohe(dependents, ['No', 'Yes']) +
    ohe(phone_service, ['No', 'Yes']) +
    ohe(multiple_lines, ['No', 'No phone service', 'Yes']) +
    ohe(internet_service, ['DSL', 'Fiber optic', 'No']) +
    ohe(online_security, ['No', 'No internet service', 'Yes']) +
    ohe(online_backup, ['No', 'No internet service', 'Yes']) +
    ohe(device_protection, ['No', 'No internet service', 'Yes']) +
    ohe(tech_support, ['No', 'No internet service', 'Yes']) +
    ohe(streaming_tv, ['No', 'No internet service', 'Yes']) +
    ohe(streaming_movies, ['No', 'No internet service', 'Yes']) +
    ohe(contract, ['Month-to-month', 'One year', 'Two year']) +
    ohe(paperless_billing, ['No', 'Yes']) +
    ohe(payment_method, ['Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check', 'Mailed check'])
)

input_transformed = np.array(encoded).reshape(1, -1)

if st.button("Predict"):
    probability = model.predict_proba(input_transformed)[0][1]
    prediction = int(probability >= 0.4)
    
    if prediction == 1:
        st.error(f"⚠️ This customer is likely to churn. Probability: {probability:.2%}")
    else:
        st.success(f"✅ This customer is not likely to churn. Probability: {probability:.2%}")

    st.subheader("Why did the model make this prediction?")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_transformed)

    user_feature_indices = {
    'tenure': 0,
    'MonthlyCharges': 1,
    'TotalCharges': 2,
    'Contract': [37, 38, 39],
    'InternetService': [16, 17, 18],
    'PaymentMethod': [42, 43, 44, 45],
    'OnlineSecurity': [19, 20, 21],
    'TechSupport': [28, 29, 30],
    'PaperlessBilling': [40, 41],
    'Gender': [3, 4]
    }

    selected_indices = []
    selected_names = []

    for feature, idx in user_feature_indices.items():
        if isinstance(idx, list):
            max_idx = max(idx, key=lambda i: abs(shap_values[0][i]))
            selected_indices.append(max_idx)
        else:
            selected_indices.append(idx)
        selected_names.append(feature)

    selected_shap = [shap_values[0][i] for i in selected_indices]

    shap_df = pd.DataFrame({'Feature': selected_names, 'SHAP Value': selected_shap})
    shap_df = shap_df.reindex(shap_df['SHAP Value'].abs().sort_values(ascending=False).index)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['red' if v > 0 else 'blue' for v in shap_df['SHAP Value']]
    ax.barh(shap_df['Feature'], shap_df['SHAP Value'], color=colors)
    ax.set_xlabel('SHAP Value (impact on churn prediction)')
    ax.set_title('Factors influencing this prediction')
    ax.axvline(x=0, color='black', linewidth=0.8)
    plt.tight_layout()
    st.pyplot(fig)
    

    fig2, ax2 = plt.subplots(figsize=(8, 4), subplot_kw={'projection': 'polar'})

    probability_pct = probability * 100
    angle = np.pi * ( probability)

    ax2.set_theta_offset(np.pi)
    ax2.set_theta_direction(-1)
    ax2.set_thetamin(0)
    ax2.set_thetamax(180)

    # Background arc colors
    theta_green = np.linspace(0, np.pi * 0.5, 100)
    theta_yellow = np.linspace(np.pi * 0.5, np.pi * 0.75, 100)
    theta_red = np.linspace(np.pi * 0.75, np.pi, 100)

    ax2.fill_between(theta_green, 0.7, 1.0, color='green', alpha=0.3)
    ax2.fill_between(theta_yellow, 0.7, 1.0, color='orange', alpha=0.3)
    ax2.fill_between(theta_red, 0.7, 1.0, color='red', alpha=0.3)

    # Needle
    ax2.annotate('', xy=(angle, 0.75), xytext=(angle, 0),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

    ax2.set_ylim(0, 1)
    ax2.set_yticks([])
    ax2.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
    ax2.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax2.set_title(f'Churn Risk: {probability_pct:.1f}%', pad=20, fontsize=14)
    ax2.spines['polar'].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig2)