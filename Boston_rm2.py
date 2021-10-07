import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
import pickle
st.set_option('deprecation.showPyplotGlobalUse', False)

st.write("""
# Boston House Price Prediction App
This app predicts the **Boston House Price**!
""")
st.write('---')

# Loads the Boston House Price Dataset
boston = datasets.load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')

def user_input_features():
    CRIM = st.sidebar.slider('CRIM', 0.0063, 88.9762, 3.6135)
    ZN = st.sidebar.slider('ZN', 0.0000, 100.0000, 11.3636)
    INDUS = st.sidebar.slider('INDUS', 0.4600, 27.7400, 11.1368)
    CHAS = st.sidebar.slider('CHAS', 0.0000, 1.0000,0.0692)
    NOX = st.sidebar.slider('NOX', 0.3850, 0.8710, 0.5547)
    RM = st.sidebar.slider('RM', 3.5610, 8.7800, 6.2846)
    AGE = st.sidebar.slider('AGE', 2.9000, 100.0000, 68.5749)
    DIS = st.sidebar.slider('DIS', 1.1296, 12.1265, 3.7950)
    RAD = st.sidebar.slider('RAD', 1.0000, 24.0000, 9.5494)
    TAX = st.sidebar.slider('TAX', 187.0000, 711.0000, 408.2372)
    PTRATIO = st.sidebar.slider('PTRATIO', 12.6000, 22.0000, 18.4555)
    B = st.sidebar.slider('B', 0.3200, 396.9000, 356.6740)
    LSTAT = st.sidebar.slider('LSTAT', 1.7300, 37.9700, 12.6531)
    
    data = {'CRIM': CRIM,
            'ZN': ZN,
            'INDUS': INDUS,
            'CHAS': CHAS,
            'NOX': NOX,
            'RM': RM,
            'AGE': AGE,
            'DIS': DIS,
            'RAD': RAD,
            'TAX': TAX,
            'PTRATIO': PTRATIO,
            'B': B,
            'LSTAT': LSTAT}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')

# Build Regression Model
model = pickle.load(open('BostonHousingPrice_model_2.pkl', 'rb'))

# Apply Model to Make Prediction
prediction = model.predict(df)

st.header('Prediction of House Median Value')
st.write(prediction)
st.write('---')

# Explaining the model's predictions using SHAP values
# https://github.com/slundberg/shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

st.header('Feature Importance')
plt.title('Feature importance based on SHAP values')
shap.summary_plot(shap_values, X)
st.pyplot(bbox_inches='tight')
st.write('---')

plt.title('Feature importance based on SHAP values (Bar)')
shap.summary_plot(shap_values, X, plot_type="bar")
st.pyplot(bbox_inches='tight')