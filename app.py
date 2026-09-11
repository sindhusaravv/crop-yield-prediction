import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 Crop Yield Prediction Under Climate Stress")

st.write(
    "Predict crop yield using environmental and agricultural factors "
    "and analyze the effect of climate stress."
)

# Load dataset
df = pd.read_csv("Custom_Crops_yield_Historical_Dataset.csv")

st.success("Dataset loaded successfully!")

st.write("### Dataset Preview")
st.dataframe(df.head())
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# Select input features
X = df[
    [
        "Area_ha",
        "pH",
        "Total_N_kg",
        "Total_P_kg",
        "Total_K_kg",
        "Temperature_C",
        "Humidity_%",
        "Rainfall_mm",
        "Wind_Speed_m_s",
        "Solar_Radiation_MJ_m2_day"
    ]
]

# Target variable
y = df["Yield_kg_per_ha"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Create and train Ridge model
model = Ridge(alpha=1.0)
model.fit(X_train_scaled, y_train)

st.success("Machine Learning model trained successfully!")
st.write("## 🌱 Enter Crop Conditions")
crop = st.selectbox(
    "Select Crop",
    sorted(df["Crop"].unique())
)

area = st.number_input("Area (hectares)", min_value=0.0, value=1000.0)
ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)

total_n = st.number_input("Total Nitrogen (kg)", min_value=0.0, value=1000.0)
total_p = st.number_input("Total Phosphorus (kg)", min_value=0.0, value=500.0)
total_k = st.number_input("Total Potassium (kg)", min_value=0.0, value=800.0)

temperature = st.number_input("Temperature (°C)", value=25.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=70.0)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=800.0)
wind_speed = st.number_input("Wind Speed (m/s)", min_value=0.0, value=2.0)
solar = st.number_input(
    "Solar Radiation (MJ/m²/day)",
    min_value=0.0,
    value=18.0
)
if st.button("🌾 Predict Crop Yield"):

    input_data = pd.DataFrame([[
        area,
        ph,
        total_n,
        total_p,
        total_k,
        temperature,
        humidity,
        rainfall,
        wind_speed,
        solar
    ]], columns=X.columns)

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    st.success(f"Predicted Crop Yield: {prediction:.2f} kg/ha")
    st.success(
    f"🌾 Predicted Yield for {crop}: {prediction:.2f} kg/ha"
)
    st.write("## 🌡️ Climate Stress Simulation")

if st.button("🔥 Simulate Climate Stress"):

    normal_input = pd.DataFrame([[
        area,
        ph,
        total_n,
        total_p,
        total_k,
        temperature,
        humidity,
        rainfall,
        wind_speed,
        solar
    ]], columns=X.columns)

    stress_input = normal_input.copy()

    # Climate stress scenario
    stress_input["Temperature_C"] += 2
    stress_input["Rainfall_mm"] *= 0.80

    normal_prediction = model.predict(
        scaler.transform(normal_input)
    )[0]

    stress_prediction = model.predict(
        scaler.transform(stress_input)
    )[0]

    change = stress_prediction - normal_prediction
    percentage_change = (change / normal_prediction) * 100

    st.write(f"**Normal predicted yield:** {normal_prediction:.2f} kg/ha")
    st.write(f"**Climate-stress yield:** {stress_prediction:.2f} kg/ha")
    st.write(f"**Change:** {change:.2f} kg/ha")
    st.write(f"**Percentage change:** {percentage_change:.2f}%")
    st.write("## 📊 Model Performance")

st.write("The model was evaluated using MAE, RMSE and R² score.")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("MAE", "457.26 kg/ha")

with col2:
    st.metric("RMSE", "682.24 kg/ha")

with col3:
    st.metric("R² Score", "0.497")
    st.write("---")

st.caption(
    "🌱 Crop Yield Prediction Under Climate Stress | "
    "Machine Learning Project"
)

st.info(
    "💡 Climate stress scenario: temperature increased by 2°C "
    "and rainfall reduced by 20%."
)