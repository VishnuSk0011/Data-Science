import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(page_title="Energy Forecasting", layout="wide")

st.title(" Energy Demand Forecasting")
st.write("Forecasting **next 30 days energy demand** using LSTM")

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

model = load_model("lstm_model.h5", compile=False)

# ---------------------------------------------------
# LOAD SCALER
# ---------------------------------------------------

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ---------------------------------------------------
# LOAD METRICS
# ---------------------------------------------------

with open("metrics.pkl", "rb") as f:
    metrics = pickle.load(f)

# ---------------------------------------------------
# LOAD LAST SEQUENCE
# ---------------------------------------------------

last_sequence = np.load("last_sequence.npy")

# ---------------------------------------------------
# LOAD LAST DATASET DATE
# ---------------------------------------------------

with open("last_date.pkl", "rb") as f:
    last_date = pickle.load(f)

# ---------------------------------------------------
# SEQUENCE INFO
# ---------------------------------------------------

SEQ_LEN = last_sequence.shape[0]
N_FEATURES = last_sequence.shape[1]

# ---------------------------------------------------
# SIDEBAR SETTINGS
# ---------------------------------------------------

st.sidebar.header("Forecast Settings")

days = st.sidebar.slider(
    "Select forecast days",
    min_value=1,
    max_value=30,
    value=30
)

future_steps = days * 24

# ---------------------------------------------------
# FORECAST LOOP
# ---------------------------------------------------

predictions = []

current_seq = last_sequence.copy()
current_time = last_date

for i in range(future_steps):

    pred = model.predict(
        current_seq.reshape(1, SEQ_LEN, N_FEATURES),
        verbose=0
    )

    pred_value = pred[0][0]
    predictions.append(pred_value)

    # move forward in time
    current_time = current_time + pd.Timedelta(hours=1)

    hour = current_time.hour
    day = current_time.day
    month = current_time.month
    dayofweek = current_time.dayofweek
    year = current_time.year

    is_weekend = 1 if dayofweek >= 5 else 0

    # one-hot season encoding
    season_spring = 1 if month in [3,4,5] else 0
    season_summer = 1 if month in [6,7,8] else 0
    season_winter = 1 if month in [12,1,2] else 0

    next_row = np.array([
        pred_value,
        hour,
        day,
        month,
        dayofweek,
        year,
        is_weekend,
        season_spring,
        season_summer,
        season_winter
    ])

    # update rolling window
    current_seq = np.vstack((current_seq[1:], next_row))

# ---------------------------------------------------
# INVERSE SCALING
# ---------------------------------------------------

dummy = np.zeros((len(predictions), N_FEATURES))

dummy[:,0] = predictions

predictions_inv = scaler.inverse_transform(dummy)[:,0]

# ---------------------------------------------------
# GENERATE FUTURE DATES
# ---------------------------------------------------

future_dates = pd.date_range(
    start=last_date,
    periods=future_steps + 1,
    freq="h"
)[1:]

forecast_df = pd.DataFrame({
    "Datetime": future_dates,
    "Forecast_MW": predictions_inv
})

forecast_df.set_index("Datetime", inplace=True)



# ---------------------------------------------------
# FORECAST TABLE
# ---------------------------------------------------

st.subheader("📋 Forecast Data")

st.dataframe(forecast_df.head(100))

# ---------------------------------------------------
# DOWNLOAD FORECAST
# ---------------------------------------------------

csv = forecast_df.to_csv()

st.download_button(
    "Download Forecast CSV",
    csv,
    "energy_forecast.csv",
    "text/csv"
)


# ---------------------------------------------------
# FORECAST GRAPH
# ---------------------------------------------------

st.subheader(f" Energy Forecast for Next {days} Days")

fig, ax = plt.subplots(figsize=(12,5))

ax.plot(
    forecast_df.index,
    forecast_df["Forecast_MW"],
)

ax.set_xlabel("Date")
ax.set_ylabel("Energy Demand (MW)")
ax.set_title("Energy Demand Forecast")

st.pyplot(fig)

# ---------------------------------------------------
# MODEL METRICS
# ---------------------------------------------------

st.subheader(" Model Performance")

col1, col2 = st.columns(2)

col1.metric("RMSE", f"{metrics['RMSE']:.2f}")
col2.metric("MAE", f"{metrics['MAE']:.2f}")


# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.write("---")
st.write("Built with an LSTM neural network to learn long-term temporal dependencies in energy consumption data.")