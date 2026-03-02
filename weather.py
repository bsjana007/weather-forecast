import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

#  LOAD & CLEAN DATA 
df = pd.read_csv("Kolkata_weather_data(2017-2022).csv")
df.columns = df.columns.str.strip()

# Ensure "Date time" column exists
if "Date time" not in df.columns:
    raise ValueError(" 'Date time' column missing!")

df["Date time"] = pd.to_datetime(df["Date time"], errors="coerce")
df = df.dropna(subset=["Date time"]).sort_values("Date time")

# Rename major weather columns for consistency
rename_map = {
    "Minimum Temperature": "min_temp",
    "Maximum Temperature": "max_temp",
    "Temperature": "avg_temp",
    "Dew Point": "dew_point",
    "Relative Humidity": "humidity",
    "Wind Speed": "wind_speed",
    "Visibility": "visibility",
    "Cloud Cover": "cloud_cover",
    "Sea Level Pressure": "pressure",
    "Precipitation": "rain",
    "Conditions": "conditions"
}
df = df.rename(columns=rename_map)

# Encode "conditions" text column
if "conditions" in df.columns:
    df["conditions"] = df["conditions"].fillna("Unknown").astype(str)
    le = LabelEncoder()
    df["conds_encoded"] = le.fit_transform(df["conditions"])
else:
    df["conds_encoded"] = 0

#  DAILY AGGREGATION 
df["date"] = df["Date time"].dt.date
daily = df.groupby("date").agg({
    "min_temp": "mean",
    "max_temp": "mean",
    "avg_temp": "mean",
    "humidity": "mean",
    "dew_point": "mean",
    "wind_speed": "mean",
    "visibility": "mean",
    "cloud_cover": "mean",
    "pressure": "mean",
    "rain": "sum",
    "conds_encoded": "mean"
}).reset_index()

daily = daily.ffill().bfill()

#  FEATURES & TARGETS 
features = [
    "avg_temp", "humidity", "dew_point", "pressure",
    "wind_speed", "visibility", "cloud_cover", "conds_encoded"
]

# YOU WANT TO PREDICT: Min Temp, Max Temp, Humidity, Rain
target_cols = ["min_temp", "max_temp", "humidity", "rain"]

X = daily[features]
y = daily[target_cols]

#  SCALING FEATURES
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#  TRAIN RF MODELS 
rf_models = {}
for col in target_cols:
    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_scaled, y[col])
    rf_models[col] = model

#  FORECAST NEXT 5 DAYS 
today = datetime.today().date()
future_dates = [today + timedelta(days=i+1) for i in range(5)]

last_row = daily.iloc[-1].copy()
future_preds = []

for _ in range(5):
    scaled_input = scaler.transform(last_row[features].values.reshape(1, -1))

    min_temp = rf_models["min_temp"].predict(scaled_input)[0]
    max_temp = rf_models["max_temp"].predict(scaled_input)[0]
    humidity = rf_models["humidity"].predict(scaled_input)[0]
    rain = max(0, rf_models["rain"].predict(scaled_input)[0])

    future_preds.append([min_temp, max_temp, humidity, rain])

    # simulate daily random changes
    last_row["avg_temp"] = (min_temp + max_temp) / 2
    last_row["humidity"] = np.clip(humidity + np.random.uniform(-1, 1), 40, 100)
    last_row["wind_speed"] += np.random.uniform(-0.3, 0.3)
    last_row["cloud_cover"] = np.clip(last_row["cloud_cover"] + np.random.uniform(-3, 3), 0, 100)
    last_row["pressure"] += np.random.uniform(-0.5, 0.5)

# BUILD OUTPUT 
forecast = pd.DataFrame(
    future_preds,
    columns=["Min Temp (°C)", "Max Temp (°C)", "Humidity (%)", "Rain (mm)"]
)
forecast.insert(0, "Date", future_dates)

# Round to 1 decimal
forecast = forecast.round(1)

#  FINAL OUTPUT 
print("\nFINAL 5-DAY WEATHER FORECAST\n")
print(forecast.to_string(index=False))
