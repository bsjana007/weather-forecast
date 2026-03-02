# 📍 Weather Forecast Model

A Python-based weather forecasting project that predicts future weather conditions for **Kolkata, India** using Machine Learning.

This project uses historical weather data (2017–2022) and applies Random Forest Regression models to generate a 5-day weather forecast.

---

## 📖 Overview

The model learns from historical weather patterns and predicts:

- 🌡 Minimum Temperature
- 🌡 Maximum Temperature
- 💧 Humidity
- 🌧 Rainfall

The forecast is generated iteratively using trained regression models.

---

## ✨ Features

- Loads and cleans historical CSV weather data
- Aggregates daily weather metrics
- Encodes categorical weather conditions
- Uses feature scaling for better predictions
- Trains multiple Random Forest models
- Generates a 5-day future forecast
- Displays predictions in a structured tabular format

---

## 📂 Repository Structure

```bash
weather-forecast/
│
├── weather.py
├── Kolkata_weather_data(2017-2022).csv
├── README.md
└── requirements.txt (recommended)
```

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/bsjana007/weather-forecast.git
cd weather-forecast
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### Usage

Run the script:

```bash
python weather.py
```

The program will:

1.Load historical weather data
2.Train machine learning models
3.Predict the next 5 days
4.Print the forecast in the console

---

## 🧠 How It Works

### 1️⃣ Data Loading & Cleaning

- Reads CSV file

- Converts date column to datetime format

- Removes missing values

- Aggregates data by date

### 2️⃣ Feature Engineering

- Encodes weather conditions

- Extracts time-based features (day, month, etc.)

- Scales input data

### 3️⃣ Model Training

Four separate Random Forest Regressors are trained to predict:

- Minimum temperature

- Maximum temperature

- Humidity

- Rainfall

### Forecast Generation

The model:

- Starts from the last available date

- Predicts one day at a time

- Uses previous predictions to forecast the next day

- Repeats this for 5 days

---

## 📊 Example Output

```bash
FINAL 5-DAY WEATHER FORECAST

Date        Min Temp (°C)  Max Temp (°C)  Humidity (%)  Rain (mm)
2026-03-02      25.4           33.1           78.2         0.0
2026-03-03      26.1           32.8           79.0         0.2
2026-03-04      25.9           33.4           77.5         0.0
2026-03-05      26.3           34.0           76.8         0.1
2026-03-06      25.8           33.2           78.9         0.0
```

(Actual results depend on the dataset and model training.)

---

## 🙌 Author

### Bhabani Sankar Jana
