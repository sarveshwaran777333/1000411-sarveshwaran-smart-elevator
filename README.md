# 1000411-sarveshwaran-smart-elevator

# 🛗 Smart Elevator Movement — Predictive Maintenance Dashboard

## 📌 Project Overview
This project presents an AI-powered predictive maintenance dashboard for monitoring elevator health and performance. The Streamlit web application analyzes telemetry sensor data to detect anomalies, visualize operational trends, and forecast system integrity.

The dashboard helps engineers and building managers understand vibration patterns and optimize elevator performance to reduce downtime and maintenance costs.

## 🎯 Problem Statement
Elevators in high-rise buildings experience wear due to frequent use and environmental conditions. Monitoring vibration levels is critical because excessive vibration indicates mechanical stress or possible failure.

This application analyzes:

- Revolutions (door motor usage)  
- Humdity levels  
- Vibration (health indicator)  
- Sensor readings (x1–x5)  

to support predictive maintenance and smarter elevator movement strategies.

## ⚙️ Features

### 📊 Telemetry Monitoring
- Real-time vibration, revolutions, and humidity tracking  
- Multi-panel time-series visualization  
- Scatter analysis of RPM vs vibration  

### 🧬 Advanced Analytics
- Correlation heatmap of sensor relationships  
- Distribution histograms & box plots  
- Regression analysis and statistical summary  

### 🚨 Anomaly Detection
- IQR-based vibration anomaly detection  
- Rolling statistics & z-score analysis  
- Energy index & vibration velocity metrics  

### 🔮 Predictive AI Forecasting
- Random Forest model to predict elevator integrity  
- Monte Carlo simulation for vibration trends  
- Health risk scoring system  

### 📋 Maintenance Ticketing
- Log incidents & observations  
- Fault protocol reference table  

### 💾 Data Export
- Download processed telemetry dataset  
- Includes engineered features for further analysis  

## 🧠 How the System Works

### 1️⃣ Data Processing
- Cleans dataset  
- Converts values to numeric  
- Removes duplicates & missing data  

### 2️⃣ Feature Engineering
Creates advanced indicators:
- Rolling vibration mean & deviation  
- Vibration velocity  
- Energy index  
- Z-score normalization  

### 3️⃣ Health Risk Calculation
Risk score based on:
- Excess vibration  
- High humidity exposure  
- Extreme revolutions frequency  

### 4️⃣ Machine Learning Prediction
A Random Forest model estimates projected system integrity.

### 5️⃣ Simulation
Monte Carlo modeling forecasts vibration behavior over future cycles.

## 📈 Key Insights from Analysis

✔ High revolutions increase vibration, indicating mechanical wear  
✔ Excess humidity contributes to operational instability  
✔ Sudden vibration spikes signal potential component failure  
✔ Predictive monitoring can reduce downtime and maintenance costs  

## 🛠️ Technologies Used

- Python  
- Streamlit  
- Pandas & NumPy  
- Plotly  
- Statsmodels  
- Scikit-learn  
- SciPy  

## 🚀 Installation & Setup

### 1️⃣ Clone Repository

git clone [github](https://github.com/sarveshwaran777333/1000411-sarveshwaran-smart-elevator)
cd elevator-ai-dashboard
2️⃣ Install Requirements
pip install -r requirements.txt
3️⃣ Run the App
streamlit run app.py

# 🌐 Live Application

[live App](https://1000411-sarveshwaran-smart-elevator-9htfxahijlkmcaxxytdybv.streamlit.app/)

# storyboard & screenshots

[canva](https://www.canva.com/design/DAHB688Hq58/Cs2oMSXTeDJYMvLw3y6INg/edit?utm_content=DAHB688Hq58&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

# Learning Outcomes Achieved

Applied statistical reasoning to sensor data

Performed exploratory data analysis

Built interactive visualizations

Implemented machine learning prediction

Deployed a functional Streamlit web app

👤 Author

Student Name: Sarveshwaran.K
Course: Artificial Intelligence
