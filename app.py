import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import statsmodels.api as sm
from scipy.stats import zscore
import requests
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="Elevator AI | Global Enterprise Suite",
    page_icon="🛗",
    layout="wide",
    initial_sidebar_state="expanded"
)

def inject_styles():
    st.markdown("""
        <style>
        .main { background-color: #f8f9fa; }
        .stTabs [data-baseweb="tab-list"] { 
            gap: 12px; background-color: #0e1117; padding: 15px; border-radius: 15px 15px 0 0;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #262730; border-radius: 8px; padding: 10px 20px;
            color: #ffffff !important; border: 1px solid #3d3f4b;
        }
        .stTabs [aria-selected="true"] {
            background-color: #ffffff !important; color: #ff4b4b !important; font-weight: bold;
        }
        .health-track { width: 100%; background: #e9ecef; border-radius: 20px; height: 35px; margin: 15px 0; overflow: hidden; }
        .health-fill { height: 100%; border-radius: 20px; display: flex; align-items: center; justify-content: center; color: white; font-weight: 600; transition: width 1.5s ease-in-out; }
        .main-header {
            background: linear-gradient(135deg, #0e1117 0%, #1c1e26 100%);
            padding: 35px; border-radius: 24px;
            color: white; margin-bottom: 30px; display: flex; align-items: center; gap: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        .mascot-icon {
            font-size: 70px; background: rgba(255,255,255,0.08);
            padding: 20px; border-radius: 50%; line-height: 1;
        }
        .header-text h1 { margin: 0; color: white !important; font-size: 32px; letter-spacing: -1px; }
        .header-text p { margin: 8px 0 0 0; opacity: 0.7; font-size: 18px; font-weight: 300; }
        </style>
    """, unsafe_allow_html=True)

class DiagnosticEngine:
    def __init__(self, dataframe):
        self.df = dataframe.copy()
        
    def engineer_features(self, window_size=15):
        self.df['vib_rolling_mean'] = self.df['vibration'].rolling(window=window_size, min_periods=1).mean()
        self.df['vib_rolling_std'] = self.df['vibration'].rolling(window=window_size, min_periods=1).std().fillna(0)
        self.df['rev_rolling_mean'] = self.df['revolutions'].rolling(window=window_size, min_periods=1).mean()
        self.df['z_score_vib'] = zscore(self.df['vibration'])
        self.df['vibration_velocity'] = self.df['vibration'].diff().fillna(0)
        self.df['energy_index'] = (self.df['revolutions'] * self.df['vibration']) / 100
        return self.df
        
    def detect_iqr_anomalies(self, column):
        Q1, Q3 = self.df[column].quantile(0.25), self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        return self.df[(self.df[column] < (Q1 - 1.5 * IQR)) | (self.df[column] > (Q3 + 1.5 * IQR))]
        
    def calculate_health_metrics(self, v_limit, h_limit):
        v_penalty = (self.df['vibration'] > v_limit).mean() * 50
        h_penalty = (self.df['humidity'] > h_limit).mean() * 20
        r_penalty = (self.df['revolutions'] > self.df['revolutions'].quantile(0.85)).mean() * 30
        risk = min(100.0, v_penalty + h_penalty + r_penalty)
        return round(risk, 2), round(100.0 - risk, 2)

    def generate_statistical_summary(self):
        cols = ['vibration', 'revolutions', 'humidity', 'x1', 'x2', 'x3', 'x4', 'x5']
        stats = self.df[cols].describe().T
        stats['skewness'] = self.df[cols].skew()
        stats['kurtosis'] = self.df[cols].kurtosis()
        return stats

class PredictiveML:
    def __init__(self, df):
        self.df = df
        self.model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
        
    def train(self):
        features = ['revolutions', 'humidity', 'x1', 'x2', 'x3', 'x4', 'x5']
        X = self.df[features]
        y = 100 - (self.df['vibration'] * 7.5 + self.df['humidity'] * 0.1)
        self.model.fit(X, y)
        
    def predict(self, current_data):
        return np.clip(self.model.predict(current_data), 0, 100)

class MonteCarlo:
    def __init__(self, start_val, drift, vol):
        self.start, self.drift, self.vol = start_val, drift, vol
        
    def simulate(self, steps, count):
        res = np.zeros((steps, count))
        res[0] = self.start
        for t in range(1, steps):
            res[t] = res[t-1] * np.exp(self.drift + self.vol * np.random.normal(0, 1, count))
        return pd.DataFrame(res)

@st.cache_data
def ingest_data(path):
    try:
        df = pd.read_csv(path)
        return df.apply(pd.to_numeric, errors='coerce').dropna().drop_duplicates().sort_values('ID')
    except: 
        return None

def render_mascot_hero(health, mascot_choice):
    mascot_map = {
        "robo-muscles": "🦾", "Drone-Eye": "🛸", "RoboTech": "🤖", 
        "CyberOwl": "🦉", "Titan-X": "⛓️", "Sparky": "⚡", 
        "AeroVibe": "🚁", "DeepCore": "🌋", "Orbit": "🛰️", "BioSynth": "🧬"
    }
    icon = mascot_map.get(mascot_choice, "🦾")
    if health >= 85: 
        status, note, color = "OPTIMAL", "Mechanical integrity is peak. All sensors nominal.", "#2e7d32"
    elif health >= 60: 
        status, note, color = "STABLE", "Operational flow is consistent. Minor friction detected.", "#f9a825"
    elif health >= 40: 
        status, note, color = "WARNING", "Predictive models suggest upcoming maintenance requirements.", "#ef6c00"
    else: 
        status, note, color = "CRITICAL", "Emergency state detected. Immediate component replacement advised.", "#c62828"
    
    st.markdown(f"""
    <div class="main-header" style="border-left: 12px solid {color};">
        <div class="mascot-icon">{icon}</div>
        <div class="header-text">
            <h1>{mascot_choice} Interface | <span style="color:{color};">{status}</span></h1>
            <p>{note}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

inject_styles()
df_raw = ingest_data("Elevator predictive-maintenance-dataset.csv")

if df_raw is not None:
    with st.sidebar:
        st.header("👤 Persona Selector")
        m_list = ["robo-muscles", "Drone-Eye", "RoboTech", "CyberOwl", "Titan-X", "Sparky", "AeroVibe", "DeepCore", "Orbit", "BioSynth"]
        selected_m = st.selectbox("Choose AI Assistant", m_list)
        st.divider()
        st.header("🎛️ Engineering Hub")
        obs_range = st.slider("Data Window", 0, len(df_raw), (0, min(1200, len(df_raw))))
        v_limit = st.number_input("Vib Threshold", 0.5, 15.0, 5.0)
        h_limit = st.slider("Humid Limit", 10, 100, 80)
        smooth_f = st.number_input("Roll Window", 5, 50, 15)

    engine = DiagnosticEngine(df_raw.iloc[obs_range[0]:obs_range[1]])
    df_p = engine.engineer_features(window_size=smooth_f)
    anom = engine.detect_iqr_anomalies('vibration')
    risk, health = engine.calculate_health_metrics(v_limit, h_limit)

    st.title("🛗 Smart Elevator Movement")
    st.markdown("Real-time telemetry and predictive maintenance tracking.")

    render_mascot_hero(health, selected_m)
    
    h_color = "#1b5e20" if health >= 85 else "#fb8c00" if health >= 50 else "#b71c1c"
    st.markdown(f'<div class="health-track"><div class="health-fill" style="width:{health}%; background:{h_color};">CORE DURABILITY: {health}%</div></div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Vib RMS", f"{df_p['vibration'].mean():.3f}", delta=f"{df_p['vibration'].max():.1f}")
    c2.metric("Mean RPM", f"{df_p['revolutions'].mean():.0f}")
    c3.metric("Rel Humid", f"{df_p['humidity'].mean():.1f}%")
    c4.metric("Anom Events", f"{len(anom)}")
    c5.metric("Energy Index", f"{df_p['energy_index'].mean():.2f}")

    t1, t2, t3, t4, t5 = st.tabs(["📊 Telemetry", "🧬 Analytics", "🔮 ML Forecast", "📋 Tasks", "💾 Data Vault"])

    with t1:
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06, subplot_titles=("Vibration Spectrum", "Engine Velocity", "Environmental Humidity"))
        fig.add_trace(go.Scatter(x=df_p['ID'], y=df_p['vibration'], name="Vibration", line=dict(color="#d32f2f", width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_p['ID'], y=df_p['revolutions'], name="RPM", fill='tozeroy', line=dict(color="#2e7d32", width=1.5)), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_p['ID'], y=df_p['humidity'], name="Humidity", line=dict(color="#0288d1", width=1.5)), row=3, col=1)
        fig.update_layout(height=800, template="plotly_dark", hovermode="x unified", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        st.markdown("### Scatter Plot Analysis")
        st.caption("Analyzing the direct correlation between Engine Velocity (RPM) and Vibration, colored by Environmental Humidity.")
        
        fig_scatter = px.scatter(
            df_p, 
            x="revolutions", 
            y="vibration", 
            color="humidity", 
            title="Vibration vs. RPM",
            labels={"revolutions": "Engine Velocity (RPM)", "vibration": "Vibration (RMS)", "humidity": "Humidity %"},
            template="plotly_dark",
            color_continuous_scale="inferno",
            opacity=0.7
        )
        fig_scatter.update_layout(height=500, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_scatter, use_container_width=True)

    with t2:
        col_a, col_b = st.columns([7, 3])
        with col_a:
            corr_m = df_p[['vibration', 'revolutions', 'humidity', 'x1', 'x2', 'x3', 'x4', 'x5']].corr()
            st.plotly_chart(px.imshow(corr_m, text_auto=".2f", color_continuous_scale='Viridis', title="Correlation Heatmap Analysis of Elevator Data"), use_container_width=True)
            
            st.markdown("### Histogram of Humidity Distribution")
            fig_hist = px.histogram(
                df_p, 
                x="humidity", 
                nbins=30, 
                title="Histogram of Humidity Distribution",
                labels={"humidity": "Relative Humidity (%)"},
                template="plotly_dark",
                color_discrete_sequence=["#0288d1"],
                marginal="box"
            )
            fig_hist.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_hist, use_container_width=True)

            st.markdown("### Visualization of Revolutions Distribution")
            fig_rev_hist = px.histogram(
                df_p, 
                x="revolutions", 
                nbins=30, 
                title="Visualization of Revolutions Distribution",
                labels={"revolutions": "Engine Velocity (RPM)"},
                template="plotly_dark",
                color_discrete_sequence=["#2e7d32"],
                marginal="box"
            )
            fig_rev_hist.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_rev_hist, use_container_width=True)
            
            st.markdown("### Box Plot Analysis of Sensor Readings")
            st.caption("Comparing the spread and outliers across primary telemetry sensors.")
            df_melted = df_p.melt(value_vars=['vibration', 'revolutions', 'humidity'], 
                                  var_name='Sensor', value_name='Reading')
            
            fig_box = px.box(
                df_melted, 
                x="Sensor", 
                y="Reading", 
                color="Sensor",
                title="Sensor Readings Distribution Profile",
                template="plotly_dark",
                color_discrete_map={
                    "vibration": "#d32f2f", 
                    "revolutions": "#2e7d32", 
                    "humidity": "#0288d1"
                }
            )
            fig_box.update_layout(height=450, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)
            
        with col_b:
            st.markdown("### Regression Logic")
            X_ols = sm.add_constant(df_p[['revolutions', 'humidity', 'x1']])
            res_ols = sm.OLS(df_p['vibration'], X_ols).fit()
            st.metric("R-Squared", f"{res_ols.rsquared:.4f}")
            st.dataframe(res_ols.params)
            
        st.dataframe(engine.generate_statistical_summary().style.format("{:.2f}").background_gradient(cmap="YlOrRd"), use_container_width=True)

    with t3:
        st.subheader("Random Forest Predictive Analytics")
        ml_model = PredictiveML(df_p)
        ml_model.train()
        pred_h = ml_model.predict(df_p[['revolutions', 'humidity', 'x1', 'x2', 'x3', 'x4', 'x5']].iloc[-1:].values)
        st.metric("Projected Integrity", f"{pred_h[0]:.2f}%")
        
        sim_e = MonteCarlo(df_p['vibration'].iloc[-1], df_p['vibration'].pct_change().mean(), df_p['vibration'].pct_change().std())
        sim_d = sim_e.simulate(60, 150)
        
        fig_mc = go.Figure()
        
        # Limiting to 50 paths to improve Streamlit rendering performance
        for col in sim_d.columns[:50]: 
            fig_mc.add_trace(go.Scatter(
                y=sim_d[col], 
                mode='lines', 
                line=dict(width=0.4, color='rgba(200, 200, 200, 0.2)'),
                showlegend=False
            ))
            
        fig_mc.add_trace(go.Scatter(
            y=sim_d.mean(axis=1), 
            mode='lines', 
            name='Projected Mean', 
            line=dict(color='#ff4b4b', width=4)
        ))
        
        fig_mc.update_layout(title="60-Cycle Stochastic Projection", template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_mc, use_container_width=True)

    with t4:
        st.header("📋 Maintenance Ticketing")
        with st.form("maint_f"):
            asset = st.selectbox("Critical Asset", ["Traction Motor", "Brake Assembly", "Guide Rails", "Door Interlock", "Main Cable"])
            note = st.text_area("Observation Log")
            if st.form_submit_button("Log Incident"):
                st.success(f"Work Order registered by {selected_m}.")
                
        kb_d = pd.DataFrame({
            "Hex_Code": ["0x01", "0x02", "0x03", "0x04", "0x05"],
            "Fault_Type": ["Resonance", "Thermal", "Friction", "Logic", "Sensor"],
            "Protocol": ["Dampen", "Ventilate", "Lubricate", "Reset", "Calibrate"]
        })
        st.table(kb_d)

    with t5:
        st.header("💾 Data Vault & Export")
        st.info("Export the full processed dataset including engineered features (Vibration Velocity, Energy Index, etc.)")
        csv_data = df_p.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download elevator_dataset.csv",
            data=csv_data,
            file_name=f"elevator_telemetry_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.divider()
        st.dataframe(df_p.head(100), use_container_width=True)
        
    st.divider()

else:
    st.error("Dataset not found. Please ensure 'Elevator predictive-maintenance-dataset.csv' is in the project folder.")

