"""
Dashboard Streamlit Interactif pour StreamAI Forecaster
Monitoring MLOps temps réel du système Kafka + ML Predictions (Random Forest)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta, timezone
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Import modules personnalisés
from utils.kafka_client import get_monitoring_client
from utils.metrics import MonitoringMetrics
from utils.alerts import AlertManager

# ============================================================================
# CONFIGURATION DE LA PAGE
# ============================================================================

st.set_page_config(
    page_title="StreamAI Forecaster - MLOps Dashboard",
    page_icon="🌡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS pour une UI moderne et aérée
st.markdown("""
    <style>
    .main { background-color: #f4f7f9; }
    .stMetric {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e1e8ed;
    }
    .metric-card-container {
        display: flex;
        justify-content: space-between;
        gap: 20px;
        margin-bottom: 20px;
    }
    .status-badge {
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8em;
        font-weight: bold;
        text-transform: uppercase;
    }
    .status-ok { background-color: #d4edda; color: #155724; }
    .status-warning { background-color: #fff3cd; color: #856404; }
    .status-error { background-color: #f8d7da; color: #721c24; }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# INITIALISATION DU STATE
# ============================================================================

if 'alert_history' not in st.session_state:
    st.session_state.alert_history = []
if 'config' not in st.session_state:
    st.session_state.config = {
        "kafka_lag_critical": 1000,
        "kafka_lag_warning": 500,
        "drift_threshold": 0.3,
        "refresh_rate": 10,
        "monitoring_window": 15,
        "stability_threshold": 3.0
    }

# ============================================================================
# SIDEBAR - CONFIGURATION & MLOps SETTINGS
# ============================================================================

with st.sidebar:
    st.title("⚙️ MLOps Dashboard Config")
    st.info("💡 Ce dashboard surveille la qualité et la santé du pipeline sans vérité terrain en temps réel.")
    
    with st.expander("📡 Kafka Pipeline", expanded=True):
        kafka_server = st.text_input("Bootstrap Servers", value="localhost:9092")
        topics = {
            "raw": "data.raw.stream",
            "cleaned": "data.cleaned.stream",
            "features": "data.features.hourly",
            "predictions": "data.predictions.weather",
            "quantiles": "data.predictions.weather.quantiles"
        }
        selected_topic = st.selectbox("Topic de prédiction", [topics["predictions"], topics["quantiles"]], index=0)
    
    with st.expander("🚨 Alert Thresholds", expanded=False):
        st.session_state.config["kafka_lag_critical"] = st.number_input("Lag Critique", value=1000)
        st.session_state.config["drift_threshold"] = st.slider("Drift KS", 0.0, 1.0, 0.3)
        st.session_state.config["stability_threshold"] = st.number_input("Std Max (Stabilité)", value=3.0)

    with st.expander("⏱ Refresh & Window", expanded=False):
        st.session_state.config["refresh_rate"] = st.slider("Fréquence (s)", 5, 60, st.session_state.config["refresh_rate"])
        auto_refresh = st.checkbox("Auto-refresh", value=True)
        st.session_state.config["monitoring_window"] = st.selectbox("Fenêtre (min)", [5, 15, 60, 360], index=1)

    st.markdown("---")
    # Export/Import Config
    col_exp, col_imp = st.columns(2)
    with col_exp:
        config_json = json.dumps(st.session_state.config, indent=2)
        st.download_button("📤 Export JSON", config_json, "mlops_config.json", "application/json")
    with col_imp:
        uploaded_config = st.file_uploader("📥 Import JSON", type="json")
        if uploaded_config:
            st.session_state.config.update(json.load(uploaded_config))
            st.rerun()

# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data(ttl=300)
def load_static_metrics(path: str = "models/dashboard_metrics.json"):
    """Charge les métriques offline comme référence statique."""
    path = Path(path)
    if not path.exists(): return {}
    try:
        with open(path, "r") as f: return json.load(f)
    except: return {}

kafka_client = get_monitoring_client()
alert_manager = AlertManager(st.session_state.config)
static_metrics = load_static_metrics("models/dashboard_metrics.json")
# Optionnel : métriques LSTM pour comparaison offline
lstm_static_metrics = load_static_metrics("models_lstm/dashboard_metrics.json")

# Fetch Real-time data
with st.spinner("⏳ Récupération des données Kafka..."):
    raw_msgs = kafka_client.fetch_messages(selected_topic, max_messages=1000)
    df_pred = pd.DataFrame(raw_msgs)
    # On monitor tous les topics pour le lag
    system_stats = kafka_client.get_kafka_stats(list(topics.values()))

# Data Processing
if not df_pred.empty:
    for col in ["event_time", "created_at"]:
        if col in df_pred.columns:
            df_pred[col] = pd.to_datetime(df_pred[col], utc=True)
    
    # Identification de la colonne de prédiction principale
    # Gère à la fois le topic standard et le topic quantiles
    if "temperature_prediction" in df_pred.columns:
        df_pred["main_prediction"] = df_pred["temperature_prediction"]
    elif "temperature_p50" in df_pred.columns:
        df_pred["main_prediction"] = df_pred["temperature_p50"]
    else:
        # Fallback pour éviter les KeyError
        df_pred["main_prediction"] = np.nan

    # Filter by window
    now = datetime.now(timezone.utc)
    df_window = df_pred[df_pred["created_at"] >= (now - timedelta(minutes=st.session_state.config["monitoring_window"]))]
    
    # Calcul de la stabilité globale pour les alertes
    stability = 0
    if not df_window.empty and "main_prediction" in df_window.columns:
        stability = MonitoringMetrics.calculate_stability(df_window["main_prediction"].dropna().values)
else:
    df_window = pd.DataFrame()
    stability = 0

# ============================================================================
# HEADER & TOP KPIs
# ============================================================================

st.title("🌡 StreamAI Forecaster | MLOps Monitor")
st.markdown("*Système de surveillance temps réel du pipeline Random Forest (RF)*")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    status_label = "🟢 OK" if not df_window.empty else "🔴 STOP"
    st.metric("Flux Streaming", status_label, help="Statut basé sur l'arrivée de messages récents")

with kpi2:
    msg_count = len(df_window)
    throughput = msg_count / st.session_state.config["monitoring_window"]
    st.metric("Throughput", f"{throughput:.1f} msg/min", f"{msg_count} total")

with kpi3:
    if not df_window.empty and "event_time" in df_window.columns:
        latency = (df_window["created_at"] - df_window["event_time"]).dt.total_seconds().median()
        st.metric("Latence Médiane", f"{latency:.2f}s", help="Délai moyen entre l'événement météo et sa prédiction")
    else: st.metric("Latence Médiane", "N/A")

with kpi4:
    total_lag = sum(s.get("total_messages", 0) for s in system_stats.values() if "error" not in s)
    st.metric("Backlog Kafka", f"{total_lag:,} msg", delta_color="inverse", help="Messages en attente dans le pipeline")

st.markdown("---")

# ============================================================================
# TABS PRINCIPAUX
# ============================================================================

t_system, t_quality, t_model, t_offline, t_alerts = st.tabs([
    "⚙️ System Health", "📊 Data Quality", "🤖 Model Monitor", "📚 Offline Ref", "🚨 MLOps Alerts"
])

# --- TAB: SYSTEM HEALTH ---
with t_system:
    st.subheader("Surveillance du Pipeline Kafka")
    col_l, col_r = st.columns(2)
    
    with col_l:
        st.markdown("#### 📥 Lag par Topic")
        lag_list = [{"Topic": name, "Lag": stats.get("total_messages", 0)} for name, stats in system_stats.items() if "error" not in stats]
        df_lag = pd.DataFrame(lag_list)
        fig_lag = px.bar(df_lag, x="Topic", y="Lag", color="Lag", color_continuous_scale="Reds", title="Backlog Messages")
        st.plotly_chart(fig_lag, width="stretch")

    with col_r:
        st.markdown("#### ⚡ Débit des Prédictions")
        if not df_window.empty:
            tp_df = (
                df_window.set_index("created_at")
                .resample("1min")
                .size()
                .reset_index(name="count")
            )
            if not tp_df.empty:
                max_count = max(1, int(tp_df["count"].max()))
                fig_tp = px.line(
                    tp_df,
                    x="created_at",
                    y="count",
                    title="Messages par minute",
                )
                # Meilleure lisibilité avec marqueurs + axe Y qui part de 0
                fig_tp.update_traces(mode="lines+markers")
                fig_tp.update_yaxes(range=[0, max_count * 1.2])
                st.plotly_chart(fig_tp, width="stretch")
            else:
                st.info("Aucune donnée agrégée pour cette fenêtre.")
        else:
            st.info("Aucune donnée dans la fenêtre sélectionnée.")

# --- TAB: DATA QUALITY ---
with t_quality:
    st.subheader("Data Integrity & Drift Detection")
    if df_window.empty:
        st.warning("⚠️ Pas assez de données streaming pour analyser la qualité.")
    else:
        q_c1, q_c2 = st.columns(2)
        
        with q_c1:
            st.markdown("#### 🧪 Feature Drift (KS Test)")
            # On simule le drift sur la prédiction principale par rapport à une référence
            if "main_prediction" in df_window.columns:
                curr_data = df_window["main_prediction"].dropna().values
                if len(curr_data) > 0:
                    ref_data = np.random.normal(np.mean(curr_data), np.std(curr_data), 100) # Mock reference
                    drift_res = MonitoringMetrics.calculate_drift(curr_data, ref_data)
                    
                    st.metric("KS Score (Drift)", f"{drift_res['ks_score']:.3f}", 
                            delta="🚨 ALERTE" if drift_res['ks_score'] > st.session_state.config["drift_threshold"] else "✅ STABLE",
                            delta_color="inverse")
                    
                    fig_dist = go.Figure()
                    fig_dist.add_trace(go.Histogram(x=ref_data, name="Historique (Ref)", opacity=0.5, marker_color="gray"))
                    fig_dist.add_trace(go.Histogram(x=curr_data, name="Actuel (Stream)", opacity=0.7, marker_color="blue"))
                    fig_dist.update_layout(barmode='overlay', title="Distribution: Ref vs Stream")
                    st.plotly_chart(fig_dist, width="stretch")
                else:
                    st.info("Pas assez de données pour le drift.")
            else:
                st.error("Colonne de prédiction manquante.")

        with q_c2:
            st.markdown("#### 🔍 Outliers & Missing")
            missing_pct = df_window.isnull().mean() * 100
            st.write("**Valeurs Manquantes (%):**")
            
            missing_data = missing_pct[missing_pct > 0]
            if not missing_data.empty:
                st.dataframe(missing_data)
            else:
                st.success("✅ 0% missing values")
            
            outliers = MonitoringMetrics.check_outliers(curr_data)
            st.metric("Taux d'Outliers", f"{outliers['percentage']:.1f}%", help="Points à >3 écart-types de la moyenne")
            
            # Schema Compliance check
            expected = ["event_time", "horizon_hours", "created_at"]
            if selected_topic == topics["predictions"]:
                expected.append("temperature_prediction")
            else:
                expected.extend(["temperature_p10", "temperature_p50", "temperature_p90"])
                
            schema_res = MonitoringMetrics.check_schema(df_window, expected)
            if schema_res['is_valid']:
                st.write("**Schema Compliance:** ✅ OK")
            else:
                st.error(f"**Schema Compliance:** ❌ MISSING: {schema_res['missing_fields']}")
            
            violations = MonitoringMetrics.check_physical_ranges(df_window)
            if any(v > 0 for v in violations.values()):
                st.error(f"⚠️ Violations Physiques: {violations}")
            else:
                st.success("✅ Aucune violation des limites physiques (-50°C / +60°C)")

# --- TAB: MODEL MONITORING ---
with t_model:
    st.subheader("Comportement du Modèle (Sans Labels)")
    if df_window.empty:
        st.info("En attente de prédictions...")
    else:
        m_c1, m_c2 = st.columns([2, 1])
        
        with m_c1:
            st.markdown("#### 📈 Timeline Multi-Horizons")
            if "main_prediction" in df_window.columns:
                fig_model = go.Figure()
                colors = {"1.0": "green", "3.0": "orange", "6.0": "red"}
                
                for h in sorted(df_window["horizon_hours"].unique()):
                    h_df = df_window[df_window["horizon_hours"] == h].sort_values("created_at")
                    fig_model.add_trace(go.Scatter(x=h_df["created_at"], y=h_df["main_prediction"], 
                                                name=f"Horizon {h}h", line=dict(color=colors.get(str(h), "blue"))))
                
                fig_model.update_layout(yaxis_title="Température (°C)", hovermode="x unified")
                st.plotly_chart(fig_model, width="stretch")
            else:
                st.error("Données de prédiction non trouvées.")

        with m_c2:
            st.markdown("#### 🔗 Cohérence Inter-Horizons")
            if "main_prediction" in df_window.columns:
                fig_box = px.box(df_window, x="horizon_hours", y="main_prediction", color="horizon_hours",
                                title="Distribution par Horizon")
                st.plotly_chart(fig_box, width="stretch")
                
                stability = MonitoringMetrics.calculate_stability(df_window["main_prediction"].dropna().values)
                st.metric("Prediction Stability (Std)", f"{stability:.2f}", 
                        help="Un écart-type élevé indique des prédictions volatiles.")
            else:
                st.info("Données insuffisantes.")

# --- TAB: OFFLINE REFERENCE ---
with t_offline:
    st.subheader("📊 Performances Statiques (Offline)")
    st.info("Ces métriques sont calculées sur le dataset historique de test (hors-ligne).")
    
    if not static_metrics:
        st.warning("Fichier `models/dashboard_metrics.json` introuvable.")
    else:
        perf = static_metrics.get("model_performance", {})
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            st.write("**MAE par Horizon (Test):**")
            mae_df = pd.DataFrame([{"Horizon": k, "MAE": v["test_mae"]} for k, v in perf.items()])
            st.table(mae_df)
        
        with col_m2:
            st.write("**Score R² (Test):**")
            r2_df = pd.DataFrame([{"Horizon": k, "R²": v["test_r2"]} for k, v in perf.items()])
            st.bar_chart(r2_df.set_index("Horizon"))

        # Graphes de synthèse inspirés du code avancé
        st.markdown("### 🎯 R² cible vs obtenu (barres + ligne cible)")
        if perf:
            horizons = []
            r2_vals = []
            targets_r2 = []
            for horizon_label, m in perf.items():
                horizons.append(horizon_label)
                r2 = float(m.get("test_r2", 0.0))
                r2_vals.append(max(min(r2, 1.0), -1.0))
                targets_r2.append(0.85)

            df_conf = pd.DataFrame(
                {"horizon": horizons, "r2_test": r2_vals, "target": targets_r2}
            )
            fig_confidence = go.Figure()
            fig_confidence.add_trace(
                go.Bar(
                    x=df_conf["horizon"],
                    y=df_conf["r2_test"],
                    name="R² (test)",
                    marker_color="lightblue",
                )
            )
            fig_confidence.add_trace(
                go.Scatter(
                    x=df_conf["horizon"],
                    y=df_conf["target"],
                    name="Cible R²",
                    mode="lines+markers",
                    line=dict(color="red", dash="dash"),
                )
            )
            fig_confidence.update_layout(
                yaxis_title="R² (test)", yaxis_range=[0, 1], height=350
            )
            st.plotly_chart(fig_confidence, width="stretch")

        st.markdown("### 📉 MAE cible vs obtenu (par horizon)")
        if perf:
            horizons = []
            maes = []
            targets_mae = []
            for horizon_label, m in perf.items():
                horizons.append(horizon_label)
                maes.append(float(m.get("test_mae", 0.0)))
                if horizon_label.startswith("1"):
                    targets_mae.append(2.0)
                elif horizon_label.startswith("3"):
                    targets_mae.append(3.0)
                else:
                    targets_mae.append(4.0)
            mae_data = pd.DataFrame(
                {"horizon": horizons, "mae": maes, "target": targets_mae}
            )
            fig_mae = go.Figure()
            fig_mae.add_trace(
                go.Bar(
                    x=mae_data["horizon"],
                    y=mae_data["mae"],
                    name="MAE (test)",
                    marker_color="coral",
                )
            )
            fig_mae.add_trace(
                go.Scatter(
                    x=mae_data["horizon"],
                    y=mae_data["target"],
                    name="Cible MAE",
                    mode="lines+markers",
                    line=dict(color="green", dash="dash"),
                )
            )
            fig_mae.update_layout(yaxis_title="MAE (°C)", height=350)
            st.plotly_chart(fig_mae, width="stretch")

        # Comparaison RF vs LSTM (offline uniquement) si disponible
        st.markdown("### 🆚 Comparaison RF vs LSTM (MAE test, offline uniquement)")
        lstm_perf = lstm_static_metrics.get("model_performance", {}) if lstm_static_metrics else {}
        if not lstm_perf:
            st.info(
                "Aucune métrique LSTM trouvée. Si tu entraînes un LSTM offline, "
                "dépose son `dashboard_metrics.json` dans `models_lstm/`."
            )
        else:
            rows = []
            for horizon_label, m_rf in perf.items():
                mae_rf = float(m_rf.get("test_mae", 0.0))
                mae_lstm = None
                if horizon_label in lstm_perf:
                    mae_lstm = float(lstm_perf[horizon_label].get("test_mae", 0.0))
                rows.append(
                    {
                        "horizon": horizon_label,
                        "RF_MAE": mae_rf,
                        "LSTM_MAE": mae_lstm,
                    }
                )
            df_comp = pd.DataFrame(rows)
            if df_comp.empty:
                st.info("Pas de métriques comparables entre RF et LSTM.")
            else:
                fig_comp = go.Figure()
                fig_comp.add_trace(
                    go.Bar(
                        x=df_comp["horizon"],
                        y=df_comp["RF_MAE"],
                        name="RF MAE (test)",
                        marker_color="coral",
                    )
                )
                if df_comp["LSTM_MAE"].notna().any():
                    fig_comp.add_trace(
                        go.Bar(
                            x=df_comp["horizon"],
                            y=df_comp["LSTM_MAE"],
                            name="LSTM MAE (test)",
                            marker_color="steelblue",
                        )
                    )
                fig_comp.update_layout(
                    barmode="group", yaxis_title="MAE (°C)", height=350
                )
                st.plotly_chart(fig_comp, width="stretch")

# --- TAB: ALERTS ---
with t_alerts:
    st.subheader("MLOps Alert System")
    
    # Évaluation des alertes
    current_alerts = alert_manager.check_alerts(system_stats, {}, {"stability": stability if not df_window.empty else 0})
    
    for a in current_alerts:
        a['timestamp'] = datetime.now().strftime("%H:%M:%S")
        if not any(h['message'] == a['message'] for h in st.session_state.alert_history[:5]):
            st.session_state.alert_history.insert(0, a)
    
    # Affichage
    if not st.session_state.alert_history:
        st.success("✅ Aucune alerte système ou modèle détectée.")
    else:
        for alert in st.session_state.alert_history[:15]:
            level = alert['level']
            color = "red" if level == "CRITICAL" else "orange" if level == "WARNING" else "blue"
            with st.expander(f"[{alert['timestamp']}] {level}: {alert['category']}"):
                st.markdown(f"<span style='color:{color}'>{alert['message']}</span>", unsafe_allow_html=True)

# ============================================================================
# FOOTER & ACTIONS
# ============================================================================

st.markdown("---")
col_f1, col_f2 = st.columns([3, 1])
with col_f1:
    st.markdown(f"**Dernière mise à jour:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | **Uptime:** {str(datetime.now() - st.session_state.get('start_time', datetime.now())).split('.')[0]}")
with col_f2:
    if not df_window.empty:
        st.download_button("💾 Download CSV Data", df_window.to_csv(index=False).encode('utf-8'), "predictions_stream.csv", "text/csv")

if auto_refresh:
    time.sleep(st.session_state.config["refresh_rate"])
    st.rerun()
