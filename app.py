import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

# ========================
# Load Data
# ========================
@st.cache_data
def load_data():
    df_sdg = pd.read_csv("df_sdg.csv")
    df_lookup = pd.read_csv("df_lookup.csv")
    return df_sdg, df_lookup

df_sdg, df_lookup = load_data()

# Validate df_lookup
expected_cols = ["code", "sdg", "group", "description"]
if not all(col in df_lookup.columns for col in expected_cols):   
    st.error(f"⚠️ df_lookup must have columns: {expected_cols}")
    st.stop()

# ========================
# Prediction Function
# ========================
def run_models(df_entity, target, year_range, selected_models, rf_params, arima_params):
    results = pd.DataFrame({"Year": year_range})
    metrics = {}

    df_clean = df_entity[["Year", target]].dropna()
    if df_clean.empty:
        return None, None

    X = df_clean[["Year"]]
    y = df_clean[target]

    # Linear Regression
    if "Linear Regression" in selected_models:
        try:
            lr = LinearRegression()
            lr.fit(X, y)
            pred_lr = lr.predict(results[["Year"]])
            results["Linear Regression"] = pred_lr
            metrics["Linear Regression"] = {
                "RMSE": float(np.sqrt(mean_squared_error(y, lr.predict(X)))),
                "R²": float(r2_score(y, lr.predict(X)))
            }
        except Exception as e:
            st.warning(f"Linear Regression failed: {e}")

    # Random Forest
    if "Random Forest" in selected_models:
        try:
            rf = RandomForestRegressor(
                n_estimators=rf_params["n_estimators"],
                max_depth=rf_params["max_depth"],
                random_state=42
            )
            rf.fit(X, y)
            pred_rf = rf.predict(results[["Year"]])
            results["Random Forest"] = pred_rf
            metrics["Random Forest"] = {
                "Params": rf_params,
                "RMSE": float(np.sqrt(mean_squared_error(y, rf.predict(X)))),
                "R²": float(r2_score(y, rf.predict(X)))
            }
        except Exception as e:
            st.warning(f"Random Forest failed: {e}")

    # ARIMA
    if "ARIMA" in selected_models:
        try:
            if len(df_clean) >= 5:  # need enough data points
                order = tuple(arima_params["order"])
                arima_model = ARIMA(df_clean[target], order=order)
                arima_fit = arima_model.fit()
                pred_arima = arima_fit.forecast(steps=len(year_range))
                results["ARIMA"] = pred_arima.values
                metrics["ARIMA"] = {
                    "Params": arima_params,
                    "AIC": float(arima_fit.aic),
                    "BIC": float(arima_fit.bic)
                }
            else:
                st.info(f"Not enough data for ARIMA on {target}")
        except Exception as e:
            st.warning(f"ARIMA failed: {e}")

    return results, metrics

# ========================
# Plot Function (Plotly)
# ========================
def plot_results(results, df_entity, target, title, key, selected_models, entity_type):
    fig = go.Figure()

    # --- Actual historical data (2000–2025) ---
    if target in df_entity.columns:
        df_clean = df_entity[["Year", target]].dropna()
        if not df_clean.empty:
            if entity_type == "Region":  # aggregate mean across region
                df_clean = df_clean.groupby("Year")[target].mean().reset_index()
            fig.add_trace(go.Scatter(
                x=df_clean["Year"], y=df_clean[target],
                mode="lines+markers",
                name="Actual (2000–2025)",
                line=dict(color="black", dash="dot"),
                marker=dict(size=6)
            ))

    # --- Predictions (2026–2030) ---
    if results is not None:
        if "Linear Regression" in selected_models and "Linear Regression" in results:
            fig.add_trace(go.Scatter(
                x=results["Year"], y=results["Linear Regression"],
                mode="lines+markers",
                name="Linear Regression",
                line=dict(color="blue")
            ))

        if "Random Forest" in selected_models and "Random Forest" in results:
            fig.add_trace(go.Scatter(
                x=results["Year"], y=results["Random Forest"],
                mode="lines+markers",
                name="Random Forest",
                line=dict(color="orange")
            ))

        if "ARIMA" in selected_models and "ARIMA" in results:
            fig.add_trace(go.Scatter(
                x=results["Year"], y=results["ARIMA"],
                mode="lines+markers",
                name="ARIMA",
                line=dict(color="green")
            ))

    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title="Value",
        template="plotly_white",
        legend=dict(x=0, y=1, bgcolor="rgba(0,0,0,0)")
    )

    st.plotly_chart(fig, use_container_width=True, key=key)

# ========================
# Streamlit UI
# ========================
st.title("🌍 GoalScope SDG Analytics Hub - Predictions")

# --- Entity Type ---
entity_type = st.selectbox("Entity Type", ["Country", "Region"])

# --- Entities ---
if entity_type == "Country":
    entities = sorted(df_sdg["Country"].unique())
else:
    entities = sorted(df_sdg["Region"].unique())

entity = st.selectbox("Select Entity", entities)

# --- Overall Score (if exists) ---
if "Overall_Score" in df_sdg.columns:
    df_score = df_sdg[df_sdg[entity_type] == entity][["Year", "Overall_Score"]].dropna()
    if not df_score.empty:
        latest_score = df_score.sort_values("Year").iloc[-1]["Overall_Score"]
        st.metric(label=f"{entity} Latest Overall SDG Score", value=f"{latest_score:.2f}")

# --- Level ---
level = st.selectbox("Level", ["Goal", "SDG", "Dimension"])

# --- Year Selection ---
year_range = st.slider("Select Prediction Range", 2026, 2030, (2026, 2030))
year_range = list(range(year_range[0], year_range[1] + 1))

# --- Model Selection ---
selected_models = st.multiselect(
    "Select Models",
    ["Linear Regression", "Random Forest", "ARIMA"],
    default=["Linear Regression", "Random Forest", "ARIMA"]
)

# --- Hyperparameter Settings ---
st.sidebar.header("⚙️ Model Parameters")

rf_params = {
    "n_estimators": st.sidebar.slider("RF n_estimators", 50, 300, 100, 10),
    "max_depth": st.sidebar.slider("RF max_depth", 2, 20, 5, 1)
}

arima_params = {
    "order": st.sidebar.selectbox("ARIMA order (p,d,q)", [(1,1,1), (2,1,2), (3,1,1)], index=0)
}

# --- Sidebar summary of selected parameters ---
st.sidebar.subheader("📋 Final Parameters Summary")
st.sidebar.write("Random Forest:", rf_params)
st.sidebar.write("ARIMA:", arima_params)

# --- Target Variable ---
if level == "Dimension":
    groups = df_lookup["group"].unique()
    target_group = st.selectbox("Select Dimension", groups)
    group_codes = df_lookup[df_lookup["group"] == target_group]["code"].tolist()
    targets = [col for col in df_sdg.columns if any(code in col for code in group_codes)]
else:
    targets = [c for c in df_sdg.columns if level.lower() in c.lower()]

if not targets:
    st.error(f"⚠️ No columns found for level {level}")
    st.stop()

if level != "Dimension":
    target = st.selectbox(f"Select {level}", ["All " + level + "s"] + targets)
else:
    target = None

# ========================
# Filter Data
# ========================
if entity_type == "Country":
    df_entity = df_sdg[df_sdg["Country"] == entity]
else:
    df_entity = df_sdg[df_sdg["Region"] == entity]

# ========================
# Run Predictions
# ========================
if df_entity.empty:
    st.error("⚠️ No data available for this selection.")
else:
    if level == "Dimension":
        st.subheader(f"📊 Predictions for Dimension: {target_group} ({year_range[0]}–{year_range[-1]})")
        for i, g_target in enumerate(targets):
            st.markdown(f"### {g_target}")
            results, metrics = run_models(df_entity, g_target, year_range, selected_models, rf_params, arima_params)
            if results is not None:
                st.dataframe(results)
                st.json(metrics)

                # ✅ Show parameters clearly
                with st.expander("Final Parameters Used"):
                    if "Random Forest" in metrics:
                        st.write("Random Forest:", metrics["Random Forest"]["Params"])
                    if "ARIMA" in metrics:
                        st.write("ARIMA:", metrics["ARIMA"]["Params"])

                plot_results(results, df_entity, g_target, f"{g_target} Predictions", key=f"{g_target}_{i}", selected_models=selected_models, entity_type=entity_type)
            else:
                st.warning(f"No data available for {g_target}.")
    else:
        if target == "All " + level + "s":
            for i, g_target in enumerate(targets):
                results, metrics = run_models(df_entity, g_target, year_range, selected_models, rf_params, arima_params)
                if results is not None:
                    st.subheader(f"{g_target} forecast for {entity} ({year_range[0]}–{year_range[-1]})")
                    st.dataframe(results)
                    st.json(metrics)

                    # ✅ Show parameters clearly
                    with st.expander("Final Parameters Used"):
                        if "Random Forest" in metrics:
                            st.write("Random Forest:", metrics["Random Forest"]["Params"])
                        if "ARIMA" in metrics:
                            st.write("ARIMA:", metrics["ARIMA"]["Params"])

                    plot_results(results, df_entity, g_target, f"{g_target} Predictions", key=f"{g_target}_{i}", selected_models=selected_models, entity_type=entity_type)
        else:
            results, metrics = run_models(df_entity, target, year_range, selected_models, rf_params, arima_params)
            if results is not None:
                st.subheader(f"{target} forecast for {entity} ({year_range[0]}–{year_range[-1]})")
                st.dataframe(results)
                st.json(metrics)

                # ✅ Show parameters clearly
                with st.expander("Final Parameters Used"):
                    if "Random Forest" in metrics:
                        st.write("Random Forest:", metrics["Random Forest"]["Params"])
                    if "ARIMA" in metrics:
                        st.write("ARIMA:", metrics["ARIMA"]["Params"])

                plot_results(results, df_entity, target, f"{target} Predictions", key=target, selected_models=selected_models, entity_type=entity_type)
            else:
                st.error("⚠️ Prediction failed. Please try another selection.")
