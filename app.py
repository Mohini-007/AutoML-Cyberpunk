# app.py - Complete AutoML System with Cyberpunk UI + SHAP live dashboard
import time
import io
import base64
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import os   # ✅ NEW

# ✅ Ensure model directory exists
os.makedirs("models", exist_ok=True)

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, f1_score
import joblib

# optional imports (may fail on some systems)
try:
    from xgboost import XGBRegressor, XGBClassifier
    xgb_available = True
except Exception:
    xgb_available = False

try:
    from lightgbm import LGBMRegressor, LGBMClassifier
    lgbm_available = True
except Exception:
    lgbm_available = False

# SHAP
import shap
shap._explanation.Explanation.__module__ = "shap"  # ✅ SHAP fix for waterfall crashes

# ---------- Global UI & Plot Styling ----------
plt.style.use('dark_background')

def neon_plot():
    plt.rcParams['axes.facecolor'] = '#0a001a'
    plt.rcParams['figure.facecolor'] = '#000000'
    plt.rcParams['axes.edgecolor'] = '#ff00ff'
    plt.rcParams['axes.labelcolor'] = '#e0e6f0'
    plt.rcParams['xtick.color'] = '#bb88ff'
    plt.rcParams['ytick.color'] = '#bb88ff'
    plt.rcParams['grid.color'] = '#5500aa'


neon_plot()


def neon_loader(text="Calibrating CyberNeurons...", seconds=1.2):
    with st.spinner(f"⚡ {text}"):
        time.sleep(seconds)


# ---------- App Title & CSS ----------
st.set_page_config(page_title="AutoML Cyberpunk", layout="wide")
st.markdown(
    """
    <style>
    body, .stApp {
      background: linear-gradient(135deg, #020014, #001133);
      color: #e0e6f0;
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    [data-testid="stSidebar"] {
      backdrop-filter: blur(9px);
      background: rgba(10, 0, 20, 0.45) !important;
      border-right: 2px solid rgba(255,0,255,0.06);
    }
    .stButton>button {
      background: linear-gradient(90deg, #ff0099, #6600ff) !important;
      color: white !important;
      border-radius: 8px;
      box-shadow: 0 0 12px #ff00ff;
      font-weight:700;
    }
    .cyber-box {
      background: rgba(10, 0, 40, 0.9);
      border: 2px solid #ff00ff;
      border-radius: 12px;
      padding: 18px;
      box-shadow: 0 0 20px #ff00ff55;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚡ AutoML System — Cyberpunk Edition")

# ---------- Helper functions ----------
def detect_problem_type(y_series):
    if pd.api.types.is_numeric_dtype(y_series):
        # numeric -> regression (except when it's clearly categorical encoded)
        if y_series.nunique() <= 6:
            # small number of unique numeric values might be classification
            return "classification"
        return "regression"
    return "classification"


def safe_get_session_model():
    return st.session_state.get("best_model_pipeline", None)


def save_best_pipeline(pipe):
    st.session_state["best_model_pipeline"] = pipe
    # also persist a copy on disk for safety
    try:
        joblib.dump(pipe, "models/best_model_pipeline.pkl")
    except Exception:
        pass


def get_feature_names_from_preprocessor(preprocessor):
    """
    Returns list of feature names after ColumnTransformer transformation.
    Assumes preprocessor is fitted.
    """
    names = []
    # ColumnTransformer.transformers_ has tuples: (name, transformer, columns)
    for name, transformer, columns in preprocessor.transformers_:
        if isinstance(columns, (list, np.ndarray)):
            cols = list(columns)
        else:
            # if columns are strings or slice, try to convert
            try:
                cols = list(columns)
            except Exception:
                cols = [columns]
        if transformer is None:
            names.extend(cols)
        else:
            # if transformer is a Pipeline, find the last step
            if hasattr(transformer, "named_steps"):
                # for our pipeline, encoder lives at named_steps['encoder']
                if "encoder" in transformer.named_steps:
                    enc = transformer.named_steps["encoder"]
                    try:
                        ohe_names = list(enc.get_feature_names_out(cols))
                        names.extend(ohe_names)
                    except Exception:
                        names.extend(cols)
                else:
                    # fallback - transformer may be a scaler etc
                    names.extend(cols)
            else:
                # direct OneHotEncoder or scaler
                if hasattr(transformer, "get_feature_names_out"):
                    try:
                        names.extend(list(transformer.get_feature_names_out(cols)))
                    except Exception:
                        names.extend(cols)
                else:
                    names.extend(cols)
    return names


def df_to_shap_row(preprocessor, df_row, feature_names):
    proc = preprocessor.transform(df_row)
    return pd.DataFrame(proc, columns=feature_names)


# ---------- File upload & core flow ----------
uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV file to begin. Example: house_price.csv")
    st.stop()

# read file
try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# Target selection
target_col = st.selectbox("🎯 Select Target Column", df.columns, key="target_select")

# Basic checks
if target_col is None:
    st.warning("Select a target column to continue.")
    st.stop()

# make X,y global variables in this run
X = df.drop(columns=[target_col])
y = df[target_col]

# Column detection
numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

st.write("### 🔧 Column Types Detected")
st.write("**Numeric:**", numeric_cols)
st.write("**Categorical:**", categorical_cols)

# Optionally drop ID-like columns
drop_ids = st.checkbox("Auto-drop ID-like columns (unique per row)", value=True, key="drop_ids")
if drop_ids:
    id_cols = [c for c in X.columns if X[c].nunique() == len(X)]
    if id_cols:
        st.info(f"Dropping ID-like columns: {id_cols}")
        X = X.drop(columns=id_cols)
        numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

# Build preprocessor
numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
categorical_transformer = Pipeline(steps=[("encoder", OneHotEncoder(handle_unknown="ignore"))])

# handle case when there are no categorical or numeric columns
transformers = []
if numeric_cols:
    transformers.append(("num", numeric_transformer, numeric_cols))
if categorical_cols:
    transformers.append(("cat", categorical_transformer, categorical_cols))

if not transformers:
    st.error("No features found after preprocessing - cannot continue.")
    st.stop()

preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

# Train/test split
test_size = st.sidebar.slider("Test set size (%)", 5, 40, 20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(test_size / 100.0), random_state=42)

st.success("✅ Data preprocessing ready and train/test split created.")

# ---------- Models registry ----------
st.subheader("🚀 Model Training & Comparison")

# Define model candidates depending on detected problem type
problem_type = detect_problem_type(y)
st.write(f"Detected task type: **{problem_type}**")

if problem_type == "regression":
    model_constructors = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    }
    if xgb_available:
        model_constructors["XGBoost"] = XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42, verbosity=0)
    if lgbm_available:
        model_constructors["LightGBM"] = LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
    primary_metric = "MAE"
else:
    model_constructors = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }
    if xgb_available:
        model_constructors["XGBoost"] = XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric="logloss")
    if lgbm_available:
        model_constructors["LightGBM"] = LGBMClassifier(n_estimators=200, random_state=42)
    primary_metric = "F1"

# Train models and evaluate
results = {}
pipelines = {}

neon_loader("Training models... (this may take a while)", seconds=1.0)

for name, model in model_constructors.items():
    st.write(f"Training: **{name}**")
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    try:
        pipe.fit(X_train, y_train)
    except Exception as e:
        st.warning(f"Training failed for {name}: {e}")
        continue

    # predictions and metrics
    preds = pipe.predict(X_test)
    if problem_type == "regression":
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        results[name] = {"MAE": mae, "R2": r2}
    else:
        f1 = f1_score(y_test, preds, average="weighted")
        acc = accuracy_score(y_test, preds)
        results[name] = {"F1": f1, "Accuracy": acc}

    pipelines[name] = pipe

# present results
st.write("### 🏁 Model Performance Comparison")
if problem_type == "regression":
    perf_df = pd.DataFrame([{ "Model": k, "MAE": v["MAE"], "R2": v["R2"] } for k,v in results.items()]).sort_values("MAE")
else:
    perf_df = pd.DataFrame([{ "Model": k, "F1": v["F1"], "Accuracy": v["Accuracy"] } for k,v in results.items()]).sort_values(primary_metric, ascending=(primary_metric!="F1"))

st.dataframe(perf_df)

# pick best model based on primary metric
if problem_type == "regression":
    best_model_name = min(results.keys(), key=lambda k: results[k]["MAE"])
else:
    best_model_name = max(results.keys(), key=lambda k: results[k]["F1"])

st.success(f"🥇 Best Model: **{best_model_name}**")

from notify import send_notification
send_notification("✅ Model retraining finished!\nBest model updated and logged to MLflow.")

# store persistent pipeline
best_pipeline = pipelines[best_model_name]
# store pipeline in session for reuse
save_best_pipeline(best_pipeline)

# allow download of best model
import joblib
import io

# Save to in-memory buffer
buffer = io.BytesIO()
joblib.dump(best_pipeline, buffer)
buffer.seek(0)

st.download_button(
    label="⬇️ Download Best Model (.joblib)",
    data=buffer,
    file_name="best_model_pipeline.joblib",
    mime="application/octet-stream",
    key="download_best_model"
)

# ---------- Prediction terminal (Neon) ----------
st.markdown("## 🧪 Neon Terminal — Make a Prediction")

# get pipeline from session safely
best_model_pipeline = safe_get_session_model()
if best_model_pipeline is None:
    st.warning("Train a model first (upload dataset and run training).")
else:
    # numeric features for input form
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()

    st.markdown('<div class="cyber-box">', unsafe_allow_html=True)
    st.write("Enter values for features (numeric only).")
    input_values = {}
    # create unique keys for each input so Streamlit doesn't complain
    for col in numeric_features:
        default = float(X[col].median()) if X[col].dtype.kind in 'fi' else 0.0
        minv = float(X[col].min()) if X[col].dtype.kind in 'fi' else 0.0
        maxv = float(X[col].max()) if X[col].dtype.kind in 'fi' else default + 1.0
        input_values[col] = st.number_input(f"🔹 {col}", value=default, key=f"input_{col}")
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("⚡ RUN PREDICTION", key="predict_btn"):
        user_df = pd.DataFrame([input_values])
        try:
            pred = best_model_pipeline.predict(user_df)[0]
            st.markdown(f"<h2 style='color:#ff55ff;'>✅ Predicted ({best_model_name}): <span style='color:#80ffea'> {pred:.4f}</span></h2>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ---------- Live Feature Attribution Dashboard ----------
st.header("⚡ Live Feature Attribution Dashboard")

# ensure pipeline exists
best_model_pipeline = safe_get_session_model()
if best_model_pipeline is None:
    st.warning("Train a model to enable live explainability dashboard.")
    st.stop()

# prepare preprocessor and model
pre = best_model_pipeline.named_steps["preprocessor"]
model_final = best_model_pipeline.named_steps["model"]

# Recover feature names after preprocessing
try:
    # ensure preprocessor is fitted; if pipeline was fit earlier it should be.
    feature_names = get_feature_names_from_preprocessor(pre)
except Exception as e:
    st.warning(f"Failed to recover feature names: {e}")
    feature_names = None

# Build shap-ready DataFrame
try:
    X_trans = pre.transform(X)
    if feature_names is None or len(feature_names) != X_trans.shape[1]:
        # fallback: name numeric + generated OHE names
        feature_names = [f"f{i}" for i in range(X_trans.shape[1])]
    X_shap_df = pd.DataFrame(X_trans, columns=feature_names)
except Exception as e:
    st.error(f"Failed to transform X for SHAP: {e}")
    st.stop()

# build explainer
explainer = None
shap_values_full = None
try:
    # Prefer tree explainer for tree models
    model_name = type(model_final).__name__
    if "RandomForest" in model_name or "XGB" in model_name or "LGBM" in model_name:
        explainer = shap.TreeExplainer(model_final)
        shap_values_full = explainer(X_shap_df)
    else:
        explainer = shap.Explainer(model_final.predict, X_shap_df)
        shap_values_full = explainer(X_shap_df)
except Exception as e:
    st.warning(f"Could not build SHAP explainer automatically. Error: {e}")

# Global SHAP summary
st.subheader("Global Feature Impact")
if shap_values_full is not None:
    fig = plt.figure(figsize=(10, 4))
    shap.summary_plot(shap_values_full, X_shap_df, show=False, max_display=12)
    st.pyplot(fig)
else:
    st.info("SHAP global summary unavailable for this model.")

# Single-sample / custom input mode
st.subheader("Live Single-sample Attribution")
col1, col2 = st.columns([1, 1])
with col1:
    sample_mode = st.radio("Pick sample source:", ["From dataset", "Create custom sample"], key="sample_mode_radio")
with col2:
    sample_index = None
    if sample_mode == "From dataset":
        sample_index = st.selectbox("Choose row index:", options=list(X.index), key="sample_index_box")

def df_to_shap_row_local(df_row):
    proc = pre.transform(df_row)
    return pd.DataFrame(proc, columns=feature_names)

if sample_mode == "From dataset":
    sample_row = X.loc[[sample_index]]
    st.markdown("**Selected sample (raw features):**")
    st.table(sample_row)

    try:
        pred = best_model_pipeline.predict(sample_row)[0]
        st.markdown(f"**Model prediction for selected sample:**  **{pred:.4f}**")
    except Exception as e:
        st.warning(f"Prediction failed for selected sample: {e}")

    if shap_values_full is not None:
        row_pos = list(X.index).index(sample_index)
        sv = shap_values_full[row_pos]
        st.markdown("**Feature contributions (descending):**")
        try:
            contribs = pd.Series(sv.values, index=feature_names).sort_values(ascending=False)
            st.dataframe(contribs.head(12).to_frame(name="SHAP contribution"))
        except Exception:
            # older shap value formats
            try:
                contribs = pd.Series(sv, index=feature_names).sort_values(ascending=False)
                st.dataframe(contribs.head(12).to_frame(name="SHAP contribution"))
            except Exception as e:
                st.warning(f"Failed to display contributions: {e}")

        # SHAP waterfall
        st.write("#### SHAP Waterfall (single prediction)")
        try:
            fig2 = plt.figure(figsize=(6, 4))
            shap.plots.waterfall(sv, show=False)
            st.pyplot(fig2)
        except Exception as e:
            st.warning(f"Waterfall plot failed: {e}")

if sample_mode == "Create custom sample":
    st.markdown("**Create custom input (numeric features only)**")
    custom_vals = {}
    for c in X.select_dtypes(include=np.number).columns:
        lo = float(X[c].min())
        hi = float(X[c].max())
        default = float(X[c].median())
        step = (hi - lo) / 100 if hi > lo else 1.0
        custom_vals[c] = st.slider(f"{c}", min_value=lo, max_value=hi, value=default, step=step, key=f"custom_{c}")

    user_df = pd.DataFrame([custom_vals])
    st.markdown("**Custom input preview:**")
    st.table(user_df)

    try:
        pred_custom = best_model_pipeline.predict(user_df)[0]
        st.markdown(f"**Model prediction (custom):**  **{pred_custom:.4f}**")
    except Exception as e:
        st.warning(f"Prediction failed for custom input: {e}")

    if explainer is not None:
        try:
            user_shap_row = df_to_shap_row_local(user_df)
            sv_custom = explainer(user_shap_row)
            contribs_c = pd.Series(sv_custom.values[0], index=feature_names).sort_values(ascending=False)
            st.dataframe(contribs_c.head(12).to_frame(name="SHAP contribution"))

            st.write("#### SHAP Waterfall (custom input)")
            fig3 = plt.figure(figsize=(6, 4))
            shap.plots.waterfall(sv_custom[0], show=False)
            st.pyplot(fig3)
        except Exception as e:
            st.warning(f"SHAP failed for custom input: {e}")

# Export contributions for chosen sample
if st.button("⬇️ Export selected sample contributions as CSV", key="export_shap"):
    if sample_mode == "From dataset":
        chosen = X.loc[[sample_index]]
    else:
        chosen = user_df
    chosen_proc = df_to_shap_row_local(chosen)
    if explainer is not None:
        try:
            sv_export = explainer(chosen_proc)
            # handle shap Values object shapes
            try:
                values = sv_export.values
            except Exception:
                values = sv_export
            contribs_export = pd.DataFrame(values, columns=feature_names)
            csv_bytes = contribs_export.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv_bytes, file_name="shap_contributions.csv", mime="text/csv", key="download_csv")
        except Exception as e:
            st.warning(f"Export failed: {e}")
    else:
        st.warning("No SHAP explainer available; nothing to export.")

st.write("✅ All done. Use the panels above to experiment with predictions & explanations.")
