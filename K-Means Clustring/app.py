import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Facebook Live K-Means Dashboard",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

@st.cache_data
def load_data():
    return pd.read_csv(BASE_DIR / "Live.csv")

@st.cache_resource
def load_artifacts():
    return joblib.load(BASE_DIR / "kmeans_clustering_artifacts.pkl")

df = load_data()
artifacts = load_artifacts()
model = artifacts["model"]
scaler = artifacts["scaler"]
le = artifacts["label_encoder"]
feature_columns = artifacts["feature_columns"]

# Same preprocessing as notebook
work_df = df.copy()
work_df.drop(
    ["Column1", "Column2", "Column3", "Column4"],
    axis=1,
    inplace=True,
    errors="ignore"
)
work_df.drop(
    ["status_id", "status_published"],
    axis=1,
    inplace=True,
    errors="ignore"
)

X = work_df.copy()
X["status_type"] = le.transform(X["status_type"])
X = X[feature_columns]

X_scaled = scaler.transform(X)
work_df["Cluster"] = model.predict(X_scaled)

# Sidebar
st.sidebar.title("Dashboard Controls")
st.sidebar.write("K-Means Clustering | k = 4")

cluster_options = ["All"] + sorted(work_df["Cluster"].unique().tolist())
selected_cluster = st.sidebar.selectbox("Select Cluster", cluster_options)

if selected_cluster == "All":
    filtered = work_df.copy()
else:
    filtered = work_df[work_df["Cluster"] == selected_cluster].copy()

st.title("📊 Facebook Live K-Means Clustering Dashboard")
st.markdown(
    "Interactive dashboard based on the **Live.csv** dataset and the "
    "preprocessing/modeling workflow from the supplied K-Means notebook."
)

# KPI cards
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Records", f"{len(work_df):,}")
c2.metric("Clusters", model.n_clusters)
c3.metric("Displayed Records", f"{len(filtered):,}")
c4.metric("Model Inertia", f"{model.inertia_:.2f}")

st.divider()

# Cluster distribution
left, right = st.columns(2)

with left:
    st.subheader("Cluster Distribution")
    counts = work_df["Cluster"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Records")
    ax.set_title("Records per Cluster")
    st.pyplot(fig, clear_figure=True)

with right:
    st.subheader("Cluster Percentage")
    pct = (counts / len(work_df) * 100).round(2)
    pct_df = pd.DataFrame({
        "Cluster": counts.index,
        "Records": counts.values,
        "Percentage": pct.values
    })
    st.dataframe(pct_df, use_container_width=True, hide_index=True)

st.divider()

# Status type distribution
st.subheader("Status Type Distribution")

status_cluster = pd.crosstab(work_df["status_type"], work_df["Cluster"])
st.dataframe(status_cluster, use_container_width=True)

fig2, ax2 = plt.subplots(figsize=(10, 4))
status_cluster.plot(kind="bar", ax=ax2)
ax2.set_xlabel("Status Type")
ax2.set_ylabel("Number of Records")
ax2.set_title("Status Type by Cluster")
ax2.legend(title="Cluster")
st.pyplot(fig2, clear_figure=True)

st.divider()

# Numerical summary
st.subheader("Numerical Feature Summary")
numeric_cols = [c for c in work_df.columns if c not in ["status_type", "Cluster"]]
summary = filtered[numeric_cols].describe().T
st.dataframe(summary, use_container_width=True)

st.divider()

# Prediction section
st.subheader("🔍 Predict Cluster for a New Record")

st.caption(
    "Enter values for the five numeric features used in the notebook. "
    "Choose the status type from the original categorical labels."
)

status_values = list(le.classes_)
input_cols = [c for c in feature_columns if c != "status_type"]

defaults = {}
for c in input_cols:
    defaults[c] = float(work_df[c].median())

form_cols = st.columns(3)
new_values = {}

for i, c in enumerate(input_cols):
    with form_cols[i % 3]:
        new_values[c] = st.number_input(
            c,
            value=float(defaults[c]),
            step=1.0
        )

new_status = st.selectbox("status_type", status_values)

if st.button("Predict Cluster", type="primary"):
    new_row = pd.DataFrame([new_values])
    new_row["status_type"] = le.transform([new_status])[0]
    new_row = new_row[feature_columns]
    new_scaled = scaler.transform(new_row)
    prediction = int(model.predict(new_scaled)[0])
    st.success(f"Predicted Cluster: {prediction}")

st.divider()

st.subheader("📋 Filtered Data")
st.dataframe(filtered.head(500), use_container_width=True, hide_index=True)

st.caption("Model: K-Means | Number of clusters: 4 | Random state: 0")
