# =============================================================================
# MohammedNahyanKhan_SupplyChainApp.py
# Supply Chain Delivery Prediction — AICTE Internship Project
# Author : Mohammed Nahyan Khan
# Dataset: Kaggle E-Commerce Shipping Data (Train.csv)
# Run    : streamlit run MohammedNahyanKhan_SupplyChainApp.py
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be the first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Supply Chain Delivery Prediction",
    page_icon="📦",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — DATA LOADING & PREPROCESSING
# @st.cache_data caches the returned DataFrames so the CSV is read only once.
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_and_prepare_data(csv_path: str = "Train.csv"):
    """
    Load Train.csv, drop the ID column (no predictive power),
    label-encode the four categorical columns, and return:
        df_raw  – original DataFrame (used for EDA plots)
        X       – feature matrix (encoded)
        y       – target Series  (Reached.on.Time_Y.N)
        encoders – dict of fitted LabelEncoder objects keyed by column name
    """
    df = pd.read_csv(csv_path)

    # Drop row-identifier — carries zero predictive signal
    df.drop(columns=["ID"], inplace=True)

    # Keep a raw copy for EDA visualisations (before encoding)
    df_raw = df.copy()

    # Categorical columns that need integer encoding for the model
    cat_cols = ["Warehouse_block", "Mode_of_Shipment", "Product_importance", "Gender"]
    encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le          # store each fitted encoder for reuse at prediction time

    # Split features / target
    X = df.drop(columns=["Reached.on.Time_Y.N"])
    y = df["Reached.on.Time_Y.N"]

    return df_raw, X, y, encoders


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — MODEL TRAINING
# @st.cache_resource caches the trained model object (not serialisable as data).
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def train_model(_X, _y):
    """
    Split the encoded dataset 80/20, train a Random Forest classifier,
    and return the fitted model together with its test-set accuracy.

    Parameters use a leading underscore so Streamlit's cache knows not to
    hash the DataFrames (avoids unhashable-type errors on large arrays).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        _X, _y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        random_state=42,
        n_jobs=-1,          # use all available CPU cores
    )
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA & TRAIN MODEL
# ─────────────────────────────────────────────────────────────────────────────
df_raw, X, y, encoders = load_and_prepare_data("Train.csv")
model, acc = train_model(X, y)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — SIDEBAR: USER INPUT WIDGETS
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📋 Enter Shipping Details")
    st.markdown("Adjust the parameters below and the prediction updates instantly.")
    st.divider()

    # ── Numerical inputs ──────────────────────────────────────────────────────
    customer_care_calls = st.slider(
        "Customer Care Calls",
        min_value=2, max_value=7, value=4,
        help="Number of calls made to customer care regarding this shipment.",
    )
    customer_rating = st.slider(
        "Customer Rating  (1 = Poor, 5 = Excellent)",
        min_value=1, max_value=5, value=3,
    )
    cost_of_product = st.number_input(
        "Cost of the Product (₹)",
        min_value=96, max_value=310, value=200, step=1,
    )
    prior_purchases = st.slider(
        "Prior Purchases by this Customer",
        min_value=2, max_value=10, value=4,
    )
    discount_offered = st.slider(
        "Discount Offered (%)",
        min_value=1, max_value=65, value=10,
    )
    weight_in_gms = st.number_input(
        "Weight of Product (grams)",
        min_value=1001, max_value=7846, value=3000, step=50,
    )

    st.divider()

    # ── Categorical inputs ────────────────────────────────────────────────────
    warehouse_block = st.selectbox(
        "Warehouse Block",
        options=["A", "B", "C", "D", "F"],
    )
    mode_of_shipment = st.selectbox(
        "Mode of Shipment",
        options=["Flight", "Road", "Ship"],
    )
    product_importance = st.selectbox(
        "Product Importance",
        options=["high", "low", "medium"],
    )
    gender = st.selectbox(
        "Customer Gender",
        options=["F", "M"],
        format_func=lambda x: "Female" if x == "F" else "Male",
    )

    st.divider()
    st.caption(f"🌲 Model accuracy on held-out test set: **{acc * 100:.1f}%**")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — MAIN PAGE
# ─────────────────────────────────────────────────────────────────────────────
st.title("📦 Supply Chain Delivery Prediction")
st.markdown(
    """
    This application predicts whether a shipment will **arrive on time or be delayed**,
    based on historical e-commerce shipping data.  
    Use the **sidebar on the left** to enter the details of a new shipment.  
    The Random Forest model was trained on **10,999 orders** and updates its prediction
    in real time as you adjust the inputs.
    """
)

st.divider()

# ── Feature Analysis Section ─────────────────────────────────────────────────
st.subheader("📊 Feature Analysis — Historical Data")

col_plot, col_info = st.columns([2, 1])

with col_plot:
    st.markdown(
        "**Discount Offered vs Delivery Status**  \n"
        "Late shipments (1) tend to carry a higher median discount, "
        "suggesting promotional demand spikes strain fulfilment capacity."
    )

    # Build a plot-friendly copy with readable labels on the x-axis
    df_plot = df_raw.copy()
    df_plot["Delivery Status"] = df_plot["Reached.on.Time_Y.N"].map(
        {0: "On Time (0)", 1: "Late (1)"}
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(
        data=df_plot,
        x="Delivery Status",
        y="Discount_offered",
        palette={"On Time (0)": "#2ecc71", "Late (1)": "#e74c3c"},
        order=["On Time (0)", "Late (1)"],
        width=0.45,
        flierprops=dict(marker="o", markerfacecolor="grey", markersize=3, alpha=0.4),
        ax=ax,
    )
    # Annotate median values
    medians = df_plot.groupby("Delivery Status")["Discount_offered"].median()
    for tick, (label, mval) in enumerate(medians.items()):
        ax.text(
            tick, mval + 0.8,
            f"Median: {mval:.1f}%",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )
    ax.set_title("Discount Offered vs On-Time Delivery", fontsize=11)
    ax.set_xlabel("Delivery Status")
    ax.set_ylabel("Discount Offered (%)")
    sns.despine()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)   # free memory — critical in long-running Streamlit sessions

with col_info:
    st.metric("Total Orders", f"{len(df_raw):,}")
    on_time_pct = (df_raw["Reached.on.Time_Y.N"] == 0).mean() * 100
    late_pct = 100 - on_time_pct
    st.metric("On-Time Rate", f"{on_time_pct:.1f}%")
    st.metric("Late Rate", f"{late_pct:.1f}%")
    st.metric("Features Used", str(X.shape[1]))
    st.metric("Model", "Random Forest (150 trees)")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — PREDICTION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("🔮 Prediction Result")

# Assemble user inputs into a single-row DataFrame with the same column order
# that the model was trained on
user_input = pd.DataFrame(
    {
        "Warehouse_block": [warehouse_block],
        "Mode_of_Shipment": [mode_of_shipment],
        "Customer_care_calls": [customer_care_calls],
        "Customer_rating": [customer_rating],
        "Cost_of_the_Product": [cost_of_product],
        "Prior_purchases": [prior_purchases],
        "Product_importance": [product_importance],
        "Gender": [gender],
        "Discount_offered": [discount_offered],
        "Weight_in_gms": [weight_in_gms],
    }
)

# Encode categorical columns using the same LabelEncoders fitted during training
# so the integer mapping is identical to what the model learned
for col in ["Warehouse_block", "Mode_of_Shipment", "Product_importance", "Gender"]:
    user_input[col] = encoders[col].transform(user_input[col])

# Reorder columns to exactly match the training feature matrix
user_input = user_input[X.columns]

# Run inference
prediction = model.predict(user_input)[0]           # scalar: 0 or 1
prediction_proba = model.predict_proba(user_input)  # [[P(0), P(1)]]
prob_on_time = prediction_proba[0][0] * 100
prob_late    = prediction_proba[0][1] * 100

# ── Display result ────────────────────────────────────────────────────────────
if prediction == 1:
    st.error(
        "🚨  **Prediction: This shipment is likely to be DELAYED.**\n\n"
        f"The model estimates a **{prob_late:.1f}% probability of delay** "
        f"and a {prob_on_time:.1f}% probability of on-time delivery."
    )
else:
    st.success(
        "✅  **Prediction: This shipment will arrive ON TIME.**\n\n"
        f"The model estimates a **{prob_on_time:.1f}% probability of on-time delivery** "
        f"and a {prob_late:.1f}% probability of delay."
    )

# Confidence gauge (visual bar)
st.markdown("**Model Confidence Breakdown**")
conf_col1, conf_col2 = st.columns(2)
conf_col1.metric("P(On Time)", f"{prob_on_time:.1f}%")
conf_col2.metric("P(Late)", f"{prob_late:.1f}%")
st.progress(int(prob_on_time))

st.divider()
st.caption(
    "AICTE Internship Project · Mohammed Nahyan Khan · "
    "Dataset: Kaggle E-Commerce Shipping Data"
)
