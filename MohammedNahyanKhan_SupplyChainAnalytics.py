import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import shap
from streamlit_shap import st_shap

# 1. Page Configuration
st.set_page_config(page_title="Supply Chain Analytics", layout="wide")
st.title("📦 Supply Chain Delivery Prediction & Analytics")

# 2. Data Loading & Resilience
@st.cache_data
def load_data():
    try:
        # Tries to load Kaggle data
        df = pd.read_csv('Train.csv')
    except FileNotFoundError:
        # Fallback synthetic data so your app never crashes
        rng = np.random.default_rng(42)
        n = 1000
        df = pd.DataFrame({
            'Warehouse_block': rng.choice(['A', 'B', 'C', 'D', 'F'], n),
            'Mode_of_Shipment': rng.choice(['Flight', 'Ship', 'Road'], n),
            'Customer_care_calls': rng.integers(2, 8, n),
            'Customer_rating': rng.integers(1, 6, n),
            'Prior_purchases': rng.integers(2, 11, n),
            'Discount_offered': rng.integers(1, 66, n),
            'Weight_in_gms': rng.integers(1001, 7847, n)
        })
        boundary = 48 - (df['Weight_in_gms'] * 0.0055)
        df['Reached.on.Time_Y.N'] = np.where(df['Discount_offered'] > boundary, 1, 0)
    return df

df = load_data()

# 3. Model Training
@st.cache_resource
def train_model(data):
    df_encoded = data.copy()
    le = LabelEncoder()
    df_encoded['Warehouse_block'] = le.fit_transform(df_encoded['Warehouse_block'])
    df_encoded['Mode_of_Shipment'] = le.fit_transform(df_encoded['Mode_of_Shipment'])

    X = df_encoded[['Customer_care_calls', 'Discount_offered', 'Weight_in_gms', 'Prior_purchases']]
    y = df_encoded['Reached.on.Time_Y.N']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)

    acc = accuracy_score(y_test, rf.predict(X_test))
    return rf, acc, X.columns

model, accuracy, feature_names = train_model(df)

# 4. Tab Layout
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Exploratory Data Analysis",
    "🎛️ Predictive Simulator",
    "🧠 Model Insights",
    "📂 Bulk Scoring & ROI"
])

# --- TAB 1: EDA Dashboard ---
with tab1:
    st.header("Exploratory Data Analysis")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Shipments", f"{len(df):,}")
    col2.metric("Overall Delay Rate", f"{(df['Reached.on.Time_Y.N'].mean() * 100):.1f}%")
    col3.metric("Average Discount", f"{df['Discount_offered'].mean():.1f}%")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Delays by Warehouse Block")
        fig_wh = px.histogram(df, x="Warehouse_block", color="Reached.on.Time_Y.N",
                              barmode="group", color_discrete_sequence=['#3b82f6', '#22c55e'],
                              labels={'Reached.on.Time_Y.N': 'Delayed (1=Yes, 0=No)'})
        st.plotly_chart(fig_wh, use_container_width=True)
    with c2:
        st.subheader("Weight vs. Discount Distribution")
        sample_df = df.sample(min(2000, len(df)), random_state=42)
        fig_scatter = px.scatter(sample_df, x="Weight_in_gms", y="Discount_offered",
                                 color="Reached.on.Time_Y.N", color_continuous_scale=['#3b82f6', '#22c55e'])
        st.plotly_chart(fig_scatter, use_container_width=True)

# --- TAB 2: The Simulator ---
with tab2:
    st.header("Delivery Classification Simulator")

    col1, col2, col3 = st.columns(3)
    with col1:
        user_weight = st.slider("Product Weight (g)", 1001, 7846, 3000, step=100)
    with col2:
        user_discount = st.slider("Discount %", 1, 65, 31, step=1)
    with col3:
        user_calls = st.slider("Care Calls", 2, 7, 3, step=1)

    input_data = pd.DataFrame([[user_calls, user_discount, user_weight, 3]], columns=feature_names)
    prob = model.predict_proba(input_data)[0]

    is_late = prob[1] > 0.5
    prediction_text = "Late" if is_late else "On-Time"
    prediction_color = "#22c55e" if is_late else "#3b82f6"
    confidence = int(max(prob) * 100)

    st.markdown(
        f"""
        <div style='text-align: center; padding: 20px;'>
<h3 style='margin:0;'>PREDICTION: <span style='color:{prediction_color}'>{prediction_text}</span>
</h3>
<p style='margin:0; font-size: 18px; color: #ccc;'>CONFIDENCE: {confidence}%</p>
</div>
        """, unsafe_allow_html=True
    )

    fig = go.Figure()
    sample_sim = df.sample(min(1500, len(df)), random_state=42)
    on_time = sample_sim[sample_sim['Reached.on.Time_Y.N'] == 0]
    late = sample_sim[sample_sim['Reached.on.Time_Y.N'] == 1]

    fig.add_trace(go.Scatter(x=on_time['Weight_in_gms'], y=on_time['Discount_offered'], mode='markers',
                             name='On-Time', marker=dict(color='#3b82f6', size=6, opacity=0.4)))
    fig.add_trace(go.Scatter(x=late['Weight_in_gms'], y=late['Discount_offered'], mode='markers',
                             name='Late', marker=dict(color='#22c55e', size=6, opacity=0.4)))

    fig.add_trace(go.Scatter(x=[user_weight], y=[user_discount], mode='markers', name='Current Input',
                             marker=dict(color=prediction_color, size=22, line=dict(color='white', width=3))))

    fig.update_layout(xaxis_title="Product Weight (g)", yaxis_title="Discount Offered (%)",
                      plot_bgcolor='#0e1117', paper_bgcolor='#0e1117', font=dict(color='white'))
    st.plotly_chart(fig, use_container_width=True)

    # ── SHAP Local Explainability ──────────────────────────────────────────────
    st.subheader("Prediction Breakdown")

    # TreeExplainer is optimised for tree-based models (Random Forest, XGBoost, etc.)
    # and runs orders of magnitude faster than the generic KernelExplainer.
    explainer   = shap.TreeExplainer(model)

    # Calculate SHAP values for the single user-input row.
    # Newer SHAP versions return a 3-D ndarray (n_samples, n_features, n_classes)
    # instead of a list; we handle both shapes for forward-compatibility.
    raw_sv = explainer.shap_values(input_data)
    if isinstance(raw_sv, list):
        # legacy format: list[class_0_array, class_1_array]
        sv_row      = raw_sv[1][0]          # 1-D array of shape (n_features,)
        base_value  = explainer.expected_value[1]
    else:
        # modern format: ndarray of shape (n_samples, n_features, n_classes)
        sv_row      = raw_sv[0, :, 1]       # 1-D array of shape (n_features,)
        base_value  = explainer.expected_value[1]

    # force_plot shows each feature's contribution pushing the prediction
    # left (toward On-Time) or right (toward Late) from the base value.
    st_shap(
        shap.force_plot(
            base_value,     # expected output for class 1 (Late)
            sv_row,         # per-feature SHAP contributions for this row
            input_data,     # feature names + values shown in the plot
        ),
        height=160,
    )

# --- TAB 3: Model Insights ---
with tab3:
    st.header("Model Insights")
    st.markdown(f"**Test Set Accuracy:** `{accuracy * 100:.1f}%`")

    importance = model.feature_importances_
    fig_imp = px.bar(x=importance, y=feature_names, orientation='h',
                     text=np.round(importance, 3),
                     labels={'x': 'Importance', 'y': 'Feature'},
                     color=importance, color_continuous_scale='Blues')
    fig_imp.update_traces(textposition='outside')
    fig_imp.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_imp, use_container_width=True)

    st.info("""
    **Business Takeaways:**
    1. **Discount Thresholds:** High discounts are the strongest indicator of delayed shipments.
    2. **Heavy Product Handling:** Product weight significantly shifts the prediction probability.
    3. **Support Call Leading Indicator:** High customer care calls correlate strongly with problematic logistics routes.
    """)

# --- TAB 4: Bulk Scoring & ROI ---
with tab4:
    st.header("Batch Processing & Financial Impact")
    st.markdown("Upload a daily shipping manifest (CSV) to predict delays at scale and calculate potential cost savings.")

    # 1. Provide a downloadable template so users know the format
    template_df = pd.DataFrame(columns=['Customer_care_calls', 'Discount_offered', 'Weight_in_gms', 'Prior_purchases'])
    csv_template = template_df.to_csv(index=False).encode('utf-8')

    col_upload, col_template = st.columns([3, 1])
    with col_template:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing
        st.download_button(
            label="📄 Download CSV Template",
            data=csv_template,
            file_name='shipping_manifest_template.csv',
            mime='text/csv',
        )

    with col_upload:
        uploaded_file = st.file_uploader("Upload Shipment Data", type=["csv"])

    if uploaded_file is not None:
        try:
            # Read uploaded data
            batch_df = pd.read_csv(uploaded_file)

            # Ensure columns match the model's expected features
            if not all(col in batch_df.columns for col in feature_names):
                st.error(f"Uploaded CSV must contain exactly these columns: {', '.join(feature_names)}")
            else:
                # Run predictions
                batch_predictions = model.predict(batch_df[feature_names])
                batch_probabilities = model.predict_proba(batch_df[feature_names])[:, 1]

                # Append results to the dataframe
                batch_df['Predicted_Delay'] = np.where(batch_predictions == 1, 'Yes', 'No')
                batch_df['Delay_Confidence_%'] = np.round(batch_probabilities * 100, 1)

                # Calculate ROI Metrics (Assuming a delayed package costs $15 in refunds/support)
                cost_per_delay = 15.00
                total_delayed = sum(batch_predictions)
                potential_cost = total_delayed * cost_per_delay

                st.markdown("### Batch Results")
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Shipments Processed", len(batch_df))
                m2.metric("Flagged for Delay", total_delayed, delta_color="inverse")
                m3.metric("Value at Risk (VAR)", f"${potential_cost:,.2f}",
                          help="Estimated cost in customer refunds and support tickets if these delays are not mitigated.")

                # Display the scored dataframe
                st.dataframe(batch_df.style.map(
                    lambda x: 'background-color: rgba(220, 38, 38, 0.2)' if x == 'Yes' else '',
                    subset=['Predicted_Delay']
                ), use_container_width=True)

                # Provide download button for the scored results
                scored_csv = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="💾 Download Scored Manifest",
                    data=scored_csv,
                    file_name='scored_shipments.csv',
                    mime='text/csv',
                    type="primary"
                )

        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.info("Awaiting CSV upload. You can generate a test file by copying a few rows from the EDA tab or downloading the template.")
