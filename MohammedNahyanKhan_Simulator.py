import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 1. Page Configuration
st.set_page_config(page_title="Delivery Simulator", layout="centered")
st.title("Delivery Classification Simulator")

# 2. Generate Synthetic Background Data (mimicking the dataset distribution)
np.random.seed(42)
weights = np.random.uniform(1000, 8000, 100)
discounts = np.random.uniform(0, 60, 100)

# Define a mathematical decision boundary: Discount = 48 - (Weight * 0.0055)
boundary_line = 48 - (weights * 0.0055)
labels = np.where(discounts > boundary_line, 'Late', 'On-Time')
df = pd.DataFrame({'Weight': weights, 'Discount': discounts, 'Status': labels})

# 3. Interactive Input Sliders
st.markdown("### Adjust Shipment Variables")
col1, col2, col3 = st.columns(3)
with col1:
    user_weight = st.slider("Product Weight (g)", 1000, 8000, 3000, step=100)
with col2:
    user_discount = st.slider("Discount %", 0, 60, 31, step=1)
with col3:
    user_calls = st.slider("Care Calls", 1, 7, 3, step=1)

# 4. Real-Time Prediction Logic
predicted_boundary = 48 - (user_weight * 0.0055)
is_late = user_discount > predicted_boundary

prediction_text = "Late" if is_late else "On-Time"
prediction_color = "#22c55e" if is_late else "#3b82f6"

# Calculate a mock confidence score based on distance from the boundary
distance = abs(user_discount - predicted_boundary)
confidence = min(99, max(50, int(50 + (distance * 2))))

# 5. Display Prediction Outputs
st.markdown(
    f"""
    <div style='text-align: center; padding: 20px;'>
<h3 style='margin:0;'>PREDICTION: <span style='color:{prediction_color}'>{prediction_text}</span>
</h3>
<p style='margin:0; font-size: 18px; color: #ccc;'>CONFIDENCE: {confidence}%</p>
</div>
    """,
    unsafe_allow_html=True
)

# 6. Plotly Interactive Chart
fig = go.Figure()

# Background Points: On-Time (Blue)
on_time = df[df['Status'] == 'On-Time']
fig.add_trace(go.Scatter(x=on_time['Weight'], y=on_time['Discount'], mode='markers',
                         name='On-Time', marker=dict(color='#3b82f6', size=8, opacity=0.6)))

# Background Points: Late (Green)
late = df[df['Status'] == 'Late']
fig.add_trace(go.Scatter(x=late['Weight'], y=late['Discount'], mode='markers',
                         name='Late', marker=dict(color='#22c55e', size=8, opacity=0.6)))

# Decision Boundary Line (Dashed White)
x_line = np.array([1000, 8000])
y_line = 48 - (x_line * 0.0055)
fig.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines', name='Decision Boundary',
                         line=dict(color='white', width=2, dash='dash')))

# The User's Current Input Point (Large Highlighted Circle)
fig.add_trace(go.Scatter(x=[user_weight], y=[user_discount], mode='markers', name='Current Input',
                         marker=dict(color=prediction_color, size=20, line=dict(color='white', width=3))))

# Chart Formatting
fig.update_layout(
    xaxis_title="Product Weight (g)",
    yaxis_title="Discount Offered (%)",
    plot_bgcolor='#0e1117',  # Streamlit Dark Mode Background
    paper_bgcolor='#0e1117',
    font=dict(color='white'),
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)
