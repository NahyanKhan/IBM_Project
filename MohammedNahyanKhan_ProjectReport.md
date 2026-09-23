# Supply Chain Delivery Prediction & Analytics
## Project Report — AICTE Internship on AI/ML

**Author:** Mohammed Nahyan Khan  
**Submission Date:** 2025  
**Dataset:** Kaggle E-Commerce Shipping Data (10,999 records)  
**Application:** MohammedNahyanKhan_SupplyChainAnalytics.py  

---

## 1. Executive Summary

In the e-commerce sector, late shipments are one of the most direct drivers of customer churn, support cost escalation, and margin erosion. Industry estimates place the average cost of a single delayed order — including refunds, re-delivery, and customer service hours — between $10 and $25. At a delay rate of 59.7% across this dataset, a mid-size retailer processing 10,000 orders per month faces potential exposure exceeding $1 million annually.

This project delivers a **production-ready machine learning web application** that addresses this problem along three dimensions:

1. **Predictive Intelligence** — A Random Forest classifier trained on 10,999 historical shipments identifies orders at risk of delay before despatch, achieving **68.9% test-set accuracy** using four operationally observable features.
2. **Local Explainability** — SHAP (SHapley Additive exPlanations) force plots break each prediction down to the individual feature level, giving logistics managers an auditable reason for every flag rather than a black-box output.
3. **Financial Quantification** — An enterprise batch-scoring pipeline accepts daily shipping manifests, predicts delay risk at scale, and surfaces a **Value at Risk (VAR)** metric in dollars — translating model output directly into business language.

The result is a tool that moves machine learning from a notebook experiment to an actionable operations dashboard.

---

## 2. Exploratory Data Analysis

### 2.1 Dataset Overview

The Kaggle E-Commerce Shipping Dataset contains **10,999 shipment records** from a fictional international e-commerce retailer. Each record captures operational attributes recorded at the time of despatch.

| Attribute | Type | Range / Categories |
|---|---|---|
| `Warehouse_block` | Categorical | A, B, C, D, F |
| `Mode_of_Shipment` | Categorical | Flight, Road, Ship |
| `Customer_care_calls` | Integer | 2 – 7 |
| `Customer_rating` | Integer | 1 – 5 |
| `Cost_of_the_Product` | Integer | 96 – 310 |
| `Prior_purchases` | Integer | 2 – 10 |
| `Product_importance` | Categorical | high, low, medium |
| `Discount_offered` | Integer | 1 – 65 (%) |
| `Weight_in_gms` | Integer | 1,001 – 7,846 |
| `Reached.on.Time_Y.N` | Binary target | 0 = On Time, 1 = Late |

**No missing values** were found across any column. The ID column was dropped as it carries no predictive signal.

### 2.2 Key Findings

- **Overall delay rate: 59.7%** — a majority of shipments in this dataset arrive late, confirming the business problem is significant and systemic.
- **Warehouse Block F** processes the largest share of orders (~33%) and shows the highest late-delivery rate (~59.8%), making it the single highest-leverage operational improvement target.
- **Discount skew:** Late orders carry a median discount of 9% versus 6% for on-time orders — a 50% relative difference — suggesting that promotional demand spikes outpace warehouse fulfilment capacity.

![EDA Dashboard — KPI Metrics and Warehouse Delay Histogram](https://encrypted-tbn0.gstatic.com/licensed-image?q=tbn:ANd9GcS03ds2GHOKxjKeg3U3w5axpn_CjfEe7LyY9_1ikMdXJ3wQcuHNh-_02bsCRouc9CizG5gmU17jT_UeJ0A)
*Figure 1: Tab 1 — EDA Dashboard showing KPI tiles and grouped delay histogram by warehouse block.*

---

## 3. Predictive Modelling — Random Forest Classifier

### 3.1 Feature Selection

Four features were selected for the model based on their operational availability at order-creation time and their correlation with the target variable:

| Feature | Rationale |
|---|---|
| `Discount_offered` | Strongest single predictor; high discounts correlate with demand spikes |
| `Weight_in_gms` | Heavier products have longer handling times and carrier constraints |
| `Customer_care_calls` | Leading indicator of order friction and logistics issues |
| `Prior_purchases` | Proxy for customer segment and order complexity |

Categorical columns (`Warehouse_block`, `Mode_of_Shipment`) were label-encoded for compatibility with the tree-based model but were not included in the final feature set, as the four numerical features yielded equivalent accuracy with a simpler, more interpretable model.

### 3.2 Training Configuration

```python
RandomForestClassifier(
    n_estimators = 100,   # 100 decision trees
    max_depth    = 5,     # prevents overfitting on noisy features
    random_state = 42,    # reproducibility
)
train_test_split(test_size=0.2, random_state=42)
# Training set: 8,799 records
# Test set    : 2,200 records
```

### 3.3 Model Performance

| Metric | Value |
|---|---|
| Test Set Accuracy | **68.9%** |
| Baseline (majority class) | 59.7% |
| Lift over baseline | +9.2 percentage points |

A 68.9% accuracy represents a meaningful and deployable improvement over the naive majority-class baseline of 59.7%. In a logistics context, even a modest improvement in precision when flagging high-risk orders enables proactive intervention (expedited carrier upgrades, pre-emptive customer communication) that reduces the financial cost of the remaining errors.

![Model Insights — Feature Importance Chart](https://encrypted-tbn0.gstatic.com/licensed-image?q=tbn:ANd9GcS03ds2GHOKxjKeg3U3w5axpn_CjfEe7LyY9_1ikMdXJ3wQcuHNh-_02bsCRouc9CizG5gmU17jT_UeJ0A)
*Figure 2: Tab 3 — Horizontal bar chart of Random Forest feature importances. `Weight_in_gms` and `Discount_offered` dominate.*

---

## 4. Interactive Simulation & Local Explainability (SHAP)

### 4.1 Real-Time Simulator

The Predictive Simulator tab exposes the trained model through three Streamlit sliders:

- **Product Weight** (1,001 – 7,846 g)
- **Discount Offered** (1 – 65%)
- **Customer Care Calls** (2 – 7)

As sliders are adjusted, the app instantaneously calls `model.predict_proba()` and re-renders the prediction label, confidence percentage, and background scatter plot — with the user's input shown as a large highlighted dot positioned relative to 1,500 historical orders.

### 4.2 SHAP Force Plot — Local Explainability

The SHAP (SHapley Additive exPlanations) framework decomposes each individual prediction into per-feature contributions. A `shap.TreeExplainer` is instantiated against the Random Forest model and applied to the single user-input row on every slider interaction.

The resulting force plot renders:
- A **base value** (the model's average prediction across all training orders: ~0.60)
- **Red arrows** representing features that push the prediction toward Late (class 1)
- **Blue arrows** representing features that push the prediction toward On-Time (class 0)
- The **final prediction value** at the tip of the combined force

This transforms the model from a black box into an auditable decision aid — a logistics manager can see not just *that* an order is flagged, but *which specific attribute* is responsible and by how much.

![Predictive Simulator with SHAP Force Plot](https://encrypted-tbn0.gstatic.com/licensed-image?q=tbn:ANd9GcS03ds2GHOKxjKeg3U3w5axpn_CjfEe7LyY9_1ikMdXJ3wQcuHNh-_02bsCRouc9CizG5gmU17jT_UeJ0A)
*Figure 3: Tab 2 — Simulator showing prediction output and SHAP force plot for a high-risk input (Weight=6,500g, Discount=45%).*

![SHAP Force Plot Detail](https://encrypted-tbn0.gstatic.com/licensed-image?q=tbn:ANd9GcS03ds2GHOKxjKeg3U3w5axpn_CjfEe7LyY9_1ikMdXJ3wQcuHNh-_02bsCRouc9CizG5gmU17jT_UeJ0A)
*Figure 4: SHAP force plot close-up — `Discount_offered=45` and `Weight_in_gms=6500` are the dominant contributors pushing toward a Late prediction.*

---

## 5. Enterprise Batch Processing & ROI

### 5.1 Bulk Scoring Pipeline

The Bulk Scoring & ROI tab elevates the application from a single-prediction simulator to an operational batch tool. A logistics coordinator can:

1. **Download a CSV template** pre-populated with the four required column headers
2. **Upload a daily shipping manifest** containing any number of new orders
3. Receive **instant scored output** with two appended columns:
   - `Predicted_Delay` — "Yes" or "No" per order
   - `Delay_Confidence_%` — the model's probability of delay (0–100%)

Delayed rows are highlighted in red within the interactive table for immediate visual triage. The scored manifest can be exported as a CSV for integration into warehouse management systems.

### 5.2 Value at Risk (VAR) Metric

The application computes a **Value at Risk** estimate using the assumption that each predicted delay carries an expected remediation cost of **$15.00** (covering customer refunds, re-delivery logistics, and incremental support-ticket handling):

```
VAR = Count(Predicted_Delay == 'Yes') × $15.00
```

This metric converts a statistical output into a dollar figure that is immediately meaningful to operations managers and C-suite stakeholders — bridging the gap between data science and business decision-making.

**Example:** A 200-order daily manifest with 60 flagged delays yields a VAR of **$900**, prompting targeted intervention on those 60 orders before they leave the warehouse.

![Bulk Scoring — Upload and Results](https://encrypted-tbn0.gstatic.com/licensed-image?q=tbn:ANd9GcS03ds2GHOKxjKeg3U3w5axpn_CjfEe7LyY9_1ikMdXJ3wQcuHNh-_02bsCRouc9CizG5gmU17jT_UeJ0A)
*Figure 5: Tab 4 — Batch scoring results showing KPI tiles (shipments processed, flagged for delay, VAR in dollars).*

![Bulk Scoring — Scored Manifest Table](https://encrypted-tbn0.gstatic.com/licensed-image?q=tbn:ANd9GcS03ds2GHOKxjKeg3U3w5axpn_CjfEe7LyY9_1ikMdXJ3wQcuHNh-_02bsCRouc9CizG5gmU17jT_UeJ0A)
*Figure 6: Interactive scored dataframe with delayed orders highlighted in red and confidence scores appended per row.*

---

## 6. Final Business Recommendations

Three actionable recommendations emerge directly from the model's feature importances and the EDA findings:

### Recommendation 1 — Enforce Discount Thresholds

**Finding:** `Discount_offered` is the strongest predictor of late delivery. Orders with discounts exceeding 40% show a disproportionately high delay rate.

**Action:** Implement a dynamic discount governance policy:
- Discounts above 35% should trigger automatic inventory pre-staging 48 hours before the promotion goes live.
- Flash sales should be capped at a volume that the lowest-capacity warehouse block can fulfil within the SLA window.
- The Bulk Scoring tool can be run on the expected order volume *before* a promotion launches, giving the operations team a VAR estimate in advance.

### Recommendation 2 — Prioritise Heavy Product Logistics

**Finding:** `Weight_in_gms` is the second-most important feature. Products above ~5,000 g show materially higher delay rates, attributable to carrier weight-tier transitions, longer packing times, and limited vehicle capacity.

**Action:**
- Negotiate dedicated weight-tier carrier contracts for products over 5,000 g.
- Route heavy-product orders exclusively through Warehouse Block A or B (lower observed delay rates) rather than the overloaded Block F.
- Apply a weight-based surcharge model to offset the higher fulfilment cost, improving margin on at-risk SKUs.

### Recommendation 3 — Use Support Call Volume as an Early-Warning System

**Finding:** `Customer_care_calls` correlates with delivery problems. An order that has generated 5 or more pre-delivery support calls has a near-certain delay history in the training data.

**Action:**
- Integrate the predictive model into the CRM system. When an order crosses 3 care calls, an automated alert should be sent to the logistics supervisor.
- Proactively notify affected customers with a revised ETA and a goodwill discount voucher — a strategy that data consistently shows reduces negative reviews even when the delay cannot be prevented.
- Track the call-to-delay conversion rate monthly as a KPI dashboard metric to measure the intervention's effectiveness.

---

## 7. Technical Architecture Summary

```
Train.csv
    │
    ▼
load_data()  ──@st.cache_data──►  df_raw (EDA) + df_encoded (model input)
    │
    ▼
train_model() ──@st.cache_resource──►  RandomForestClassifier + accuracy + feature_names
    │
    ├──► Tab 1: Plotly histogram + scatter (EDA)
    ├──► Tab 2: predict_proba → SHAP TreeExplainer → force_plot (Simulator)
    ├──► Tab 3: feature_importances_ → Plotly bar chart (Insights)
    └──► Tab 4: file_uploader → batch predict → VAR → styled dataframe + download
```

**Deployment:** The `render.yaml` blueprint commits the full infrastructure definition to Git, enabling one-click cloud deployment. `requirements.txt` pins minimum compatible versions of all seven dependencies.

---

*Report prepared by Mohammed Nahyan Khan as part of the AICTE Internship on AI/ML — Supply Chain Data Analytics track.*
