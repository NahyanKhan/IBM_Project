# 📦 Supply Chain Delivery Prediction & Analytics

> **An end-to-end machine learning web application** that predicts e-commerce shipment delays in real time, explains individual predictions with SHAP force plots, and quantifies financial exposure through an enterprise-grade bulk-scoring pipeline.  
> Built as part of the **AICTE Internship on IBM SkillsBuild Data Analytics with AI Academic Internship Program** by **Mohammed Nahyan Khan**.
---

## 📂 Dataset

[Kaggle — E-Commerce Shipping Data](https://www.kaggle.com/datasets/prachi13/customer-analytics)  
10,999 shipment records · 11 features · Binary target: `Reached.on.Time_Y.N`

---

## ✨ Key Features

| Tab | Feature |
|-----|---------|
| **📊 Exploratory Data Analysis** | KPI metrics (total shipments, delay rate, avg discount) · Grouped histogram of delays by warehouse block · Weight vs Discount scatter coloured by delivery status |
| **🎛️ Predictive Simulator** | Real-time Random Forest prediction via 3 sliders · Probability output (P(On-Time) / P(Late)) · Historical scatter with highlighted user input · **SHAP force plot** showing per-feature contribution to the individual prediction |
| **🧠 Model Insights** | Test-set accuracy (68.9%) · Horizontal feature importance bar chart · 3 business takeaways |
| **📂 Bulk Scoring & ROI** | CSV manifest upload with format template · Batch `predict` + `predict_proba` · Flagged-delay count · **Value at Risk (VAR)** in dollars · Downloadable scored manifest with red-highlighted delay rows |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Web Framework | [Streamlit](https://streamlit.io/) |
| Data Manipulation | [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/) |
| Machine Learning | [Scikit-Learn](https://scikit-learn.org/) — `RandomForestClassifier` |
| Visualisation | [Plotly Express](https://plotly.com/python/plotly-express/), [Plotly Graph Objects](https://plotly.com/python/graph-objects/) |
| Explainability | [SHAP](https://shap.readthedocs.io/), [streamlit-shap](https://github.com/snehankekre/streamlit-shap) |
| Deployment | [Render](https://render.com/) (Blueprint via `render.yaml`) |

---

## 🚀 Installation & Execution

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. (Recommended) Create and activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the application
streamlit run MohammedNahyanKhan_SupplyChainAnalytics.py
```

The app will open automatically at `http://localhost:8501`.

> **Note:** `Train.csv` must be present in the project root.  
> If it is missing the app falls back to 1,000 rows of synthetic data automatically.

---

## 📁 Project Structure

```
├── MohammedNahyanKhan_SupplyChainAnalytics.py   # Main Streamlit application
├── MohammedNahyanKhan_Simulator.py              # Standalone decision-boundary simulator
├── supply_chain_eda.ipynb                       # Jupyter EDA notebook
├── Train.csv                                    # Kaggle dataset
├── requirements.txt                             # Python dependencies
├── render.yaml                                  # Render cloud deployment blueprint
└── README.md                                    # This file
```

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Algorithm | Random Forest Classifier |
| Estimators | 100 trees, max depth 5 |
| Train / Test Split | 80% / 20% |
| Test Set Accuracy | **68.9%** |
| Training Records | 10,999 shipments |

---

## 💡 Key Business Insights

1. **Discount Thresholds** — High discounts are the strongest predictor of delayed shipments; promotions should be staggered to prevent fulfilment surges.  
2. **Heavy Product Handling** — Product weight significantly shifts delay probability; priority carrier contracts for heavy SKUs can close this gap.  
3. **Support Call Leading Indicator** — High customer care call volumes correlate strongly with problematic logistics routes and can be used as an early-warning signal.

---

## 👤 Author

**Mohammed Nahyan Khan**  
AICTE Internship on IBM SkillsBuild Data Analytics with AI Academic Internship Program — Supply Chain Data Analytics  
