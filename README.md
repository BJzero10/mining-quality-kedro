# Mining Process Quality Prediction with Kedro

## 📌 Overview
This project implements an **end-to-end, reproducible Machine Learning pipeline** to predict **silica concentration in a mining flotation process**, using **Kedro** as the orchestration framework.

The solution is based on the Kaggle dataset **“Quality Prediction in a Mining Process”** and refactors a traditional notebook-based approach into a **production-grade data pipeline**, including:
- Data ingestion and cleaning
- Time-based resampling
- Temporal train/test split
- XGBoost regression model
- Model evaluation
- SHAP explainability

The goal is not only prediction accuracy, but **traceability, reproducibility, and interpretability**, following industry best practices.

---

## 🏭 Business Context
In mineral processing plants, **silica concentration in the concentrate** is a critical quality indicator:
- High silica reduces concentrate quality
- Impacts downstream processing and costs
- Requires continuous monitoring and control

This project demonstrates how historical sensor data can be leveraged to **predict quality deviations** and **understand which process variables drive them**.

---

## 📊 Dataset
- Source: Kaggle – *Quality Prediction in a Mining Process*
- Sampling rate: ~20 seconds
- Size: ~737,000 rows
- Features: process sensors (flows, pressures, reagents, air flow, etc.)
- Target variable: % Silica Concentrate


Raw data is **not stored in the repository** to keep it lightweight and reproducible.

---

## 🧱 Project Architecture (Kedro)

The project follows Kedro’s standard structure:

```text
mining-quality-kedro/
├── conf/
│   ├── base/
│   │   ├── catalog.yml
│   │   └── parameters.yml
├── src/
│   └── mining_quality_kedro/
│       ├── pipelines/
│       │   └── mining_quality/
│       │       ├── nodes.py
│       │       └── pipeline.py
│       └── pipeline_registry.py
├── pyproject.toml
└── README.md
```

---

## Pipeline Description

### 1. Data Cleaning
- Datetime parsing
- Conversion of decimal-comma values (e.g. `55,3 → 55.3`)
- Duplicate and invalid row removal

### 2. Time Resampling
- Sensor data resampled to **hourly averages**
- Noise reduction and operational alignment

### 3. Temporal Train/Test Split
- Split performed strictly by time
- No shuffling, preserving real forecasting conditions

### 4. Model Training
- XGBoost Regressor
- Robust to non-linear relationships and sensor noise

### 5. Model Evaluation
- RMSE
- MAE
- R² score

### 6. Explainability
- SHAP (SHapley Additive exPlanations)
- Global feature importance visualization
- Interpretability suitable for process engineers

---

## Outputs
After executing the pipeline, the following artifacts are generated locally:
- Cleaned and resampled datasets (Parquet)
- Trained XGBoost model
- Evaluation metrics (`metrics.json`)
- SHAP summary plot (`shap_summary.png`)

These artifacts are excluded from version control.

---

## How to Run

### 1. Create and activate virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install kedro kedro-datasets pandas numpy scikit-learn xgboost shap pyarrow matplotlib
```

### 3. Run the pipeline
```bash
kedro run
```


### 4. Visualize pipeline DAG
```bash
kedro viz
```

## Reference

This project is inspired by a Kaggle solution using XGBoost and SHAP, re-engineered here into a modular, maintainable and reproducible ML pipeline.

## Future Improvements

- MLflow experiment tracking
- Hyperparameter optimization
- Feature selection pipelines
- Real-time inference integration
- Deployment-ready packaging