# Take-Home-Assignment

## Fraud Detection Case – Proof of Concept

This repository contains my submission for the **Snappt Data Scientist Take-Home Assignment**.  
The objective was to build a small end-to-end fraud detection **proof-of-concept (POC)** and clearly communicate the approach within ~2 hours of work.

---

##  Repository Structure
- **`Snappt Technical project test.ipynb`** – Main Jupyter Notebook with:
  - Data exploration & preprocessing
  - Feature inspection (mutual information, feature importance)
  - Model training (RandomForest, XGBoost, Logistic Regression baseline)
  - Cross-validation and test evaluation (ROC AUC, Average Precision, PR/ROC curves)
  - Sanity checks for leakage (shuffled labels test)
  - Threshold analysis for actionable precision/recall trade-offs
- **`writeup.pdf`** – One-page summary covering:
  - Model choice rationale
  - Key results and trade-offs
  - Next steps for deployment and monitoring
- **`requirements.txt`** – Dependencies to reproduce the notebook.

---

##  Approach
1. **Data Handling**
   - Stratified train/test split (75/25).
   - Median imputation and simple scaling when required.
   - Target encoding to {0,1}.
   - Exploratory feature ranking (Mutual Information, RandomForest importance).

2. **Modeling**
   - **RandomForestClassifier** with class balancing.
   - **XGBoostClassifier** with `scale_pos_weight` from training data.
   - **Logistic Regression** baseline for linear separability check.

3. **Evaluation**
   - 5-fold **Stratified CV** with ROC AUC and Average Precision.
   - Final holdout test evaluation with ROC/PR curves.
   - **Sanity checks**: shuffled-labels dropped performance to chance → pipeline confirmed leak-free.
   - Threshold calibration functions to report precision at recall ≥0.80 and ≥0.90.

4. **Results**
   - All models achieved ROC AUC ≈ 1.0 and AP ≈ 1.0 due to highly separable synthetic features.
   - Logistic Regression baseline also ≈ 1.0 → confirms dataset is trivial to separate.
   - Key business insight: perfect performance here won’t generalize; monitoring and threshold calibration are critical in production.

5. **Deployment & Monitoring (outline)**
   - Package preprocessing + model as a single artifact (`joblib`).
   - Serve via **FastAPI** in Docker with `/predict` and `/health` endpoints.
   - Monitor:
     - Data & prediction drift (Evidently)
     - Delayed performance (AP, precision@recall)
   - Threshold configurable via environment variable.
   - Model versioning with MLflow, rollback via blue/green deployments.
   - Retraining triggered by drift or metric degradation.

---

##  Trade-offs
- Chose **RandomForest** for simplicity and interpretability vs. XGBoost for scalability.
- Avoided complex feature engineering due to small dataset & POC scope.
- Focused on clarity, correctness, and reproducibility over hyperparameter tuning.

---

## 📈 Next Steps
- Test robustness on more realistic datasets with overlapping fraud/clean distributions.
- Extend explainability with SHAP values and partial dependence plots.
- Implement live monitoring pipelines for drift and precision@recall metrics.
- Build retraining workflow integrated with CI/CD.

---

##  Running the Notebook
```bash
# Clone repository
git clone https://github.com/<your-username>/snappt-fraud-detection-poc.git
cd snappt-fraud-detection-poc

# Create environment & install requirements
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start Jupyter
jupyter notebook
