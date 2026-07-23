# Post-HCT Survival Analysis and Prediction

Comparing five survival modeling approaches to predict outcomes for patients who underwent hematopoietic stem cell transplantation (HCT), alongside an equity-focused analysis of survival disparities by demographic group.

## Problem

HCT is a critical procedure for treating hematological diseases like leukemia and lymphoma, but predicting patient survival outcomes reliably from clinical and demographic data is difficult, and understanding whether survival differs across demographic groups matters for both prediction and fairness in healthcare.

*Originally developed as a team project for a graduate course at Drexel University.*

## Approach

**Data:** 28,800 patient records with 60 clinical and demographic features (comorbidity scores, prior tumors, HLA match levels, conditioning intensity, donor/recipient sex match, race, ethnicity), inspired by a Kaggle competition on post-HCT survival prediction.

**Preprocessing:** Encoded 35 categorical features (ordinal mapping for ordered categories, one-hot encoding for unordered ones) and standardized 25 numerical features. Removed highly correlated features to reduce collinearity before modeling.

**Five models compared:**
1. Kaplan-Meier estimator, for the overall non-parametric survival curve and median survival time
2. Cox Proportional Hazards, with L1 (lasso) regularization to prevent overfitting
3. Random Survival Forest
4. XGBoost with AFT (accelerated failure time) loss, tuned with a grid search over tree depth, learning rate, and regularization
5. DeepSurv, a deep learning Cox model, trained on the top 30 most predictive features selected via XGBoost feature importance
6. A Graph Neural Network, using cosine similarity between patients to construct graph edges

**Evaluation metric:** Concordance index (C-index), which measures whether the model correctly orders patients by predicted risk relative to their actual survival times. 0.5 is random, 1.0 is perfect concordance.

**Equity analysis:** Beyond prediction, examined event-free survival rates broken down by sex-match (donor/recipient) and by race/ethnicity group.

## Results

| Model | C-index |
|---|---|
| Random Survival Forest | 0.63 |
| Cox Proportional Hazards | 0.66 |
| Graph Neural Network | 0.58 |
| DeepSurv | 0.65-0.66 |
| **XGBoost (AFT)** | **0.67** |

Median survival time across the full cohort: 9.98 months.

**Equity finding:** same-sex donor-recipient transplant pairs (M-M and F-F) showed better long-term event-free survival than opposite-sex pairs (M-F, F-M), which showed earlier complication onset in both raw event rates and the Kaplan-Meier survival curve comparison.

## Limitations

- Data is highly right-skewed, with most patients experiencing events within the first two years post-transplant, which makes long-term survival prediction harder to validate
- Models trained on a single dataset with no external validation on a different patient population
- Average C-index across models (~0.66) leaves meaningful room for improvement
- Determining which of the 60 raw features are most clinically meaningful required domain knowledge the team was building from documentation rather than clinical expertise

## Tech Stack

Python, XGBoost, lifelines (CoxPH, Kaplan-Meier), scikit-survival (Random Survival Forest), PyTorch, pycox (DeepSurv), PyTorch Geometric (GNN)

## Repository Structure

```
hct-survival-analysis/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── EDA.ipynb
│   ├── Feature_Handling.ipynb
│   ├── CoxPH_KaplanMeier.ipynb
│   ├── Xgboost_model.ipynb
│   ├── RSF_Model.ipynb
│   ├── GNN_Model.ipynb
│   └── Deep_Model.ipynb
├── data/
│   └── (Kaggle dataset not included; see Data Sources below)
└── outputs/
    └── (model comparison results, feature importance rankings)
```

## Data Sources

Kaggle competition dataset on post-HCT survival prediction (28,800 patient records, 60 features). See `data_dictionary.csv` for full variable descriptions.

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/EDA.ipynb
```

Run `Feature_Handling.ipynb` before any of the model notebooks, since it produces the cleaned, encoded dataset every model depends on.
