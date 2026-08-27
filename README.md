# Credit Card Fraud Detection: Balancing ML Metrics with Business ROI

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit_Learn-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/Library-XGBoost-green.svg)](https://xgboost.ai/)
[![SHAP](https://img.shields.io/badge/Library-SHAP-red.svg)](https://shap.readthedocs.io/en/latest/)

## Project Overview
This repository contains the source code and research for my Bachelor's thesis on Detecting Fraudulent Financial Transactions using Machine Learning.

The main challenge of this project is the extreme class imbalance, where frauds account for only 0.172% of all transactions. Rather than chasing abstract mathematical metrics like Accuracy or ROC-AUC (which are misleading in this context), this project focuses on optimizing business metrics. The goal is to minimize financial losses from fraud while drastically reducing False Positives (unnecessary card blocks) to preserve customer loyalty and save operational costs.

Note: The complete academic paper (in Russian) detailing the mathematical models, methodology, and economic justification is available in the `docs/` folder.

## Key Findings & Business Impact

### 1. Feature Engineering
Raw transaction data (Time and Amount) was heavily skewed. Applying trigonometric cyclic encoding for time (Time_sin, Time_cos) and logarithmic transformation for transaction amounts significantly improved the models' ability to detect cyclical nighttime attack patterns and micro-transactions. SHAP analysis confirmed these constructed features were among the top predictors.

### 2. Solving Extreme Imbalance (0.172%)
I compared data-level approaches (SMOTE) with algorithmic penalty weights (Class Weights).
* LightGBM suffered from "probability calibration collapse" under extreme class weights, making SMOTE mandatory for this specific architecture.
* XGBoost + Class Weights proved to be the most robust and computationally efficient (approximately 2x faster than SMOTE with better precision).

### 3. Threshold Optimization & Business ROI
Default decision thresholds (p=0.5) yielded too many False Positives (~1,468 false alarms). By performing dynamic threshold optimization maximizing the F1-score (optimal p=0.912), the model achieved:
* Precision surge from 90% to 96.39%.
* A 3x reduction in False Positives, massively reducing the load on the manual verification call center.
* An economic ROI of 28,224% (calculated using a custom cost matrix). Every $1 spent on manual verification of false alarms saves ~$282 of customer funds.

### 4. Model Explainability
Using SHAP (SHapley Additive exPlanations), the XGBoost ensemble was fully interpreted. Analysts can see exactly which features triggered a block. For example, anomalously low values of the V14 feature were identified as the strongest fraud trigger, ensuring the model acts as a transparent tool rather than a "black box".

## Tech Stack
* Data Processing & Engineering: pandas, numpy, scikit-learn
* Handling Imbalance: imbalanced-learn (SMOTE), Algorithmic Class Weights
* Machine Learning: XGBoost, CatBoost, LightGBM, Logistic Regression
* Interpretability: SHAP
* Visualization: matplotlib, seaborn
* Environment Management: uv

## Repository Structure

```text
├── README.md               <- Project overview
├── data/                   <- Folder for dataset 
├── docs/
│   └── diploma.pdf         <- Full thesis document
├── notebooks/
│   ├── 01_EDA_and_Feature_Engineering.ipynb
│   └── 02_Model_Training_and_Evaluation.ipynb
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   └── visualisation.py    <- SHAP and PR-AUC plotting scripts
├── pyproject.toml          <- uv project configuration
└── uv.lock                 <- Lockfile for exact dependency versions
```

## How to Run (Using uv)

This project uses [uv](https://github.com/astral-sh/uv), a fast Python package and project manager.

1. Clone the repository:
   ```bash
   git clone https://github.com/CuteZerg/diploma.git
   cd diploma
   ```

2. Install uv (if not already installed):
   ```bash
   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Windows
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

3. Install dependencies and sync environment:
   ```bash
   uv sync
   ```
   This will automatically create a `.venv` and install all dependencies from `uv.lock`.

4. Download the dataset:
   Download the Credit Card Fraud Detection dataset from Kaggle (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place creditcard.csv in the data/ folder.


5. Run the notebooks:
   To run Jupyter Lab using the managed environment:
   ```bash
   uv run jupyter lab
   ```

## Author
**Alexey Krasnikov**
* [GitHub](https://github.com/CuteZerg)
* [Telegram](https://t.me/xCuteZerGx)
