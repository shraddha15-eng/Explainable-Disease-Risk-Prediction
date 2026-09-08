# Explainable Disease Risk Prediction Using XGBoost and SHAP

An explainable machine-learning system for predicting heart disease risk using clinical features, XGBoost, and SHAP-based individual prediction explanations.

> **Research / Educational Prototype:** This project is intended for research and educational purposes. It is not a medical diagnostic system and should not replace professional clinical evaluation.

---

## 📌 Overview

Machine-learning models can achieve strong predictive performance in healthcare applications, but their decisions are often difficult to interpret.

This project develops an **Explainable Machine Learning (XAI) pipeline** for heart disease risk prediction. The system combines:

* Data preprocessing
* Exploratory data analysis
* Multiple machine-learning models
* Stratified cross-validation
* Randomized hyperparameter optimization
* XGBoost classification
* Probability-based risk prediction
* Brier score evaluation
* SHAP explainability
* Error analysis
* Interactive Streamlit deployment

The final application allows a user to enter clinical characteristics and obtain a predicted probability together with an explanation of which model features contributed to the prediction.

---

## 🎯 Problem Statement

Traditional machine-learning prediction systems may provide a classification result without explaining why the prediction was made.

For healthcare-related applications, interpretability is particularly important because researchers and potential users need to understand the factors influencing a model's prediction.

The objective of this project is therefore to develop a predictive system that combines:

**Predictive Performance + Probability Estimation + Explainability**

rather than relying only on a final classification label.

---

## 🔬 Research Objective

The main objectives are:

1. Develop a machine-learning pipeline for heart disease risk prediction.
2. Compare Logistic Regression, Random Forest, and XGBoost.
3. Evaluate models using stratified cross-validation.
4. Optimize XGBoost using randomized hyperparameter search.
5. Evaluate the final model on a held-out test set.
6. Assess probability quality using the Brier score.
7. Use SHAP to explain individual predictions.
8. Develop an interactive Streamlit prototype.
9. Validate prediction consistency between the research notebook and deployed application.
10. Identify limitations and potential directions for future research.

---

## 📊 Dataset

The dataset contains:

* **303 observations**
* **13 clinical input features**
* Binary target variable

### Input Features

| Feature    | Description                         |
| ---------- | ----------------------------------- |
| `age`      | Age                                 |
| `sex`      | Sex                                 |
| `cp`       | Chest pain type                     |
| `trestbps` | Resting blood pressure              |
| `chol`     | Serum cholesterol                   |
| `fbs`      | Fasting blood sugar                 |
| `restecg`  | Resting electrocardiographic result |
| `thalach`  | Maximum heart rate achieved         |
| `exang`    | Exercise-induced angina             |
| `oldpeak`  | ST depression                       |
| `slope`    | Slope of the ST segment             |
| `ca`       | Number of major vessels             |
| `thal`     | Thalassemia category                |

### Target Distribution

| Class | Samples | Percentage |
| ----- | ------: | ---------: |
| 0     |     164 |     54.13% |
| 1     |     139 |     45.87% |

The target distribution is reasonably balanced, with no severe class imbalance.

---

## 🧠 Methodology

The overall workflow is:

```text
Dataset
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Train/Test Split
   ↓
Preprocessing Pipeline
   ↓
Model Comparison
   ↓
5-Fold Stratified Cross-Validation
   ↓
XGBoost Randomized Hyperparameter Search
   ↓
Final XGBoost Model
   ↓
Test-Set Evaluation
   ↓
Probability Calibration Assessment
   ↓
SHAP Explainability
   ↓
Streamlit Prototype
   ↓
Deployment Validation
```

---

## ⚙️ Data Preprocessing

A unified preprocessing pipeline was created to ensure that the same transformations are applied during training and inference.

### Numerical Features

The numerical variables are processed using:

```text
Missing-value imputation
        ↓
Median
        ↓
StandardScaler
```

Numerical features:

```text
age
trestbps
chol
thalach
oldpeak
```

### Categorical Features

Categorical variables are processed using:

```text
Missing-value imputation
        ↓
Most frequent value
        ↓
One-Hot Encoding
```

Categorical features:

```text
sex
cp
fbs
restecg
exang
slope
ca
thal
```

The complete preprocessing pipeline is saved together with the final XGBoost model to maintain consistency between training and deployment.

---

## 🤖 Model Comparison

Three classification approaches were evaluated:

* Logistic Regression
* Random Forest
* XGBoost

### Cross-Validation Results

Five-fold stratified cross-validation was used.

| Model               | Accuracy | Precision | Recall |     F1 | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -----: | ------: |
| Logistic Regression |   85.47% |    87.44% | 79.79% | 83.32% |  91.88% |
| Random Forest       |   84.17% |    85.60% | 79.15% | 82.14% |  91.56% |
| XGBoost             |   82.52% |    83.68% | 77.67% | 80.35% |  90.00% |

An important observation is that **Logistic Regression achieved the strongest average cross-validation performance among the evaluated configurations**.

XGBoost was nevertheless retained as the final model for the explainable prototype because of its strong held-out test performance and compatibility with tree-based SHAP explanations.

---

## 🔧 Hyperparameter Optimization

XGBoost was optimized using:

```text
RandomizedSearchCV
```

with:

```text
5-Fold Stratified Cross-Validation
Scoring Metric: ROC-AUC
Random State: 42
```

### Selected Parameters

| Parameter               | Value |
| ----------------------- | ----: |
| Learning rate           |  0.01 |
| Number of estimators    |   500 |
| Maximum depth           |     3 |
| Minimum child weight    |     5 |
| Gamma                   |   0.1 |
| Subsample               |   0.7 |
| Column sampling by tree |   0.7 |

Best cross-validation ROC-AUC obtained during the randomized search:

**0.9151**

---

## 📈 Final Model Evaluation

The optimized XGBoost model was evaluated on a separate held-out test set containing:

**61 observations**

### Test Results

| Metric      |      Score |
| ----------- | ---------: |
| Accuracy    | **93.44%** |
| Precision   | **90.00%** |
| Recall      | **96.43%** |
| F1-score    | **93.10%** |
| ROC-AUC     | **98.38%** |
| Brier Score | **0.0587** |

### Confusion Matrix

```text
[[30   3]
 [ 1  27]]
```

This corresponds to:

```text
True Negatives  = 30
False Positives = 3
False Negatives = 1
True Positives  = 27
```

The model correctly classified **57 of 61 test observations**.

---

## 📊 Probability Evaluation

Unlike a simple classification system, this project also evaluates predicted probabilities.

The final model achieved a:

**Brier Score = 0.0587**

The Brier score provides an additional assessment of probabilistic prediction quality by measuring the difference between predicted probabilities and observed outcomes.

---

## 🔎 Explainable AI with SHAP

To improve model interpretability, **SHAP (SHapley Additive exPlanations)** is used.

SHAP helps identify how individual features influence a particular prediction.

The application provides:

* Individual feature contributions
* SHAP values
* Feature importance ranking
* Positive contributions toward predicted risk
* Negative contributions toward predicted risk

The underlying model and preprocessing pipeline remain unchanged during explanation.

---

## 🩺 Interactive Streamlit Application

The project includes an interactive Streamlit interface.

The application allows users to enter clinical information such as:

* Age
* Sex
* Chest pain type
* Resting blood pressure
* Cholesterol
* Maximum heart rate
* Exercise-induced angina
* ST depression
* ST segment slope
* Number of major vessels
* Thalassemia category
* Other clinical indicators

The application then provides:

```text
Clinical Input
      ↓
Saved Preprocessing Pipeline
      ↓
XGBoost Model
      ↓
Prediction Probability
      ↓
Risk Interpretation
      ↓
Individual SHAP Explanation
```

---

## ✅ Deployment Validation

The deployed application was validated against the original research notebook.

For the same test patient:

```text
Notebook probability : 22.07%
Streamlit probability: 22.07%

Prediction matches   : True
Probability matches  : True
SHAP values consistent: True
```

This confirms that the Streamlit application uses the same preprocessing and model logic as the research pipeline.

---

## 🔬 Research Integrity & Leakage Check

The dataset was divided into:

```text
Training: 242 observations (79.87%)
Testing : 61 observations (20.13%)
```

The train and test sets contained:

```text
Overlapping rows = 0
```

This provides evidence that there was no direct row overlap between the training and held-out test observations.

Hyperparameter optimization was performed using cross-validation on the training data rather than using the held-out test set for tuning.

---

## ⚠️ Limitations

This project has several important limitations.

### 1. Small Dataset

The dataset contains only **303 observations**, which limits the statistical generalizability of the findings.

### 2. No External Validation

The final model has not yet been validated on an independent external clinical dataset.

### 3. Test-Set Variability

The held-out test set contains only 61 observations. Therefore, the reported test metrics may vary considerably with a different train/test split.

### 4. Cross-Validation vs. Test Performance

The cross-validation results were lower than the final held-out test performance. This difference highlights the importance of avoiding conclusions based on a single test split.

### 5. Clinical Applicability

The model has not undergone clinical validation, prospective evaluation, regulatory assessment, or deployment in a real healthcare environment.

### 6. Dataset Representation

The dataset may not fully represent the diversity of real-world patient populations.

### 7. Explainability Limitations

SHAP explains the behavior of the trained machine-learning model. It should not be interpreted as establishing clinical causality.

---

## 🚀 Future Research Directions

Potential future improvements include:

* External validation using larger clinical datasets
* Temporal validation
* Improved probability calibration
* Fairness and subgroup analysis
* Robustness testing
* Prospective validation
* Explainable ensemble models
* Uncertainty estimation
* Comparison with additional machine-learning methods
* Evaluation on geographically diverse populations
* Clinical expert assessment of explanations

---

## 📁 Project Structure

```text
Explainable-Disease-Risk-Prediction/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   └── heart_disease_xgboost_pipeline.pkl
│
├── notebooks/
│   └── disease_risk_prediction.ipynb
│
└── reports/
    ├── figures/
    └── results/
```

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* SHAP
* Matplotlib
* Joblib
* Streamlit
* Google Colab
* GitHub

---

## ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Explainable-Disease-Risk-Prediction.git
```

Move into the project directory:

```bash
cd Explainable-Disease-Risk-Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Streamlit Application

Run:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📓 Research Notebook

The complete research workflow is contained in:

```text
notebooks/disease_risk_prediction.ipynb
```

The notebook includes the end-to-end experimental workflow, including:

* Data preparation
* Exploratory analysis
* Preprocessing
* Model development
* Cross-validation
* Hyperparameter optimization
* Evaluation
* SHAP analysis
* Validation

---

## 🔐 Security

API keys, authentication tokens, passwords, and other credentials are **not included in this repository**.

Temporary deployment credentials such as ngrok authentication tokens should never be committed to a public repository.

---

## 📌 Disclaimer

This project is a **research and educational prototype**.

It does not provide medical advice, diagnosis, or treatment recommendations. Model predictions should not be used as a substitute for evaluation by a qualified healthcare professional.

---

## 👩‍💻 Author

**Shraddha Pal**

B.Tech — Computer Science & Information Technology

Interested in:

* Machine Learning
* Explainable AI
* Data Analytics
* Research Applications

---

## ⭐ Project Highlights

```text
303 clinical observations
13 input features
3 ML models compared
5-fold stratified cross-validation
Randomized hyperparameter optimization
XGBoost final prototype
93.44% test accuracy
98.38% test ROC-AUC
0.0587 Brier score
SHAP individual explanations
Streamlit interactive application
Notebook-to-application consistency validation
```

---

## 📜 License

This project is intended for educational and research purposes.
