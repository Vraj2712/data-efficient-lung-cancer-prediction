
# 🫁 Lung Cancer Risk Prediction

## Data Efficiency & Stability Analysis of a Super-Stacking Ensemble

<!-- 🔰 Project Badges -->

<p align="center">

<a href="https://www.python.org/">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue" />
</a>

<img src="https://img.shields.io/badge/Task-Binary%20Classification-purple" />

<img src="https://img.shields.io/badge/Data-Clinical%20%7C%20Tabular-orange" />

<img src="https://img.shields.io/badge/Model-Super%20Stacking%20Ensemble-indigo" />

<img src="https://img.shields.io/badge/Domain-Healthcare%20AI-red" />

<img src="https://img.shields.io/badge/Dataset-PLCO-brightgreen" />

<img src="https://img.shields.io/badge/Metrics-AUC--ROC%20%7C%20AUC--PR-yellow" />

<img src="https://img.shields.io/badge/Stability-Data%20Efficiency%20Study-teal" />

<a href="https://opensource.org/licenses/MIT">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" />
</a>

<img src="https://img.shields.io/github/stars/Vraj2712/lung-cancer-risk-ml-PLCO?style=social" />

</p>



---

## 📌 Project Summary

This project evaluates the **data efficiency, robustness, and stability** of a **Super-Stacking Ensemble** for **lung cancer risk prediction** using **15 clinically interpretable variables** from the **PLCO Cancer Screening Trial**.

Most ML projects report only peak performance. This study answers a deployment-critical question:

> **How much training data does the model need to remain reliable?**

To test this, the model is trained on **progressively smaller fractions of the training set (100% → 10%)**, while keeping a **fixed, frozen test set** to ensure a fair and leakage-free comparison.

---

## 🧠 Model Architecture

A **Super-Stacking Ensemble** is used to combine diverse model families and improve generalization under class imbalance.

### Base Learners

* Logistic Regression
* Random Forest
* Gradient Boosting
* XGBoost
* LightGBM
* CatBoost

### Meta-Learner

* Logistic Regression (**probability stacking**)

### Key Design Choices

* Strong regularization across learners (bias–variance control)
* Class imbalance handled via `class_weight` and `scale_pos_weight`
* Decision threshold **tuned only on validation set** (F1-maximization)
* Test set remains **frozen across all experiments** (clean generalization evaluation)

---

## 🧪 Experimental Design

| Component          | Description                               |
| ------------------ | ----------------------------------------- |
| Dataset            | Lung cancer classification (15 variables) |
| Test split         | 20% (stratified, fixed once)              |
| Training fractions | 1.0 → 0.1                                 |
| Random seeds       | 5 per fraction                            |
| Threshold tuning   | Validation set only                       |
| Metrics            | AUC-ROC, AUC-PR, Recall, Precision, F1    |
| Reporting          | Mean ± standard deviation                 |

### Why this design is reliable

* ✅ No data leakage
* ✅ No optimistic bias from test-time tuning
* ✅ Multi-seed results for stability and reproducibility
* ✅ Measures both performance **and robustness** under reduced data

---

## 📊 Main Results

**Frozen test-set performance (mean ± std across 5 seeds)**

| Training Fraction | Runs | AUC-ROC         | AUC-PR          | Recall          | Precision       | F1              |
| ----------------- | ---- | --------------- | --------------- | --------------- | --------------- | --------------- |
| 1.0               | 5    | 0.8360 ± 0.0004 | 0.1175 ± 0.0007 | 0.8760 ± 0.0022 | 0.0545 ± 0.0006 | 0.1025 ± 0.0010 |
| 0.9               | 5    | 0.8361 ± 0.0004 | 0.1182 ± 0.0008 | 0.8757 ± 0.0036 | 0.0548 ± 0.0005 | 0.1031 ± 0.0009 |
| 0.8               | 5    | 0.8359 ± 0.0012 | 0.1189 ± 0.0015 | 0.8749 ± 0.0024 | 0.0543 ± 0.0007 | 0.1023 ± 0.0013 |
| 0.7               | 5    | 0.8357 ± 0.0006 | 0.1155 ± 0.0026 | 0.8760 ± 0.0028 | 0.0544 ± 0.0004 | 0.1025 ± 0.0007 |
| 0.6               | 5    | 0.8357 ± 0.0007 | 0.1162 ± 0.0030 | 0.8768 ± 0.0024 | 0.0547 ± 0.0005 | 0.1029 ± 0.0009 |
| 0.5               | 5    | 0.8354 ± 0.0017 | 0.1157 ± 0.0020 | 0.8765 ± 0.0041 | 0.0547 ± 0.0010 | 0.1030 ± 0.0017 |
| 0.4               | 5    | 0.8356 ± 0.0008 | 0.1161 ± 0.0027 | 0.8811 ± 0.0049 | 0.0536 ± 0.0011 | 0.1011 ± 0.0019 |
| 0.3               | 5    | 0.8337 ± 0.0016 | 0.1103 ± 0.0039 | 0.8808 ± 0.0035 | 0.0531 ± 0.0007 | 0.1001 ± 0.0013 |
| 0.2               | 5    | 0.8331 ± 0.0010 | 0.1090 ± 0.0033 | 0.8757 ± 0.0103 | 0.0540 ± 0.0017 | 0.1018 ± 0.0029 |
| 0.1               | 5    | 0.8311 ± 0.0035 | 0.1085 ± 0.0053 | 0.8808 ± 0.0092 | 0.0520 ± 0.0028 | 0.0982 ± 0.0049 |

✅ **Key takeaway:** Even at **10% training data**, AUC-ROC drops by only ~**0.005**, showing strong data efficiency and robust ranking performance.

---

## 📈 Visual Analysis & Interpretation

### 1️⃣ Data Efficiency: AUC-ROC & AUC-PR

![Data Efficiency: AUC-ROC and AUC-PR](figures/gh_dual_aucroc_aucpr.png)

**What this shows**
How ranking quality (AUC-ROC) and rare-event performance (AUC-PR) change as training data decreases.

**Key observations**

* AUC-ROC stays **nearly flat** from 100% → ~40%
* At **10% training data**, the drop is small and smooth
* AUC-PR declines gradually (expected under extreme imbalance)

**Interpretation**
The model learns strong generalizable patterns early, remaining **highly data-efficient**.

---

### 2️⃣ Performance Degradation vs Full Training Data

![Performance Degradation](figures/gh_drop_vs_full.png)

**What this shows**
Absolute metric drop compared to 100% training.

**Key observations**

* AUC-ROC drop ≈ **0.005** at 10%
* F1 and AUC-PR decline smoothly without sudden collapse
* Recall remains consistently high across fractions

**Interpretation**
Performance degradation is **controlled**, suggesting robustness rather than overfitting.

---

### 3️⃣ Stability Across Random Seeds

![Stability Across Seeds](figures/gh_stability_std.png)

**What this shows**
Standard deviation across 5 seeds at each training fraction.

**Key observations**

* Variance increases gradually with less data (expected)
* No instability spikes or collapse
* AUC-ROC remains the most stable metric

**Interpretation**
Results are **repeatable and consistent**, indicating low sensitivity to sampling variation.

---

### 4️⃣ Threshold Stability

![Threshold Stability](figures/threshold_stability.png)

**What this shows**
Optimal validation-tuned threshold (F1-max) across training fractions.

**Key observations**

* Threshold stays near **0.30** across data sizes
* Slight variance increase at very low fractions
* No erratic shifts in decision boundary

**Interpretation**
The model’s decision behavior is **stable**, supporting reliable deployment settings.

---

## 🧠 Key Takeaways (Plain Language)

* The model remains strong even with **90% less training data**
* Generalization is consistent across different random samples
* Threshold behavior is stable (predictable decision boundary)
* This is the kind of reliability needed for **real-world screening systems**

---

## 🧰 Tech Stack

* Python
* scikit-learn
* XGBoost, LightGBM, CatBoost
* pandas, NumPy
* Matplotlib

---

## 🚀 Why This Matters

Healthcare data is often limited, imbalanced, and noisy.
This project demonstrates a model that is not only accurate, but also:

✅ **data-efficient**
✅ **stable**
✅ **robust under scarcity**
That combination is what makes ML deployable in the real world.

---

## 👤 Author

**Vraj Shaileshbhai Patel**
MSCS — NJIT

---

## 🏛️ Data Access

PLCO data available from:

> [https://cdas.cancer.gov/plco/](https://cdas.cancer.gov/plco/)

Use requires approval — therefore **not shared in this repo**.
A **fake sample CSV** is included for demonstration only.

---

## 🛠️ Requirements

Install Python libs:

```
pip install -r requirements.txt
```

---

## 🙌 Acknowledgements

* PLCO Cancer Screening Trial participants
* NIH / NCI CDAS program
* Advisors and faculty who guided methodology

---

