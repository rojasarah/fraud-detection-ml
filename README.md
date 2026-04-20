## Fraud Detection using Machine Learning

This project was developed as part of a team, focused on detecting fraudulent transactions using the BAF dataset.

---

### My Contributions
- Designed and implemented the data cleaning pipeline:
  - Outlier detection and handling
  - Feature transformations (Yeo-Johnson, log)
  - Missing value imputation
  - Categorical encoding (One-Hot Encoding)

- Developed and trained the LightGBM model:
  - Model training and hyperparameter tuning
  - Handling imbalanced data (SMOTE / resampling)
  - Threshold selection based on 5% FPR

- Evaluated model performance:
  - Metrics: AUC, Recall, F1-score, Accuracy
  - Confusion matrix analysis
  - Learning curve generation and interpretation

---

### Pipeline
Raw Data → Cleaning → Feature Processing → Model → Evaluation

---

### Model Used
- LightGBM

---

### Key Challenge
Fraud detection is a highly imbalanced problem (~1% fraud cases), requiring careful evaluation beyond accuracy.

---

### Results
- Metrics: AUC, Recall, Precision
- Observed trade-offs between recall and false positives

---

### Original Team Repository
(https://github.com/mrcolor-blind/ML_ForFraudDetection)
