---
hide:
  - toc
---

# Chapter 11: Model Evaluation Metrics

> *"The map is not the territory, but metrics are our compass in the landscape of model performance."*

---

## Learning Objectives

By the end of this chapter, you will be able to:

- Understand the purpose and limitations of common evaluation metrics for classification models
- Compute and interpret accuracy, precision, recall, F1-score, and confusion matrix
- Generate and analyze Receiver Operating Characteristic (ROC) and Precision-Recall (PR) curves
- Identify situations where different metrics provide conflicting information and choose appropriate metrics based on the problem context

---

## Intuitive Introduction

Imagine you're building a spam email classifier. Your model correctly identifies 95% of emails as spam or not spam. Sounds great, right? But what if 99% of emails are not spam? Your model could achieve 95% accuracy by simply labeling everything as "not spam"—missing all the spam! This highlights why accuracy alone is insufficient.

In real-world applications, the cost of different types of errors varies. In medical diagnosis, missing a disease (false negative) might be more critical than a false alarm (false positive). In fraud detection, incorrectly flagging legitimate transactions (false positive) could annoy customers, while missing fraud (false negative) could lead to financial loss.

This chapter explores evaluation metrics that go beyond simple accuracy, providing a nuanced view of model performance. We'll start with basic metrics, then move to more sophisticated curve-based evaluations that help us understand trade-offs between different types of errors.

---

## Mathematical Development

Classification models make predictions that can be categorized into four outcomes relative to the true labels:

- **True Positive (TP)**: Correctly predicted positive class
- **True Negative (TN)**: Correctly predicted negative class  
- **False Positive (FP)**: Incorrectly predicted positive class (Type I error)
- **False Negative (FN)**: Incorrectly predicted negative class (Type II error)

These form the foundation of most evaluation metrics.

### Basic Metrics

**Accuracy** measures the overall correctness of predictions:

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

While straightforward, accuracy can be misleading in imbalanced datasets where one class dominates.

**Precision** (also called Positive Predictive Value) measures the accuracy of positive predictions:

$$\text{Precision} = \frac{TP}{TP + FP}$$

Precision answers: "Of all instances predicted as positive, how many were actually positive?"

**Recall** (also called Sensitivity or True Positive Rate) measures the model's ability to find all positive instances:

$$\text{Recall} = \frac{TP}{TP + FN}$$

Recall answers: "Of all actual positive instances, how many did we correctly identify?"

**F1-Score** provides a balanced measure combining precision and recall:

$$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

The F1-score is the harmonic mean of precision and recall, giving equal weight to both metrics.

### Confusion Matrix

The confusion matrix organizes these four outcomes into a table:

|                  | Predicted Negative | Predicted Positive |
|------------------|--------------------|--------------------|
| **Actual Negative** | True Negative (TN) | False Positive (FP) |
| **Actual Positive** | False Negative (FN) | True Positive (TP) |

This matrix provides a complete picture of model performance across all prediction outcomes.

### Curve-Based Metrics

**Receiver Operating Characteristic (ROC) Curve** plots the True Positive Rate (TPR) against the False Positive Rate (FPR) at various threshold settings:

$$\text{TPR} = \frac{TP}{TP + FN} = \text{Recall}$$

$$\text{FPR} = \frac{FP}{FP + TN}$$

The **Area Under the ROC Curve (AUC-ROC)** summarizes the ROC curve's performance. An AUC of 1.0 represents perfect classification, while 0.5 represents random guessing.

**Precision-Recall (PR) Curve** plots precision against recall at different thresholds. The **Area Under the PR Curve (AUC-PR)** provides a summary measure, particularly useful for imbalanced datasets.

For web sources on these metrics, see:
- Scikit-learn documentation: [https://scikit-learn.org/stable/modules/model_evaluation.html](https://scikit-learn.org/stable/modules/model_evaluation.html)
- Wikipedia articles on precision and recall, ROC curves

---

## Implementation Guide

Scikit-learn provides comprehensive tools for computing these metrics through the `sklearn.metrics` module. Let's explore the key functions:

### Basic Metrics

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Assuming y_true and y_pred are your true labels and predictions
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# For multi-class problems, specify average method
precision_macro = precision_score(y_true, y_pred, average='macro')
recall_micro = recall_score(y_true, y_pred, average='micro')
```

**Parameter Explanations:**

- `average`: For multi-class problems
  - `'macro'`: Calculate metrics for each class and average (equal weight)
  - `'micro'`: Calculate metrics globally by counting total TP, FP, FN
  - `'weighted'`: Average weighted by class support
  - `None`: Return metrics for each class separately

### Confusion Matrix

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_true, y_pred)
print(cm)

# For visualization
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
disp.plot()
```

### Classification Report

The `classification_report` function provides a comprehensive summary:

```python
from sklearn.metrics import classification_report

report = classification_report(y_true, y_pred, target_names=['Negative', 'Positive'])
print(report)
```

This outputs precision, recall, F1-score, and support for each class.

### ROC and PR Curves

```python
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

# Get prediction probabilities (not just classes)
y_prob = model.predict_proba(X_test)[:, 1]  # Probability of positive class

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
roc_display.plot()

# Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
pr_auc = auc(recall, precision)

# Plot PR
pr_display = PrecisionRecallDisplay(precision=precision, recall=recall, average_precision=pr_auc)
pr_display.plot()
```

**Key Parameters:**	

- `roc_curve(y_true, y_score)`: y_score should be prediction probabilities or confidence scores
- `precision_recall_curve`: Similar requirements
- `pos_label`: Specify which class is considered positive (default=1)
- `average_precision`: For multi-class, specify averaging method

---

## Practical Applications

Let's apply these metrics to a real dataset. We'll use the breast cancer dataset from scikit-learn to demonstrate evaluation techniques:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc, precision_recall_curve)

# Load data
data = load_breast_cancer()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Compute basic metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-Score: {f1:.3f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Benign', 'Malignant']))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_prob)
pr_auc = auc(recall, precision)

plt.subplot(1, 2, 2)
plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")

plt.tight_layout()
plt.show()
```

**Interpreting Results:**

In this breast cancer example, we see:
- High accuracy (around 97%), but let's examine the confusion matrix
- The model correctly identified 105 malignant cases but missed 3 (FN)
- It incorrectly flagged 2 benign cases as malignant (FP)

The ROC curve shows excellent discriminative ability (AUC ≈ 0.99), while the PR curve highlights strong performance in the relevant range.

---

## Expert Insights

### When Metrics Disagree

Different metrics can tell conflicting stories about model performance:

- **High Accuracy, Low Precision/Recall**: Common in imbalanced datasets where the model predicts the majority class
- **High Precision, Low Recall**: Conservative models that only predict positive when very confident, missing many true positives
- **High Recall, Low Precision**: Aggressive models that cast a wide net, catching most positives but with many false alarms

### Choosing the Right Metric

- **Balanced datasets**: Accuracy or F1-score
- **Imbalanced datasets**: Precision, Recall, or F1-score depending on the cost of errors
- **Medical diagnosis**: Prioritize Recall (catch all diseases) or use domain-specific thresholds
- **Spam detection**: Balance Precision (avoid false positives) and Recall
- **Fraud detection**: Often prioritize Recall to catch fraudulent transactions

### Threshold Selection

Most metrics depend on the classification threshold (default 0.5). In practice:

- Use ROC/PR curves to visualize performance across thresholds
- Choose threshold based on business requirements (e.g., cost-benefit analysis)
- Consider probability calibration for reliable probability estimates

### Common Pitfalls

- **Data leakage**: Evaluating on training data leads to overly optimistic metrics
- **Class imbalance**: Accuracy can be misleading; use stratified sampling
- **Multi-class confusion**: Micro vs macro averaging can give different results
- **Threshold dependence**: Metrics change with classification threshold

### Computational Considerations

- Most metrics are O(n) in computation time
- ROC/PR curve computation involves sorting predictions: O(n log n)
- For large datasets, consider sampling for curve plotting

---

## Self-Check Questions

1. Why is accuracy insufficient for evaluating models on imbalanced datasets?
2. What does a high precision but low recall indicate about a model's behavior?
3. When would you prefer AUC-PR over AUC-ROC for model evaluation?
4. How does the choice of classification threshold affect precision and recall?

---

## Try This Exercise

**Evaluate a Model on an Imbalanced Dataset**

1. Load the credit card fraud dataset from Kaggle (or use `make_classification` with class imbalance)
2. Train a logistic regression model
3. Compute accuracy, precision, recall, and F1-score
4. Generate ROC and PR curves
5. Compare performance when using different classification thresholds (0.1, 0.5, 0.9)
6. Analyze how the confusion matrix changes with threshold

**Expected Outcome**: You'll observe how accuracy remains high while precision and recall vary significantly, demonstrating the importance of choosing appropriate metrics for imbalanced problems.

---

## Builder's Insight

Model evaluation isn't just about picking the "best" number—it's about understanding your model's behavior in the context of your application. A model with 90% accuracy might be perfect for one use case but completely inadequate for another where specific types of errors are costly.

Remember: Your evaluation metrics should reflect the real-world impact of your model's decisions. Choose metrics that align with business objectives, not just mathematical convenience. The most sophisticated model is worthless if it doesn't solve the right problem.

As you progress in your machine learning journey, developing intuition for when and how to apply different metrics will become as important as understanding the algorithms themselves.

---