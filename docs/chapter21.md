---
hide:
  - toc
---

# Chapter 18: Dealing with Imbalanced Datasets

> *"In imbalanced datasets, the minority class is like a needle in a haystack—finding it requires the right tools and strategies."*

---

## Learning Objectives

By the end of this chapter, you will be able to:

- Recognize when datasets are imbalanced and why it matters for model performance
- Understand evaluation metrics appropriate for imbalanced classification problems
- Implement class weighting, oversampling (SMOTE), and undersampling techniques
- Choose appropriate strategies for handling imbalanced data based on dataset characteristics

---

## Intuitive Introduction

Imagine you're building a fraud detection system for credit card transactions. Out of 100,000 transactions, only 100 are fraudulent. If your model simply predicts "not fraud" for every transaction, it would be 99.9% accurate—but completely useless.

This is the challenge of imbalanced datasets: when one class (the minority or positive class) is severely underrepresented compared to the majority class. Traditional accuracy becomes misleading because models can achieve high accuracy by ignoring the minority class entirely.

Imbalanced data is common in real-world applications:
- Fraud detection (fraudulent transactions are rare)
- Medical diagnosis (diseases are rare)
- Anomaly detection (abnormal events are rare)
- Quality control (defects are rare)

The key insight is that we care more about correctly identifying the minority class, even if it means accepting more false positives from the majority class. This requires different evaluation metrics and training strategies.

---

## Mathematical Development

Imbalanced datasets require metrics that focus on the minority class performance rather than overall accuracy.

### Class Imbalance Ratio

The imbalance ratio is defined as:

$$\text{Imbalance Ratio} = \frac{N_{\text{majority}}}{N_{\text{minority}}}$$

Where $N_{\text{majority}}$ and $N_{\text{minority}}$ are the number of samples in each class.

### Confusion Matrix and Derived Metrics

For binary classification with imbalanced data:

| Actual/Predicted | Positive | Negative |
|------------------|----------|----------|
| Positive        | TP       | FN       |
| Negative        | FP       | TN       |

Key metrics:

- **Precision**: $\frac{TP}{TP + FP}$ (fraction of predicted positives that are correct)
- **Recall (Sensitivity)**: $\frac{TP}{TP + FN}$ (fraction of actual positives found)
- **Specificity**: $\frac{TN}{TN + FP}$ (fraction of actual negatives correctly identified)
- **F1-Score**: $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ (harmonic mean of precision and recall)

### Area Under ROC Curve (AUC-ROC)

AUC measures the model's ability to discriminate between classes across all classification thresholds:

$$\text{AUC} = \int_0^1 \text{TPR}(t) \cdot (-\text{FPR}'(t)) dt$$

Where TPR is True Positive Rate (Recall) and FPR is False Positive Rate.

### Class Weighting

In weighted loss functions, the minority class gets higher weight:

$$\mathcal{L}_{\text{weighted}} = w_p \sum_{i \in P} \ell(f(x_i)) + w_n \sum_{i \in N} \ell(f(x_i))$$

Where $w_p$ and $w_n$ are weights for positive and negative classes.

For web sources on imbalanced learning:
- Scikit-learn imbalanced-learn documentation: https://imbalanced-learn.org/stable/
- "Learning from Imbalanced Data" by He and Garcia

---

## Implementation Guide

Scikit-learn provides basic class weighting, while the `imbalanced-learn` library offers advanced techniques.

### Class Weighting in Scikit-learn

```python
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Example with imbalanced data
X = np.random.randn(1000, 2)
y = np.random.choice([0, 1], size=1000, p=[0.9, 0.1])  # 10% minority class

# Automatic class weighting
model_balanced = LogisticRegression(class_weight='balanced', random_state=42)
model_balanced.fit(X, y)

# Manual class weighting
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = dict(zip(np.unique(y), class_weights))

model_manual = LogisticRegression(class_weight=class_weight_dict, random_state=42)
model_manual.fit(X, y)

print("Class weights:", class_weight_dict)
```

**LogisticRegression class_weight parameter:**	

- `'balanced'`: Automatically computes weights inversely proportional to class frequencies
- Dictionary: Manual weights for each class
- None (default): No weighting

### Oversampling with SMOTE

SMOTE (Synthetic Minority Oversampling Technique) creates synthetic samples for the minority class:

```python
# Note: Requires imbalanced-learn package
# pip install imbalanced-learn

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Split first, then oversample only training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"Original training class distribution: {np.bincount(y_train)}")
print(f"SMOTE training class distribution: {np.bincount(y_train_smote)}")

# Train model on oversampled data
model_smote = LogisticRegression(random_state=42)
model_smote.fit(X_train_smote, y_train_smote)
```

**SMOTE Parameters:**	

- `k_neighbors=5`: Number of nearest neighbors to use for generating synthetic samples
- `random_state`: For reproducible results
- `sampling_strategy='auto'`: How to balance classes ('minority', 'not majority', 'all')

### Undersampling Techniques

```python
from imblearn.under_sampling import RandomUnderSampler

# Random undersampling
rus = RandomUnderSampler(random_state=42)
X_train_under, y_train_under = rus.fit_resample(X_train, y_train)

print(f"Undersampled training class distribution: {np.bincount(y_train_under)}")

# Train model
model_under = LogisticRegression(random_state=42)
model_under.fit(X_train_under, y_train_under)
```

### Combined Sampling (SMOTE + Tomek Links)

```python
from imblearn.combine import SMOTETomek

# SMOTE + Tomek links (removes noisy samples)
smt = SMOTETomek(random_state=42)
X_train_combined, y_train_combined = smt.fit_resample(X_train, y_train)

print(f"Combined sampling class distribution: {np.bincount(y_train_combined)}")
```

---

## Practical Applications

Let's demonstrate handling imbalanced data on a credit card fraud detection scenario:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_curve, auc, precision_recall_curve)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings('ignore')

# Create imbalanced dataset (simulating fraud detection)
X, y = make_classification(n_samples=10000, n_features=20, n_informative=15,
                          n_redundant=5, n_clusters_per_class=1,
                          weights=[0.95, 0.05], flip_y=0.01, random_state=42)

print(f"Dataset shape: {X.shape}")
print(f"Class distribution: {np.bincount(y)}")
print(f"Imbalance ratio: {np.bincount(y)[0] / np.bincount(y)[1]:.1f}:1")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train models with different strategies
strategies = {}

# 1. Baseline (no handling)
model_baseline = LogisticRegression(random_state=42)
model_baseline.fit(X_train, y_train)
y_pred_baseline = model_baseline.predict(X_test)
y_prob_baseline = model_baseline.predict_proba(X_test)[:, 1]

# 2. Class weighting
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
model_weighted = LogisticRegression(class_weight=dict(zip(np.unique(y_train), class_weights)), random_state=42)
model_weighted.fit(X_train, y_train)
y_pred_weighted = model_weighted.predict(X_test)
y_prob_weighted = model_weighted.predict_proba(X_test)[:, 1]

# 3. SMOTE oversampling
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
model_smote = LogisticRegression(random_state=42)
model_smote.fit(X_train_smote, y_train_smote)
y_pred_smote = model_smote.predict(X_test)
y_prob_smote = model_smote.predict_proba(X_test)[:, 1]

# 4. Random undersampling
rus = RandomUnderSampler(random_state=42)
X_train_under, y_train_under = rus.fit_resample(X_train, y_train)
model_under = LogisticRegression(random_state=42)
model_under.fit(X_train_under, y_train_under)
y_pred_under = model_under.predict(X_test)
y_prob_under = model_under.predict_proba(X_test)[:, 1]

# Evaluate all strategies
strategies = {
    'Baseline': (y_pred_baseline, y_prob_baseline),
    'Class Weights': (y_pred_weighted, y_prob_weighted),
    'SMOTE': (y_pred_smote, y_prob_smote),
    'Undersampling': (y_pred_under, y_prob_under)
}

# Print classification reports
for name, (y_pred, y_prob) in strategies.items():
    print(f"\n{name} Strategy:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))

# Plot ROC curves
plt.figure(figsize=(12, 5))

# ROC curves
plt.subplot(1, 2, 1)
for name, (y_pred, y_prob) in strategies.items():
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.grid(True)

# Precision-Recall curves
plt.subplot(1, 2, 2)
for name, (y_pred, y_prob) in strategies.items():
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, label=f'{name} (AUC = {pr_auc:.3f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Confusion matrices
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.ravel()

for i, (name, (y_pred, y_prob)) in enumerate(strategies.items()):
    cm = confusion_matrix(y_test, y_pred)
    axes[i].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[i].set_title(f'{name} Confusion Matrix')
    
    # Add labels
    thresh = cm.max() / 2.
    for j in range(cm.shape[0]):
        for k in range(cm.shape[1]):
            axes[i].text(k, j, format(cm[j, k], 'd'),
                        ha="center", va="center",
                        color="white" if cm[j, k] > thresh else "black")
    
    axes[i].set_xticks([0, 1])
    axes[i].set_yticks([0, 1])
    axes[i].set_xticklabels(['Normal', 'Fraud'])
    axes[i].set_yticklabels(['Normal', 'Fraud'])
    axes[i].set_ylabel('True label')
    axes[i].set_xlabel('Predicted label')

plt.tight_layout()
plt.show()
```

**Interpreting Results:**	

The example demonstrates:
- Baseline model achieves high accuracy but poor fraud detection (low recall)
- Class weighting improves recall without sacrificing too much precision
- SMOTE oversampling provides the best balance of precision and recall
- Undersampling can be effective but may lose important information
- ROC curves show discrimination ability, PR curves better reflect imbalanced performance

---

## Expert Insights

### When to Use Each Technique

**Class Weighting:**	
- Simple to implement, no data modification
- Works with any algorithm that supports weights
- Good for moderate imbalance
- May not be sufficient for extreme imbalance

**Oversampling (SMOTE):**	
- Creates synthetic data, preserves information
- Effective for small datasets
- Can introduce noise if minority class has outliers
- Computationally intensive for large datasets

**Undersampling:**	
- Reduces training time and memory
- Good for very large majority classes
- Risk of losing important information
- May not work well with small minority classes

**Combined Approaches:**	
- SMOTE + Tomek: Oversample minority, remove noisy majority samples
- SMOTE + ENN: Oversample minority, remove noisy samples from both classes

### Choosing Evaluation Metrics

- **Accuracy**: Misleading for imbalanced data
- **Precision**: Important when false positives are costly
- **Recall**: Critical when false negatives are costly
- **F1-Score**: Balances precision and recall
- **AUC-ROC**: Good for ranking/discrimination
- **AUC-PR**: Better for imbalanced data evaluation

### Common Pitfalls

- **Data leakage**: Never oversample before cross-validation
- **Evaluation bias**: Use stratified sampling for imbalanced data
- **Overfitting**: Oversampling can cause synthetic data overfitting
- **Class distribution**: Real-world imbalance may differ from training

### Performance Considerations

- Oversampling increases dataset size (memory)
- Undersampling reduces dataset size (may lose information)
- Class weighting has minimal computational overhead
- Consider ensemble methods for imbalanced data

### Best Practices

- Always use stratified cross-validation
- Evaluate on multiple metrics, not just accuracy
- Consider the cost of false positives vs false negatives
- Validate on held-out test set with real class distribution
- Document imbalance handling decisions

---

## Self-Check Questions

1. Why is accuracy misleading for imbalanced datasets?
2. What are the key differences between precision and recall?
3. When should you use SMOTE versus class weighting?
4. How does undersampling affect model performance?

---

## Try This Exercise

**Imbalanced Data Handling Comparison**

1. Load a highly imbalanced dataset (e.g., using make_classification with severe imbalance)
2. Compare performance of LogisticRegression with different imbalance handling strategies:
   - No handling (baseline)
   - Class weighting
   - SMOTE oversampling
   - Random undersampling
3. Evaluate using precision, recall, F1-score, and AUC-PR
4. Plot precision-recall curves for all strategies
5. Analyze the confusion matrices and discuss trade-offs

**Expected Outcome**: You'll understand how different techniques affect model behavior on minority classes and learn to choose appropriate strategies based on business requirements.

---

## Builder's Insight

Imbalanced data handling is where machine learning meets real-world constraints. The "perfect" model that ignores business costs is worthless in production.

Remember: Your model's success isn't measured by accuracy alone, but by its ability to find what mattersâ€”the rare events that drive business value. Understanding imbalance forces you to think deeply about what you're really trying to predict and why it matters.

As you tackle more complex problems, imbalance handling becomes part of your modeling toolkit. The key insight: different applications demand different trade-offs between precision and recall. Master this, and you'll build models that actually solve real problems.


