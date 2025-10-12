---
hide:
  - toc
---

# Chapter 14: Probability Calibration

> *"The probability of an event is not the same as our confidence in its occurrence—calibration bridges that gap."*

---

## Learning Objectives

By the end of this chapter, you will be able to:

- Understand why machine learning models often produce poorly calibrated probabilities
- Explain the mathematical foundations of Platt scaling and isotonic regression
- Implement probability calibration using scikit-learn's CalibratedClassifierCV
- Evaluate and compare different calibration methods on real datasets

---

## Intuitive Introduction

Imagine you're playing poker. Your opponent bets aggressively, and you need to decide whether they have a strong hand. A well-calibrated poker player doesn't just think "they might have a good hand"—they assign probabilities: "there's a 70% chance they have at least three of a kind."

Machine learning classifiers often output probabilities, but these probabilities are frequently uncalibrated. An uncalibrated model might predict a 0.9 probability for an event that actually occurs only 60% of the time. This is like a weather forecast that says "90% chance of rain" but it only rains 60% of the time.

Probability calibration transforms these unreliable probability estimates into well-calibrated ones where the predicted probability matches the true frequency of the event. This is crucial when you need reliable probability estimates for decision-making, risk assessment, or cost-sensitive applications.

---

## Mathematical Development

Probability calibration addresses the mismatch between predicted probabilities and observed frequencies. For a well-calibrated classifier, if we group predictions by their predicted probability p, the fraction of positive examples in each group should be approximately p.

### Platt Scaling (Sigmoid Calibration)

Platt scaling fits a sigmoid function to the decision values or uncalibrated probabilities. The calibrated probability is:

$$P(y=1|x) = \frac{1}{1 + \exp(A \cdot f(x) + B)}$$

Where $f(x)$ is the decision function output, and A, B are learned parameters.

This is equivalent to logistic regression on the decision values.

### Isotonic Regression

Isotonic regression is a non-parametric approach that fits a piecewise constant function to minimize the mean squared error while preserving monotonicity. It finds a function g such that:

g(f(x)) ≈ P(y=1|f(x))

Subject to g being non-decreasing.

For web sources on probability calibration:
- Scikit-learn documentation: https://scikit-learn.org/stable/modules/calibration.html
- "Predicting Good Probabilities With Supervised Learning" (Platt, 1999)

---

## Implementation Guide

Scikit-learn provides `CalibratedClassifierCV` for probability calibration. It supports both sigmoid (Platt scaling) and isotonic regression methods.

### Basic Calibration Usage

```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.datasets import load_iris

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Create uncalibrated classifier (SVM with decision function)
svm = SVC(probability=False)  # Don't use built-in probabilities

# Calibrate with sigmoid (Platt scaling)
calibrated_svm = CalibratedClassifierCV(svm, method='sigmoid', cv=3)
calibrated_svm.fit(X, y)

# Get calibrated probabilities
probabilities = calibrated_svm.predict_proba(X)
print(f"Calibrated probabilities shape: {probabilities.shape}")
```

**Parameter Explanations:**

- `base_estimator`: The uncalibrated classifier (must have decision_function or predict_proba)
- `method`: 'sigmoid' (Platt scaling) or 'isotonic' (isotonic regression)
- `cv`: Cross-validation folds for calibration (default=3)
- `ensemble`: Whether to use ensemble calibration (default=True)

### Calibration with Isotonic Regression

```python
# Calibrate with isotonic regression
calibrated_svm_iso = CalibratedClassifierCV(svm, method='isotonic', cv=3)
calibrated_svm_iso.fit(X, y)

# Compare methods
prob_sigmoid = calibrated_svm.predict_proba(X[:5])
prob_isotonic = calibrated_svm_iso.predict_proba(X[:5])

print("Sigmoid calibration probabilities:")
print(prob_sigmoid)
print("\nIsotonic calibration probabilities:")
print(prob_isotonic)
```

### Prefit Calibration

```python
from sklearn.model_selection import train_test_split

# Split data for calibration
X_train, X_cal, y_train, y_cal = train_test_split(X, y, test_size=0.3, random_state=42)

# Train base classifier
svm.fit(X_train, y_train)

# Calibrate on separate data
calibrated_svm_prefit = CalibratedClassifierCV(svm, method='sigmoid', cv='prefit')
calibrated_svm_prefit.fit(X_cal, y_cal)

# Now calibrated_svm_prefit can make calibrated predictions
```

---

## Practical Applications

Let's demonstrate probability calibration using the breast cancer dataset, comparing calibrated and uncalibrated probabilities.

### Calibration Comparison

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import train_test_split

# Load breast cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train uncalibrated SVM
svm_uncalibrated = SVC(probability=True)  # Use built-in probabilities for comparison
svm_uncalibrated.fit(X_train, y_train)

# Train calibrated SVM
svm_base = SVC(probability=False)  # No built-in probabilities
svm_calibrated = CalibratedClassifierCV(svm_base, method='sigmoid', cv=3)
svm_calibrated.fit(X_train, y_train)

# Train calibrated Naive Bayes
nb_base = GaussianNB()
nb_calibrated = CalibratedClassifierCV(nb_base, method='isotonic', cv=3)
nb_calibrated.fit(X_train, y_train)

# Get probabilities on test set
prob_uncal = svm_uncalibrated.predict_proba(X_test)[:, 1]
prob_svm_cal = svm_calibrated.predict_proba(X_test)[:, 1]
prob_nb_cal = nb_calibrated.predict_proba(X_test)[:, 1]

# Plot calibration curves
plt.figure(figsize=(12, 4))

# SVM uncalibrated
plt.subplot(1, 3, 1)
prob_true, prob_pred = calibration_curve(y_test, prob_uncal, n_bins=10)
plt.plot(prob_pred, prob_true, 's-', label='SVM (uncalibrated)')
plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.title('SVM Uncalibrated')
plt.legend()
plt.grid(True)

# SVM calibrated
plt.subplot(1, 3, 2)
prob_true, prob_pred = calibration_curve(y_test, prob_svm_cal, n_bins=10)
plt.plot(prob_pred, prob_true, 's-', label='SVM (calibrated)')
plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.title('SVM Calibrated (Sigmoid)')
plt.legend()
plt.grid(True)

# Naive Bayes calibrated
plt.subplot(1, 3, 3)
prob_true, prob_pred = calibration_curve(y_test, prob_nb_cal, n_bins=10)
plt.plot(prob_pred, prob_true, 's-', label='Naive Bayes (calibrated)')
plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.title('Naive Bayes Calibrated (Isotonic)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

**Interpreting Results:**

The calibration curves show how well the predicted probabilities match the true frequencies. A perfectly calibrated model follows the diagonal line. The plots demonstrate:

- Uncalibrated SVM probabilities are poorly calibrated
- Sigmoid calibration significantly improves SVM calibration
- Isotonic regression provides good calibration for Naive Bayes

### Expected Calibration Error (ECE)

```python
def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Compute Expected Calibration Error (ECE)
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0
    total_samples = len(y_true)
    
    for i in range(n_bins):
        bin_mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if np.sum(bin_mask) > 0:
            bin_prob = np.mean(y_prob[bin_mask])
            bin_acc = np.mean(y_true[bin_mask])
            bin_size = np.sum(bin_mask)
            ece += (bin_size / total_samples) * abs(bin_acc - bin_prob)
    
    return ece

# Calculate ECE for different methods
ece_uncal = expected_calibration_error(y_test, prob_uncal)
ece_svm_cal = expected_calibration_error(y_test, prob_svm_cal)
ece_nb_cal = expected_calibration_error(y_test, prob_nb_cal)

print(f"ECE - SVM Uncalibrated: {ece_uncal:.4f}")
print(f"ECE - SVM Calibrated: {ece_svm_cal:.4f}")
print(f"ECE - Naive Bayes Calibrated: {ece_nb_cal:.4f}")
```

**ECE Interpretation:**

Lower ECE values indicate better calibration. The calibrated models should show significantly lower ECE compared to the uncalibrated SVM.

---

## Expert Insights

### When to Use Calibration

**Calibration is essential when:**
- You need reliable probability estimates for decision-making
- Using cost-sensitive classification
- Building probabilistic models or ensembles
- Interpreting model confidence for risk assessment

**Calibration may be less important when:**
- Only class predictions matter (not probabilities)
- All classes have equal misclassification costs
- Using threshold-independent metrics

### Choosing Between Sigmoid and Isotonic

**Sigmoid (Platt Scaling):**
- Parametric approach, fits logistic regression
- Works well when the calibration curve is S-shaped
- More stable with small datasets
- Faster to train and apply

**Isotonic Regression:**
- Non-parametric, can fit any monotonic function
- Better for complex calibration curves
- More prone to overfitting with small datasets
- Can be slower and more memory-intensive

### Common Pitfalls

- **Calibrating on the same data used for training**: Always use separate calibration data or cross-validation
- **Ignoring class imbalance**: Calibration performance can be affected by class distribution
- **Over-calibrating**: Don't calibrate already well-calibrated models like logistic regression
- **Using calibration for feature engineering**: Calibrated probabilities shouldn't be used as features without careful consideration

### Advanced Techniques

- **Ensemble calibration**: Combining multiple calibration methods
- **Temperature scaling**: Simple scaling for neural networks
- **Beta calibration**: Using beta distributions for better uncertainty quantification
- **Platt binning**: Combining Platt scaling with histogram binning

### Performance Considerations

- **Computational cost**: Isotonic regression is more expensive than sigmoid
- **Memory usage**: Calibration requires storing additional parameters
- **Cross-validation overhead**: CalibratedClassifierCV uses nested CV
- **Scalability**: Both methods scale well with data size

### Best Practices

- Always evaluate calibration on held-out data
- Use cross-validation for robust calibration parameter estimation
- Compare calibration methods using proper metrics (ECE, MCE)
- Consider domain knowledge when choosing calibration method
- Validate that calibration improves decision-making in your application

---

## Self-Check Questions

1. What is probability calibration and why is it important?
2. What are the main differences between Platt scaling and isotonic regression?
3. When should you use probability calibration in practice?
4. How do you evaluate the quality of probability calibration?

---

## Try This Exercise

**Calibration Comparison Study**

1. Load the wine dataset from sklearn.datasets
2. Train SVM, Random Forest, and Logistic Regression classifiers
3. Compare their calibration curves before and after calibration
4. Calculate Expected Calibration Error (ECE) for each method
5. Analyze which models benefit most from calibration
6. Apply calibrated probabilities to a cost-sensitive classification scenario

**Expected Outcome**: You'll understand how different models respond to calibration and when calibration provides the most benefit.

---

## Builder's Insight

Probability calibration is often overlooked but crucial for reliable machine learning systems. Many practitioners focus on accuracy or AUC, but when probabilities matter—whether for medical diagnosis, financial risk assessment, or autonomous driving—calibration becomes paramount.

The key insight is that calibration is a post-processing step that can dramatically improve the reliability of your model's probability estimates without changing the underlying decision boundaries. It's like having a model that's great at ranking examples but terrible at quantifying uncertainty—calibration fixes the uncertainty quantification.

As you build more sophisticated ML systems, remember that well-calibrated probabilities enable better decision-making, more reliable uncertainty estimates, and more trustworthy AI systems. The difference between a model that "thinks" it has 90% confidence and one that actually has 90% accuracy can be the difference between success and failure in critical applications.



