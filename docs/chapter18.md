---
hide:
  - toc
---

# Chapter 15: Choosing Decision Thresholds

> *"The default threshold of 0.5 is often just a starting point—choosing the right threshold can transform model performance."*

---

## Learning Objectives

By the end of this chapter, you will be able to:

- Understand the difference between probability predictions and class predictions
- Implement cost-sensitive classification by adjusting decision thresholds
- Optimize thresholds for different evaluation metrics (precision, recall, F1)
- Use precision-recall curves and other tools for threshold selection

---

## Intuitive Introduction

Imagine you're a doctor deciding whether to administer an expensive treatment. The treatment works 90% of the time but costs $10,000 and has side effects. A false positive means unnecessary treatment and expense, while a false negative means missing a life-saving opportunity.

Machine learning models typically use a default threshold of 0.5 to convert probabilities into class predictions. But this arbitrary threshold doesn't consider the real-world costs of different types of errors. By choosing the right threshold, you can optimize your model for specific scenarios—prioritizing precision when false positives are costly, or recall when false negatives are dangerous.

This chapter explores how to move beyond the default 0.5 threshold to make more informed classification decisions that align with business objectives and real-world constraints.

---

## Mathematical Development

The decision threshold transforms probability estimates into binary classifications. For a binary classifier with probability output P(y=1|x), the prediction is:

ŷ = 1 if P(y=1|x) ≥ t, else 0

Where t is the decision threshold (default t=0.5).

### Cost-Sensitive Classification

Different misclassification errors can have different costs. Let FP_cost be the cost of false positive, FN_cost the cost of false negative. The expected cost for a prediction is:

Cost = P(y=1|x) * FN_cost + (1 - P(y=1|x)) * FP_cost

The optimal threshold minimizes expected cost:

t* = argmin_t [P(y=1|x) * FN_cost + (1 - P(y=1|x)) * FP_cost]

For equal costs, this simplifies to t* = 0.5.

### Threshold and Performance Metrics

**Precision-Recall Trade-off:**

Precision = TP / (TP + FP)

Recall = TP / (TP + FN)

F1 = 2 * Precision * Recall / (Precision + Recall)

As threshold increases:
- Precision typically increases (fewer false positives)
- Recall typically decreases (more false negatives)
- F1 has a maximum at some optimal threshold

For web sources on threshold selection:
- Scikit-learn documentation: https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics
- "The Relationship Between Precision-Recall and ROC Curves" (Davis and Goadrich, 2006)

---

## Implementation Guide

Scikit-learn provides tools for threshold analysis and optimization. The key functions are in sklearn.metrics.

### Basic Threshold Operations

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load data
data = load_breast_cancer()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Get probabilities
probabilities = model.predict_proba(X_test)[:, 1]  # Probability of positive class

# Manual threshold prediction
def predict_with_threshold(probabilities, threshold=0.5):
    return (probabilities >= threshold).astype(int)

# Compare different thresholds
thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
for t in thresholds:
    predictions = predict_with_threshold(probabilities, t)
    accuracy = np.mean(predictions == y_test)
    print(f"Threshold {t}: Accuracy = {accuracy:.3f}")
```

### Precision-Recall Curve Analysis

```python
from sklearn.metrics import precision_recall_curve, auc

# Calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, probabilities)

# Calculate F1 scores for each threshold
f1_scores = 2 * precision * recall / (precision + recall)

# Find optimal threshold for F1
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]
optimal_f1 = f1_scores[optimal_idx]

print(f"Optimal threshold for F1: {optimal_threshold:.3f}")
print(f"Maximum F1 score: {optimal_f1:.3f}")

# Calculate area under PR curve
pr_auc = auc(recall, precision)
print(f"Area under PR curve: {pr_auc:.3f}")
```

**Parameter Explanations:**

- `precision_recall_curve`: Returns precision, recall, and thresholds
- `thresholds`: Array of threshold values where metrics change
- `auc`: Computes area under curve for PR curve evaluation

### Cost-Sensitive Threshold Selection

```python
def find_cost_optimal_threshold(y_true, y_prob, fp_cost=1, fn_cost=1):
    """
    Find threshold that minimizes expected cost
    """
    thresholds = np.linspace(0, 1, 100)
    costs = []
    
    for threshold in thresholds:
        predictions = (y_prob >= threshold).astype(int)
        
        # Calculate confusion matrix elements
        tp = np.sum((predictions == 1) & (y_true == 1))
        fp = np.sum((predictions == 1) & (y_true == 0))
        fn = np.sum((predictions == 0) & (y_true == 1))
        
        # Calculate expected cost
        cost = fp * fp_cost + fn * fn_cost
        costs.append(cost)
    
    # Find minimum cost threshold
    min_cost_idx = np.argmin(costs)
    optimal_threshold = thresholds[min_cost_idx]
    min_cost = costs[min_cost_idx]
    
    return optimal_threshold, min_cost

# Example with different cost ratios
fp_costs = [1, 5, 10]  # False positive costs
fn_costs = [1, 1, 1]   # False negative costs

for fp_cost, fn_cost in zip(fp_costs, fn_costs):
    threshold, cost = find_cost_optimal_threshold(y_test, probabilities, fp_cost, fn_cost)
    print(f"FP cost: {fp_cost}, FN cost: {fn_cost}")
    print(f"Optimal threshold: {threshold:.3f}, Min cost: {cost}")
    print()
```

### Threshold vs Performance Plot

```python
from sklearn.metrics import precision_score, recall_score, f1_score

def plot_threshold_performance(y_true, y_prob, thresholds=np.linspace(0, 1, 100)):
    """
    Plot precision, recall, and F1 vs threshold
    """
    import matplotlib.pyplot as plt
    
    precisions = []
    recalls = []
    f1s = []
    
    for threshold in thresholds:
        predictions = (y_prob >= threshold).astype(int)
        precisions.append(precision_score(y_true, predictions, zero_division=0))
        recalls.append(recall_score(y_true, predictions, zero_division=0))
        f1s.append(f1_score(y_true, predictions, zero_division=0))
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(thresholds, precisions, 'b-', label='Precision')
    plt.xlabel('Threshold')
    plt.ylabel('Precision')
    plt.title('Precision vs Threshold')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(thresholds, recalls, 'r-', label='Recall')
    plt.xlabel('Threshold')
    plt.ylabel('Recall')
    plt.title('Recall vs Threshold')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(thresholds, f1s, 'g-', label='F1')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 vs Threshold')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return thresholds, precisions, recalls, f1s

# Plot threshold performance
thresholds, precisions, recalls, f1s = plot_threshold_performance(y_test, probabilities)
```

---

## Practical Applications

Let's apply threshold tuning to optimize model performance for different scenarios using the breast cancer dataset.

### Optimizing for F1 Score

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

# Load and split data
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Get probabilities
probabilities = model.predict_proba(X_test)[:, 1]

# Default threshold (0.5)
default_predictions = model.predict(X_test)
print("Default threshold (0.5) performance:")
print(classification_report(y_test, default_predictions))

# Find optimal threshold for F1
thresholds = np.linspace(0.1, 0.9, 50)
f1_scores = []

for threshold in thresholds:
    predictions = (probabilities >= threshold).astype(int)
    f1 = f1_score(y_test, predictions)
    f1_scores.append(f1)

optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]
optimal_f1 = f1_scores[optimal_idx]

print(f"\nOptimal threshold for F1: {optimal_threshold:.3f}")
print(f"Optimal F1 score: {optimal_f1:.3f}")

# Predictions with optimal threshold
optimal_predictions = (probabilities >= optimal_threshold).astype(int)
print("\nOptimal threshold performance:")
print(classification_report(y_test, optimal_predictions))

# Plot F1 vs threshold
plt.figure(figsize=(8, 5))
plt.plot(thresholds, f1_scores, 'b-', linewidth=2)
plt.axvline(x=0.5, color='r', linestyle='--', label='Default (0.5)')
plt.axvline(x=optimal_threshold, color='g', linestyle='--', label=f'Optimal ({optimal_threshold:.3f})')
plt.xlabel('Decision Threshold')
plt.ylabel('F1 Score')
plt.title('F1 Score vs Decision Threshold')
plt.legend()
plt.grid(True)
plt.show()
```

**Interpreting Results:**

The example shows how threshold tuning can improve F1 score. The optimal threshold (around 0.4-0.6) balances precision and recall better than the default 0.5.

### Cost-Sensitive Classification Example

```python
# Scenario: Medical diagnosis where false negatives are very costly
# FP cost: $100 (unnecessary treatment)
# FN cost: $1000 (missed cancer diagnosis)

fp_cost = 100
fn_cost = 1000

# Calculate expected cost for different thresholds
thresholds = np.linspace(0.01, 0.99, 50)
costs = []

for threshold in thresholds:
    predictions = (probabilities >= threshold).astype(int)
    
    # Confusion matrix
    tp = np.sum((predictions == 1) & (y_test == 1))
    fp = np.sum((predictions == 1) & (y_test == 0))
    fn = np.sum((predictions == 0) & (y_test == 1))
    tn = np.sum((predictions == 0) & (y_test == 0))
    
    # Total cost
    total_cost = fp * fp_cost + fn * fn_cost
    costs.append(total_cost)

# Find minimum cost threshold
min_cost_idx = np.argmin(costs)
optimal_threshold = thresholds[min_cost_idx]
min_cost = costs[min_cost_idx]

print(f"Cost-optimal threshold: {optimal_threshold:.3f}")
print(f"Minimum cost: ${min_cost:.0f}")

# Compare with default threshold
default_cost = np.sum((default_predictions == 1) & (y_test == 0)) * fp_cost + \
               np.sum((default_predictions == 0) & (y_test == 1)) * fn_cost

print(f"Default threshold cost: ${default_cost:.0f}")
print(f"Cost savings: ${(default_cost - min_cost):.0f}")

# Plot cost vs threshold
plt.figure(figsize=(8, 5))
plt.plot(thresholds, costs, 'r-', linewidth=2)
plt.axvline(x=0.5, color='b', linestyle='--', label='Default (0.5)')
plt.axvline(x=optimal_threshold, color='g', linestyle='--', 
            label=f'Cost-optimal ({optimal_threshold:.3f})')
plt.xlabel('Decision Threshold')
plt.ylabel('Total Cost ($)')
plt.title('Total Cost vs Decision Threshold')
plt.legend()
plt.grid(True)
plt.show()
```

**Interpreting Results:**

In cost-sensitive scenarios, the optimal threshold shifts based on relative costs. When false negatives are much more expensive (as in medical diagnosis), the optimal threshold decreases to catch more positive cases, even at the expense of more false positives.

### Precision-Recall Curve with Threshold Selection

```python
from sklearn.metrics import precision_recall_curve, auc

# Calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, probabilities)
pr_auc = auc(recall, precision)

# Find threshold for specific precision or recall targets
target_precision = 0.9
target_recall = 0.9

# Threshold for target precision (find highest threshold that meets precision)
precision_thresholds = thresholds[precision[:-1] >= target_precision]
if len(precision_thresholds) > 0:
    threshold_for_precision = precision_thresholds[0]  # Highest threshold
    print(f"Threshold for {target_precision} precision: {threshold_for_precision:.3f}")
else:
    print(f"No threshold achieves {target_precision} precision")

# Threshold for target recall (find lowest threshold that meets recall)
recall_thresholds = thresholds[recall[:-1] >= target_recall]
if len(recall_thresholds) > 0:
    threshold_for_recall = recall_thresholds[-1]  # Lowest threshold
    print(f"Threshold for {target_recall} recall: {threshold_for_recall:.3f}")
else:
    print(f"No threshold achieves {target_recall} recall")

# Plot precision-recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, 'b-', linewidth=2, label=f'PR curve (AUC = {pr_auc:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.show()
```

**Interpreting Results:**

The precision-recall curve shows the trade-off between precision and recall. Different applications require different operating points on this curve, which correspond to different thresholds.

---

## Expert Insights

### When to Adjust Thresholds

**Adjust thresholds when:**
- Class distributions are imbalanced
- Different error types have different costs
- You need to optimize for specific metrics (precision, recall, F1)
- Business requirements dictate specific performance targets

**Don't adjust thresholds when:**
- Classes are perfectly balanced
- All misclassifications have equal cost
- You're using threshold-independent metrics for model comparison

### Choosing the Right Threshold

**For High Precision Applications:**
- Medical screening (minimize false positives)
- Fraud detection (avoid false alarms)
- Content moderation (avoid blocking legitimate content)

**For High Recall Applications:**
- Medical diagnosis (catch all diseases)
- Security systems (detect all threats)
- Quality control (find all defects)

**For Balanced Performance:**
- Use F1 score optimization
- Consider cost-benefit analysis
- Use domain expertise to set appropriate trade-offs

### Common Pitfalls

- **Threshold overfitting**: Don't tune threshold on test data
- **Ignoring class imbalance**: Thresholds behave differently with imbalanced data
- **Fixed thresholds across datasets**: Optimal thresholds vary by dataset and model
- **Neglecting probability calibration**: Threshold tuning works best with well-calibrated probabilities

### Advanced Techniques

- **Cost curves**: Visualize expected cost vs threshold
- **Utility theory**: Formal decision-making under uncertainty
- **Multi-threshold classification**: Different thresholds for different scenarios
- **Dynamic thresholds**: Thresholds that adapt based on input features

### Performance Considerations

- **Computational cost**: Threshold tuning is fast (no retraining needed)
- **Robustness**: Thresholds can be sensitive to probability calibration
- **Interpretability**: Thresholds provide clear decision rules
- **Model agnostic**: Works with any probabilistic classifier

### Best Practices

- Always use cross-validation for threshold selection
- Consider the full cost-benefit analysis
- Validate threshold performance on held-out data
- Document threshold choices and their rationale
- Monitor threshold performance in production

---

## Self-Check Questions

1. What is the difference between predict_proba() and predict() in scikit-learn?
2. Why might you want to adjust the decision threshold from the default 0.5?
3. How do precision and recall change as you increase the decision threshold?
4. What is cost-sensitive classification and when should you use it?

---

## Try This Exercise

**Threshold Optimization Challenge**

1. Load the credit card fraud detection dataset (or simulate imbalanced data)
2. Train a classifier and compare default threshold performance
3. Implement threshold tuning to optimize for:
   - F1 score
   - Precision at 95%
   - Cost minimization (assign appropriate FP/FN costs)
4. Plot precision-recall curves and cost curves
5. Compare the performance improvements from threshold tuning
6. Analyze how class imbalance affects optimal thresholds

**Expected Outcome**: You'll understand how threshold selection can dramatically improve model performance for specific use cases and learn to balance competing objectives through cost-sensitive decision making.

---

## Builder's Insight

Threshold tuning is where machine learning meets real-world decision-making. While algorithms optimize mathematical objectives, the final classification decision should align with business goals and human values.

The default 0.5 threshold is a mathematical convenience, not a business requirement. By thoughtfully choosing thresholds, you can create models that are not just accurate, but truly useful—catching the diseases that matter, detecting the fraud that costs money, or moderating content in ways that respect human dignity.

Remember that threshold selection is a design choice, not a technical optimization. It requires understanding your users, your costs, and your ethical responsibilities. The best models don't just predict—they help make better decisions.


