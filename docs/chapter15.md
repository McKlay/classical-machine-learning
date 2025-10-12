---
hide:
  - toc
---

# Chapter 12: Cross-Validation & StratifiedKFold

> *"Cross-validation is the reality check for machine learning models ensuring our performance estimates aren't just lucky guesses."*

---

## Learning Objectives

By the end of this chapter, you will be able to:

- Understand the importance of cross-validation for reliable model evaluation
- Differentiate between KFold and StratifiedKFold cross-validation strategies
- Implement cross-validation using scikit-learn's `cross_validate`, `GridSearchCV`, and `RandomizedSearchCV`
- Choose appropriate cross-validation strategies for different dataset characteristics

---

## Intuitive Introduction

Imagine you're training a model to predict house prices. You split your data into training and test sets, train on the training set, and evaluate on the test set. But what if that particular test set was unusually easy or hard? Your performance estimate might be misleading.

Cross-validation solves this by systematically rotating which portion of data serves as the test set. Instead of one train/test split, you get multiple evaluations, providing a more robust estimate of your model's true performance.

This is especially crucial in machine learning because models can easily overfit to specific data splits. Cross-validation helps ensure your model generalizes well to unseen data, not just the particular subset you happened to choose for testing.

---

## Mathematical Development

Cross-validation provides a statistical method to estimate the generalization error of a model by partitioning the data into complementary subsets.

### K-Fold Cross-Validation

In K-fold CV, the dataset is divided into K equally sized folds. The model is trained K times, each time using K-1 folds for training and 1 fold for validation. The performance scores are then averaged.

For a dataset of size N and K folds:

- Each fold has approximately N/K samples
- Training set size: (K-1)N/K samples
- Validation set size: N/K samples

The cross-validation score is:

$$\text{CV Score} = \frac{1}{K} \sum_{i=1}^{K} \text{Score}_i$$

Where Score_i is the performance metric on fold i.

### Stratified K-Fold Cross-Validation

StratifiedKFold ensures that each fold maintains the same proportion of samples from each class as the original dataset. This is crucial for imbalanced datasets where random splits might create folds with very different class distributions.

For a binary classification problem with class proportions p (positive) and 1-p (negative), each fold will contain approximately p*N/K positive samples and (1-p)*N/K negative samples.

### Leave-One-Out Cross-Validation (LOOCV)

A special case where K = N, providing maximum training data but computationally expensive:

$$\text{LOOCV Score} = \frac{1}{N} \sum_{i=1}^{N} \text{Score}_{-i}$$

Where Score_{-i} is the score when sample i is left out for validation.

For web sources on cross-validation theory, see:
- Scikit-learn documentation: https://scikit-learn.org/stable/modules/cross_validation.html
- Elements of Statistical Learning (Hastie et al.)

---

## Implementation Guide

Scikit-learn provides comprehensive cross-validation tools in `sklearn.model_selection`. Let's explore the key functions:

### Basic Cross-Validation

```python
from sklearn.model_selection import cross_validate, KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Create model
model = LogisticRegression(random_state=42, max_iter=1000)

# K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = cross_validate(model, X, y, cv=kf, scoring=['accuracy', 'precision_macro', 'recall_macro'])

print("K-Fold CV Results:")
print(f"Accuracy: {cv_results['test_accuracy'].mean():.3f} (+/- {cv_results['test_accuracy'].std() * 2:.3f})")
print(f"Precision: {cv_results['test_precision_macro'].mean():.3f}")
print(f"Recall: {cv_results['test_recall_macro'].mean():.3f}")
```

**Parameter Explanations:**

- `n_splits`: Number of folds (K). Default=5, common values 5 or 10
- `shuffle`: Whether to shuffle data before splitting. Recommended for time-series data
- `random_state`: Ensures reproducible results

### Stratified K-Fold

```python
# Stratified K-Fold (recommended for classification)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results_stratified = cross_validate(model, X, y, cv=skf, scoring='accuracy')

print("\nStratified K-Fold CV Results:")
print(f"Accuracy: {cv_results_stratified['test_score'].mean():.3f} (+/- {cv_results_stratified['test_score'].std() * 2:.3f})")
```

StratifiedKFold automatically stratifies based on the target variable `y`.

### Cross-Validation with Multiple Metrics

```python
# Multiple scoring metrics
scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision_macro',
    'recall': 'recall_macro',
    'f1': 'f1_macro'
}

cv_results_multi = cross_validate(model, X, y, cv=skf, scoring=scoring)

for metric, scores in cv_results_multi.items():
    if metric.startswith('test_'):
        print(f"{metric[5:]}: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
```

### Grid Search with Cross-Validation

```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# Define parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'linear']
}

# Create GridSearchCV
grid_search = GridSearchCV(
    SVC(),
    param_grid,
    cv=5,  # 5-fold CV
    scoring='accuracy',
    n_jobs=-1,  # Use all available cores
    verbose=1
)

# Fit on data
grid_search.fit(X, y)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
```

**Key Parameters for GridSearchCV:**	

- `cv`: Cross-validation strategy (default=5)
- `scoring`: Evaluation metric
- `n_jobs`: Number of parallel jobs (-1 uses all cores)
- `refit`: Whether to refit on entire training set with best params (default=True)

### Randomized Search

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, loguniform

# Define parameter distributions
param_dist = {
    'C': loguniform(1e-4, 1e2),
    'gamma': loguniform(1e-4, 1e-1),
    'kernel': ['rbf', 'linear']
}

# Create RandomizedSearchCV
random_search = RandomizedSearchCV(
    SVC(),
    param_dist,
    n_iter=20,  # Number of parameter settings sampled
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

random_search.fit(X, y)

print(f"Best parameters: {random_search.best_params_}")
print(f"Best cross-validation score: {random_search.best_score_:.3f}")
```

**RandomizedSearchCV advantages:**
- More efficient for large parameter spaces
- Can sample from continuous distributions
- Often finds good solutions faster than exhaustive grid search

---

## Practical Applications

Let's apply cross-validation to a real-world example using the breast cancer dataset:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (cross_validate, KFold, StratifiedKFold, 
                                     GridSearchCV, validation_curve)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score

# Load data
data = load_breast_cancer()
X, y = data.data, data.target

print(f"Dataset shape: {X.shape}")
print(f"Class distribution: {np.bincount(y)}")

# Compare KFold vs StratifiedKFold
model = RandomForestClassifier(n_estimators=100, random_state=42)

# K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
kf_scores = cross_validate(model, X, y, cv=kf, scoring='f1')

# Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
skf_scores = cross_validate(model, X, y, cv=skf, scoring='f1')

print("K-Fold F1 scores:", kf_scores['test_score'])
print("Stratified K-Fold F1 scores:", skf_scores['test_score'])
print(f"K-Fold mean F1: {kf_scores['test_score'].mean():.3f}")
print(f"Stratified K-Fold mean F1: {skf_scores['test_score'].mean():.3f}")

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

grid_search.fit(X, y)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV F1 score: {grid_search.best_score_:.3f}")

# Plot validation curve for one parameter
train_scores, val_scores = validation_curve(
    RandomForestClassifier(random_state=42),
    X, y,
    param_name='n_estimators',
    param_range=[10, 50, 100, 200, 500],
    cv=5,
    scoring='f1'
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot([10, 50, 100, 200, 500], train_mean, 'o-', label='Training score')
plt.plot([10, 50, 100, 200, 500], val_mean, 'o-', label='Cross-validation score')
plt.fill_between([10, 50, 100, 200, 500], train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.fill_between([10, 50, 100, 200, 500], val_mean - val_std, val_mean + val_std, alpha=0.1)
plt.xlabel('Number of estimators')
plt.ylabel('F1 Score')
plt.title('Validation Curve for Random Forest')
plt.legend()
plt.grid(True)
plt.show()
```

**Interpreting Results:**

The example demonstrates:
- StratifiedKFold provides more consistent performance across folds due to balanced class distributions
- GridSearchCV finds optimal hyperparameters through systematic cross-validation
- Validation curves help visualize the bias-variance tradeoff

---

## Expert Insights

### When to Use Which Cross-Validation Strategy

- **KFold**: General purpose, works for any problem type
- **StratifiedKFold**: Preferred for classification, especially with imbalanced classes
- **LeaveOneOut**: Maximum training data, useful for small datasets but computationally expensive
- **TimeSeriesSplit**: For time-dependent data where future data shouldn't influence past predictions

### Choosing K (Number of Folds)

- **Small K (3-5)**: Faster computation, higher variance in estimates
- **Large K (10)**: More reliable estimates, slower computation
- **LOOCV (K=N)**: Lowest bias, highest variance, very slow

### Common Pitfalls

- **Data leakage**: Ensure no information from validation set leaks into training
- **Temporal dependencies**: Use appropriate CV for time-series data
- **Computational cost**: Balance K with dataset size and model complexity
- **Nested CV**: Use for unbiased hyperparameter tuning evaluation

### Performance Considerations

- Cross-validation is O(K Ã— training_time)
- Parallelization helps: set `n_jobs=-1` in GridSearchCV
- For large datasets, consider GroupKFold or TimeSeriesSplit
- Memory usage scales with K for some algorithms

### Best Practices

- Always use stratified sampling for classification
- Report mean and standard deviation of CV scores
- Use multiple metrics, not just accuracy
- Validate final model on held-out test set after CV tuning

---

## Self-Check Questions

1. Why is a single train/test split insufficient for model evaluation?
2. When should you use StratifiedKFold instead of regular KFold?
3. What are the trade-offs between GridSearchCV and RandomizedSearchCV?
4. How does cross-validation help prevent overfitting?

---

## Try This Exercise

**Cross-Validation Comparison**

1. Load a classification dataset (e.g., wine dataset from sklearn)
2. Compare KFold vs StratifiedKFold performance across different K values (3, 5, 10)
3. Use GridSearchCV to tune hyperparameters for a RandomForest classifier
4. Plot validation curves for at least two hyperparameters
5. Analyze how CV scores vary with different random seeds

**Expected Outcome**: You'll understand the variability in model performance estimates and the importance of robust cross-validation strategies.

---

## Builder's Insight

Cross-validation isn't just a technical requirementâ€”it's the foundation of trustworthy machine learning. Without proper validation, you're building on sand.

Remember: Your model's performance on unseen data is what matters, not how well it memorizes your training set. Cross-validation gives you confidence that your model will generalize.

As you advance in your ML journey, developing intuition for when and how to apply different validation strategies becomes as crucial as understanding the algorithms themselves. A poorly validated model might look impressive on paper but fail spectacularly in production.

---

