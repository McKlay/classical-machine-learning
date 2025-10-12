---
hide:
  - toc
---

# Chapter 9: Random Forests and Bagging

## Learning Objectives

By the end of this chapter, you will be able to:
- Understand the principles of ensemble learning through bagging and random forests
- Explain how bootstrap aggregating reduces variance and improves model stability
- Implement Random Forest classifiers using scikit-learn with proper parameter tuning
- Interpret out-of-bag scores and feature importance metrics
- Compare bagging with other ensemble methods and know when to use each
- Apply Random Forests to real datasets and diagnose potential overfitting

## Intuitive Introduction

Imagine you're trying to predict whether it will rain tomorrow. Instead of relying on a single weather expert, you consult a panel of meteorologists, each with their own forecasting method. Some focus on cloud patterns, others on wind speed, and a few consider historical data. By combining their predictions – perhaps through majority voting – you often get a more reliable forecast than any single expert could provide.

This is the essence of ensemble learning: combining multiple models to create a stronger, more robust predictor. Random Forests take this idea further by using bagging (bootstrap aggregating) to train decision trees on random subsets of data and features. Each tree is like an expert with a slightly different perspective, and their collective wisdom reduces the errors that plague individual trees, such as overfitting to noise or being overly sensitive to small data changes.

Bagging works by creating multiple versions of the training set through bootstrapping – sampling with replacement – and training a separate model on each. The final prediction is an average (for regression) or majority vote (for classification). This approach is particularly effective for high-variance models like decision trees, transforming them from unstable learners into reliable ensemble predictors.

## Mathematical Development

Ensemble methods combine multiple base learners to improve predictive performance. For bagging, we create B bootstrap samples from the original training set of size N:

$$D_b = \{ (x_i, y_i) | i \in \{1, 2, \dots, N\} \} \text{ sampled with replacement}$$

Each base learner \( h_b \) is trained on \( D_b \), and the ensemble prediction is:

For classification (majority vote):
$$\hat{y} = \arg\max_c \sum_{b=1}^B I(h_b(x) = c)$$

For regression (average):
$$\hat{y} = \frac{1}{B} \sum_{b=1}^B h_b(x)$$

Random Forests extend bagging by introducing feature randomness. At each split in each tree, only a random subset of m features (typically $m = \sqrt{p}$ for classification, $m = p/3$ for regression, where p is total features) is considered for splitting.

The out-of-bag (OOB) error provides an unbiased estimate of generalization error without cross-validation. For each observation, the OOB prediction uses only trees trained on bootstrap samples that didn't include that observation.

Feature importance in Random Forests can be measured by the decrease in node impurity (Gini or entropy) averaged across all trees, or by permutation importance – measuring performance drop when a feature's values are randomly shuffled.

Web sources for further reading:
- [https://en.wikipedia.org/wiki/Random_forest](https://en.wikipedia.org/wiki/Random_forest)
- [https://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees](https://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees)

## Implementation Guide

Scikit-learn's `RandomForestClassifier` implements the Random Forest algorithm with bagging and feature randomization.

### RandomForestClassifier API

```python
from sklearn.ensemble import RandomForestClassifier

# Initialize the classifier
rf = RandomForestClassifier(
    n_estimators=100,      # Number of trees in the forest
    criterion='gini',      # Function to measure split quality
    max_depth=None,        # Maximum depth of trees
    min_samples_split=2,   # Minimum samples to split a node
    min_samples_leaf=1,    # Minimum samples at leaf node
    max_features='sqrt',   # Number of features to consider for best split
    bootstrap=True,        # Whether to use bootstrap samples
    oob_score=False,       # Whether to use out-of-bag samples for scoring
    random_state=None,     # Random state for reproducibility
    n_jobs=None            # Number of jobs to run in parallel
)
```

Key parameters:
- `n_estimators`: More trees generally improve performance but increase computation time. Default 100 is often sufficient.
- `max_features`: Controls feature randomization. 'sqrt' (default) works well for most cases.
- `max_depth`: Limits tree depth to prevent overfitting. None allows full growth.
- `bootstrap`: Enables bagging. Set to False for comparison with plain random trees.
- `oob_score`: When True, uses OOB samples to estimate generalization error.

The `fit` method trains the forest:

```python
rf.fit(X_train, y_train)
```

`predict` and `predict_proba` work as expected:

```python
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)
```

Access OOB score and feature importances:

```python
print(f"OOB Score: {rf.oob_score_}")
print(f"Feature Importances: {rf.feature_importances_}")
```

### BaggingClassifier for General Bagging

For bagging with other base estimators:

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bagging = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=1.0,    # Fraction of samples to draw
    max_features=1.0,   # Fraction of features to draw
    bootstrap=True,
    oob_score=True
)
```

## Practical Applications

Let's apply Random Forest to the Wine dataset for quality classification:

```python
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load the dataset
wine = load_wine()
X, y = wine.data, wine.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the model with OOB scoring
rf = RandomForestClassifier(
    n_estimators=100,
    max_features='sqrt',
    oob_score=True,
    random_state=42
)
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)

# Evaluate
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=wine.target_names))

print(f"OOB Score: {rf.oob_score_:.3f}")

# Feature importance analysis
feature_importance = rf.feature_importances_
feature_names = wine.feature_names

# Sort features by importance
indices = np.argsort(feature_importance)[::-1]

print("Top 5 Most Important Features:")
for i in range(5):
    print(f"{feature_names[indices[i]]}: {feature_importance[indices[i]]:.3f}")

# Visualize feature importances
plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importance)), feature_importance[indices])
plt.xticks(range(len(feature_importance)), [feature_names[i] for i in indices], rotation=45)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importances in Random Forest')
plt.tight_layout()
plt.show()
```

This example demonstrates a complete Random Forest workflow: training with OOB scoring, prediction, evaluation, and feature importance analysis. The OOB score provides an internal estimate of performance, while feature importances help understand which wine characteristics are most predictive of quality.

For comparison with bagging:

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# Bagging with decision trees
bagging = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=100,
    oob_score=True,
    random_state=42
)
bagging.fit(X_train, y_train)

print(f"Bagging OOB Score: {bagging.oob_score_:.3f}")
print(f"Random Forest OOB Score: {rf.oob_score_:.3f}")
```

## Expert Insights

Random Forests excel at handling mixed data types, missing values, and high-dimensional datasets. Their built-in feature selection through random subspace selection makes them robust to irrelevant features.

Common pitfalls include:
- Overfitting with too many trees or deep trees: Monitor OOB score during training
- Ignoring feature scaling: Trees are invariant to monotonic transformations, but other preprocessing may help
- Class imbalance: Random Forests can be sensitive; consider balanced sampling or class weights

For model tuning:
- Start with default parameters; they're often near-optimal
- Use OOB score for efficient hyperparameter search
- Increase `n_estimators` for better performance, but watch for diminishing returns
- Adjust `max_features` based on dataset characteristics

Computational complexity is $O(B \times N \times \log N)$ for training, where B is number of trees and N is sample size. Prediction scales linearly with B.

Random Forests provide reliable baselines and often outperform single decision trees. They're particularly useful when you need interpretable feature importances alongside strong predictive performance.

## Self-Check Questions

1. How does bagging reduce the variance of high-variance base learners like decision trees?
2. What is the difference between bagging and random forests?
3. Why is out-of-bag scoring useful, and how does it work?
4. When would you choose random forests over gradient boosting methods?

## Try This Exercise

Compare Random Forest performance across different datasets:

1. Train Random Forest classifiers on the Iris, Wine, and Breast Cancer datasets
2. Experiment with different numbers of estimators (10, 50, 100, 200)
3. Compare OOB scores, test accuracies, and feature importances across datasets
4. Try different `max_features` values ('sqrt', 'log2', None) and observe the effects
5. Visualize how OOB score changes with increasing number of trees

This exercise will demonstrate Random Forests' versatility and the importance of parameter tuning.

## Builder's Insight

Random Forests represent the power of collective intelligence in machine learning. By combining many weak learners through bagging and randomization, they create robust models that often outperform individual algorithms. As you build more sophisticated systems, remember that ensemble methods like Random Forests provide a reliable foundation – simple yet powerful, interpretable yet accurate. In the world of machine learning, sometimes the wisdom of crowds truly is wiser than the expert.

