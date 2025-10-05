---
hide:
  - toc
---

# Chapter 4: Logistic and Linear Regression

> "*The simplest model is the hardest to beat.*" – Unknown

---

## Why This Chapter Matters

Logistic and linear regression form the foundation of supervised learning. Despite their simplicity, they remain powerful tools for prediction and interpretation. Logistic regression handles classification by modeling probabilities, while linear regression predicts continuous values through linear relationships.

This chapter bridges the gap between basic concepts and practical implementation, showing how these "simple" models can outperform complex algorithms when used correctly.

---

## Learning Objectives

By the end of this chapter, you will be able to:

- Understand the mathematical foundations of logistic and linear regression
- Implement both algorithms in scikit-learn with proper parameter tuning
- Interpret model coefficients for feature importance and decision-making
- Apply regularization techniques to prevent overfitting
- Evaluate and diagnose regression models using appropriate metrics

---

## Intuitive Introduction

Imagine you're predicting whether a student will pass an exam (classification) or their final grade (regression). Logistic regression answers "yes/no" questions by estimating probabilities, like assessing admission chances. Linear regression predicts exact values, like forecasting house prices based on size and location.

These methods are like drawing the best straight line through your data points. Logistic regression bends this line with a sigmoid curve to handle probabilities, while linear regression keeps it straight for continuous predictions. They're the workhorses of ML, reliable and interpretable.

---

## Mathematical Development

### Logistic Regression

**Sigmoid Function and Log-Odds:**
The sigmoid function maps any real number to (0,1):

σ(z) = 1 / (1 + e^(-z))

Where z = w·x + b is the linear combination of features.

Log-odds represent the logarithm of the odds ratio:

log(p/(1-p)) = w·x + b

Solving for p gives the probability: p = σ(w·x + b)

**Decision Boundary:**
For binary classification, predict class 1 if p > 0.5, else class 0.
Geometrically, this creates a hyperplane separating classes in feature space.

**Derivation:**
Given data {(x_i, y_i)}, we maximize likelihood:

L(w,b) = ∏ [σ(w·x_i + b)]^y_i * [1-σ(w·x_i + b)]^(1-y_i)

Taking log and minimizing negative log-likelihood gives the loss function optimized by gradient descent.

### Linear Regression

**Least Squares:**
Minimize the sum of squared residuals:

min_w ||y - Xw||²

**Solution:**
w = (X^T X)^(-1) X^T y

**Ridge and Lasso Regularization:**
Ridge: min_w ||y - Xw||² + α||w||²
Lasso: min_w ||y - Xw||² + α||w||₁

Ridge shrinks coefficients, Lasso can set them to zero for feature selection.

---

## Implementation Guide

### LogisticRegression API
Key parameters:
- `C`: float, default=1.0. Inverse regularization strength (smaller = stronger regularization)
- `penalty`: str, default='l2'. 'l1', 'l2', 'elasticnet', or 'none'
- `solver`: str, default='lbfgs'. Optimization algorithm ('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga')
- `multi_class`: str, default='auto'. 'ovr' (one-vs-rest), 'multinomial', 'auto'
- `max_iter`: int, default=100. Maximum iterations

Methods: `fit`, `predict`, `predict_proba`, `score`

See: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

### LinearRegression API
Key parameters:
- `fit_intercept`: bool, default=True. Whether to calculate intercept
- `normalize`: bool, default=False. Deprecated, use preprocessing

For regularization, use `Ridge` or `Lasso`:
- `alpha`: float, default=1.0. Regularization strength
- `solver`: str, default='auto'. Optimization method

See: [https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)

Best practices: Scale features for regularization, use cross-validation for C/alpha tuning.

---

## Practical Applications

### Logistic Regression on Wine Dataset

```python
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# Load data
wine = load_wine()
X, y = wine.data, wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Grid search for C
param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(LogisticRegression(random_state=42, max_iter=1000), param_grid, cv=5)
grid.fit(X_train_scaled, y_train)

print(f"Best C: {grid.best_params_['C']}")
print(f"Best CV Score: {grid.best_score_:.3f}")

# Evaluate
y_pred = grid.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

# Coefficient interpretation
feature_names = wine.feature_names
for name, coef in zip(feature_names, grid.best_estimator_.coef_[0]):
    print(f"{name}: {coef:.3f}")
```

### Linear Regression on Boston Housing

```python
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
import numpy as np

# Load data
boston = load_boston()
X, y = boston.data, boston.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Compare OLS and Ridge
models = {'OLS': LinearRegression(), 'Ridge': Ridge(alpha=1.0)}
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    print(f"{name} CV MSE: {-scores.mean():.3f} (+/- {scores.std()*2:.3f})")

# Fit and evaluate Ridge
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {mse:.3f}")

# Feature importance
feature_names = boston.feature_names
for name, coef in zip(feature_names, ridge.coef_):
    print(f"{name}: {coef:.3f}")
```

Interpretation: Logistic regression shows class probabilities and feature impacts. Linear regression reveals linear relationships and regularization effects.

---

## Expert Insights

- **Common Pitfalls**: Not scaling features before regularization. Using default C=1.0 without tuning. Ignoring multi-class strategies.
- **Debugging Strategies**: Check coefficient magnitudes for feature importance. Use predict_proba for uncertainty. Plot residuals for linear regression.
- **Parameter Selection**: Start with C=1.0, tune logarithmically. For Ridge, alpha=1.0 is good default. Use liblinear for small datasets, lbfgs for large.
- **Advanced Optimization**: Computational complexity O(n*d) for fitting, where n=samples, d=features. Use saga solver for large, sparse data.

Remember: Interpretability is key—these models explain "why" predictions are made.

---

## Self-Check Questions

1. How does the sigmoid function enable probability estimation in logistic regression?
2. What's the difference between L1 and L2 regularization?
3. Why scale features before applying regularization?
4. How do you interpret logistic regression coefficients?
5. When would you choose Ridge over Lasso regression?

---

## Try This Exercise

> **Regression Comparison**:  
> Load the Diabetes dataset from scikit-learn. Compare LinearRegression, Ridge, and Lasso models using cross-validation. Analyze which features are most important and how regularization affects coefficients. Experiment with different alpha values.

---

## Builder's Insight

Logistic and linear regression aren't outdated—they're essential. Their simplicity forces you to understand your data deeply. Master these, and you'll have the foundation for any ML challenge.

Start with the basics; they'll take you farthest.

---

