---
hide:
  - toc
---

# Chapter 13: Hyperparameter Tuning

> *"Hyperparameters are the dials that turn good models into great ones—finding the right settings is both art and science."*

---

## Learning Objectives

By the end of this chapter, you will be able to:

- Understand the difference between grid search and random search for hyperparameter optimization
- Design effective hyperparameter search spaces for different algorithms
- Implement systematic hyperparameter tuning using scikit-learn
- Apply hyperparameter tuning to real-world examples with SVM and Random Forest

---

## Intuitive Introduction

Imagine you're baking cookies. The recipe calls for flour, sugar, and butter—but the perfect cookie depends on how much of each ingredient you use, and how long you bake them. Too much flour makes them dense, too little sugar makes them bland, and wrong baking time can burn them.

Machine learning models are similar. The algorithm provides the basic recipe, but hyperparameters control how the model learns. Finding the right hyperparameters can transform a mediocre model into one that performs exceptionally well.

This chapter explores systematic approaches to hyperparameter tuning, moving beyond trial-and-error to structured search strategies that efficiently explore the parameter space to find optimal model configurations.

---

## Mathematical Development

Hyperparameter tuning involves searching for the optimal point in a high-dimensional parameter space that maximizes model performance. While the underlying model training involves complex mathematics, the tuning process itself is primarily about search strategies.

### Parameter Space and Optimization Landscape

Consider a model with d hyperparameters, each taking values in continuous or discrete ranges. The parameter space Ω is the Cartesian product of these ranges:

Ω = Ω₁ × Ω₂ × ... × Ω_d

The objective is to find:

$$\theta^* = \arg\max_{\theta \in \Omega} \text{Performance}(f(\theta))$$

Where Performance is typically cross-validation score, and $f(\theta)$ is the model trained with hyperparameters $\theta$.

### Grid Search Complexity

Grid search evaluates all combinations of parameter values. For d parameters with $n_i$ possible values each, the total evaluations are:

$$\prod_{i=1}^d n_i$$

This exponential growth makes grid search impractical for large parameter spaces.

### Random Search Efficiency

Random search samples randomly from the parameter space. Bergstra and Bengio (2012) showed that random search often outperforms grid search, especially when some parameters are more important than others.

For web sources on hyperparameter optimization:
- Scikit-learn documentation: https://scikit-learn.org/stable/modules/grid_search.html
- "Random Search for Hyper-Parameter Optimization" (Bergstra and Bengio, 2012)

---

## Implementation Guide

Scikit-learn provides `GridSearchCV` and `RandomizedSearchCV` for systematic hyperparameter tuning. Both integrate cross-validation for robust performance estimation.

### Grid Search with Cross-Validation

```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import load_iris

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Define parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf']
}

# Create GridSearchCV
grid_search = GridSearchCV(
    SVC(),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# Fit and find best parameters
grid_search.fit(X, y)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")
print(f"Best estimator: {grid_search.best_estimator_}")
```

**Parameter Explanations:**

- `param_grid`: Dictionary with parameter names as keys and lists of values to try
- `cv`: Cross-validation folds (default=5)
- `scoring`: Evaluation metric (default uses estimator's score method)
- `n_jobs`: Parallel jobs (-1 uses all cores)
- `refit`: Whether to refit on full training data with best params (default=True)

### Randomized Search

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, loguniform

# Define parameter distributions
param_dist = {
    'C': loguniform(1e-4, 1e2),  # Log-uniform for wide range
    'gamma': loguniform(1e-4, 1e-1),
    'kernel': ['rbf', 'linear']
}

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
print(f"Best CV score: {random_search.best_score_:.3f}")
```

**RandomizedSearchCV specific parameters:**

- `n_iter`: Number of parameter settings to sample
- `param_distributions`: Distributions to sample from (can use scipy.stats)

### Advanced Parameter Sampling

```python
from sklearn.model_selection import ParameterSampler
from scipy.stats import randint, uniform

# Custom parameter sampling
param_dist = {
    'n_estimators': randint(10, 200),  # Discrete uniform
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': uniform(0.01, 0.19),  # Continuous uniform
    'min_samples_leaf': randint(1, 20)
}

sampler = ParameterSampler(param_dist, n_iter=50, random_state=42)
sampled_params = list(sampler)

print(f"Sampled {len(sampled_params)} parameter combinations")
```

---

## Practical Applications

Let's apply hyperparameter tuning to SVM and Random Forest models using a real dataset.

### SVM Hyperparameter Tuning

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, validation_curve
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load and preprocess data
data = load_breast_cancer()
X, y = data.data, data.target

# Create pipeline with scaling
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC())
])

# Parameter grid for SVM
param_grid_svm = {
    'svm__C': [0.1, 1, 10, 100],
    'svm__gamma': [1, 0.1, 0.01, 0.001],
    'svm__kernel': ['rbf']
}

# Grid search for SVM
grid_search_svm = GridSearchCV(
    pipeline,
    param_grid_svm,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

grid_search_svm.fit(X, y)

print("SVM Grid Search Results:")
print(f"Best parameters: {grid_search_svm.best_params_}")
print(f"Best CV F1: {grid_search_svm.best_score_:.3f}")

# Randomized search comparison
param_dist_svm = {
    'svm__C': loguniform(1e-3, 1e3),
    'svm__gamma': loguniform(1e-4, 1e-1),
    'svm__kernel': ['rbf', 'linear']
}

random_search_svm = RandomizedSearchCV(
    pipeline,
    param_dist_svm,
    n_iter=30,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    random_state=42
)

random_search_svm.fit(X, y)

print("\nSVM Random Search Results:")
print(f"Best parameters: {random_search_svm.best_params_}")
print(f"Best CV F1: {random_search_svm.best_score_:.3f}")
```

### Random Forest Hyperparameter Tuning

```python
from sklearn.ensemble import RandomForestClassifier

# Parameter grid for Random Forest
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# Grid search for Random Forest
grid_search_rf = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid_rf,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

grid_search_rf.fit(X, y)

print("\nRandom Forest Grid Search Results:")
print(f"Best parameters: {grid_search_rf.best_params_}")
print(f"Best CV F1: {grid_search_rf.best_score_:.3f}")

# Randomized search for comparison
param_dist_rf = {
    'n_estimators': randint(50, 300),
    'max_depth': [None] + list(range(10, 31, 5)),
    'min_samples_split': randint(2, 11),
    'min_samples_leaf': randint(1, 5),
    'max_features': ['sqrt', 'log2', None]
}

random_search_rf = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_dist_rf,
    n_iter=50,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    random_state=42
)

random_search_rf.fit(X, y)

print("\nRandom Forest Random Search Results:")
print(f"Best parameters: {random_search_rf.best_params_}")
print(f"Best CV F1: {random_search_rf.best_score_:.3f}")
```

### Validation Curves for Parameter Analysis

```python
# Plot validation curve for SVM C parameter
train_scores, val_scores = validation_curve(
    SVC(kernel='rbf', gamma='scale'),
    X, y,
    param_name='C',
    param_range=[0.01, 0.1, 1, 10, 100],
    cv=5,
    scoring='f1'
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot([0.01, 0.1, 1, 10, 100], train_mean, 'o-', label='Training score')
plt.plot([0.01, 0.1, 1, 10, 100], val_mean, 'o-', label='Cross-validation score')
plt.fill_between([0.01, 0.1, 1, 10, 100], train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.fill_between([0.01, 0.1, 1, 10, 100], val_mean - val_std, val_mean + val_std, alpha=0.1)
plt.xscale('log')
plt.xlabel('C parameter')
plt.ylabel('F1 Score')
plt.title('Validation Curve for SVM C Parameter')
plt.legend()
plt.grid(True)
plt.show()
```

**Interpreting Results:**

The examples demonstrate:
- Grid search systematically explores all parameter combinations
- Random search can find good solutions with fewer evaluations
- Different algorithms require different parameter tuning strategies
- Validation curves help understand parameter sensitivity and overfitting

---

## Expert Insights

### Grid Search vs Random Search

**When to use Grid Search:**
- Small parameter spaces (≤10 parameters)
- Understanding parameter interactions is crucial
- Computational resources are abundant
- Need exhaustive coverage of parameter combinations

**When to use Random Search:**
- Large parameter spaces (>10 parameters)
- Some parameters are more important than others
- Limited computational budget
- Exploring continuous parameter ranges

### Designing Effective Search Spaces

**Parameter Scale Considerations:**
- Use log scale for parameters spanning orders of magnitude (C, gamma)
- Consider parameter interactions (e.g., C and gamma in SVM are related)
- Include boundary values and reasonable defaults

**Algorithm-Specific Guidelines:**

**SVM:**
- C: $[10^{-3}, 10^3]$ log scale
- gamma: $[10^{-4}, 10^{-1}]$ log scale for RBF kernel
- Try both linear and RBF kernels

**Random Forest:**
- n_estimators: [50, 500] - more is usually better
- max_depth: [None, 10, 20, 30] - None allows full growth
- min_samples_split: [2, 10] - higher values prevent overfitting
- max_features: ['sqrt', 'log2', None] - sqrt is common default

### Common Pitfalls

- **Over-tuning on validation data**: Use nested cross-validation
- **Ignoring parameter interactions**: Parameters often interact (e.g., learning rate and n_estimators)
- **Fixed search spaces**: Adapt search space based on initial results
- **Computational inefficiency**: Use parallel processing and smart stopping

### Advanced Techniques

- **Bayesian Optimization**: Uses probabilistic models to guide search
- **Hyperband**: Efficient resource allocation for expensive evaluations
- **Multi-fidelity optimization**: Uses cheaper approximations for initial screening

### Performance Considerations

- Grid search complexity: exponential in number of parameters
- Random search: linear in number of iterations
- Parallelization: Both methods benefit from multiple cores
- Memory usage: Scales with CV folds and parameter combinations

### Best Practices

- Start with wide parameter ranges, then narrow based on results
- Use cross-validation consistently (same CV as final evaluation)
- Report both best parameters and confidence intervals
- Consider domain knowledge when setting parameter bounds
- Validate final model on held-out test set

---

## Self-Check Questions

1. What are the main differences between grid search and random search?
2. Why is it important to use cross-validation during hyperparameter tuning?
3. How should you choose the range of values to search for each hyperparameter?
4. What are the computational trade-offs between grid search and random search?

---

## Try This Exercise

**Comprehensive Hyperparameter Tuning**

1. Load the digits dataset from sklearn
2. Implement grid search and random search for SVM hyperparameters
3. Compare their performance and computational efficiency
4. Plot validation curves for at least two parameters
5. Analyze which parameters have the biggest impact on performance
6. Apply the best hyperparameters to a held-out test set

**Expected Outcome**: You'll gain practical experience with different tuning strategies and understand how to efficiently find optimal model configurations.

---

## Builder's Insight

Hyperparameter tuning is where machine learning transitions from science to craft. While algorithms provide the mathematical foundation, finding the right hyperparameters requires understanding your data, your problem, and the subtle interactions between parameters.

Don't treat tuning as an afterthought—it's often where the biggest performance gains come from. Start systematically, learn from each search, and remember that the best hyperparameters for one dataset might not work for another.

As you become more experienced, you'll develop intuition for which parameters matter most for different problems, allowing you to tune more efficiently and effectively. The goal isn't just finding good parameters—it's understanding why they work.



