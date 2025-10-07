---
hide:
  - toc
---

# Chapter 5: K-Nearest Neighbors (KNN)

> "*The best way to predict the future is to look at the past.*" - Unknown

---

## Why This Chapter Matters

K-Nearest Neighbors (KNN) is one of the simplest yet most intuitive machine learning algorithms. It classifies data points based on the majority vote of their nearest neighbors, making it a powerful baseline and easy to understand.

This chapter explores KNN's geometry in feature space, its distance-based decision making, and practical implementation in scikit-learn. You'll learn when KNN excels and its limitations, particularly the curse of dimensionality.

---

## Learning Objectives

By the end of this chapter, you will be able to:

- Understand KNN's distance-based classification and regression principles
- Implement KNN in scikit-learn with appropriate parameter selection
- Visualize and interpret KNN decision boundaries in feature space
- Apply cross-validation to tune the k parameter effectively
- Recognize when KNN is suitable and its computational trade-offs

---

## Intuitive Introduction

Imagine you're trying to classify a new fruit based on its size and color. KNN works by looking at the k closest known fruits in a "feature space" and voting on the most common type among them. If most nearby fruits are apples, it predicts apple.

This lazy learning approach doesn't build a model during training; it stores the data and computes predictions on-the-fly. It's like asking your neighbors for advice - simple, intuitive, and often surprisingly effective.

---

## Mathematical Development

**Distance Metrics (Euclidean):**
KNN uses distance to find nearest neighbors. The most common is Euclidean distance:

d(x, y) = sqrt(∑(x_i - y_i)²)

For multi-dimensional data, this measures straight-line distance in feature space.

**Voting in Feature Space:**
For classification: Find k nearest neighbors, predict the majority class.
For regression: Predict the average of k nearest targets.

**Geometry:**
In 2D feature space, KNN creates irregular decision boundaries based on local densities. Voronoi diagrams illustrate how space is partitioned by nearest neighbors.

**Curse of Dimensionality:**
As dimensions increase, distances become uniform, making neighbors less meaningful. Performance degrades in high dimensions.

---

## Implementation Guide

### KNeighborsClassifier API
Key parameters:
- `n_neighbors`: int, default=5. Number of neighbors to use
- `weights`: str or callable, default='uniform'. 'uniform' (equal weights) or 'distance' (inverse distance weighting)
- `metric`: str or callable, default='minkowski'. Distance metric ('euclidean', 'manhattan', 'minkowski')
- `p`: int, default=2. Power parameter for Minkowski metric (p=2 is Euclidean)
- `algorithm`: str, default='auto'. Algorithm for neighbor search ('auto', 'ball_tree', 'kd_tree', 'brute')

Methods: `fit`, `predict`, `predict_proba`, `kneighbors`, `score`

See: [https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)

Best practices: Scale features for distance-based methods. Use odd k for binary classification to avoid ties.

---

## Practical Applications

Let's classify the Iris dataset with KNN, varying k and visualizing decision boundaries:

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Load data
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Test different k values
k_values = [1, 3, 5, 7, 9]
cv_scores = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5)
    cv_scores.append(scores.mean())
    print(f"k={k}: CV Accuracy = {scores.mean():.3f}")

# Best k
best_k = k_values[np.argmax(cv_scores)]
print(f"Best k: {best_k}")

# Train and evaluate
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train_scaled, y_train)
y_pred = knn_best.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

# Visualize decision boundary (2D projection)
X_2d = X_train_scaled[:, :2]  # First two features
knn_2d = KNeighborsClassifier(n_neighbors=best_k)
knn_2d.fit(X_2d, y_train)

# Create mesh grid
x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Predict on mesh
Z = knn_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_train, s=20, edgecolor='k')
plt.title(f'KNN Decision Boundary (k={best_k})')
plt.xlabel('Feature 1 (scaled)')
plt.ylabel('Feature 2 (scaled)')
plt.show()
```

Interpretation: KNN performs well on scaled data. Cross-validation helps select optimal k. Decision boundaries show how KNN adapts to local data distributions.

---

## Expert Insights

- **Common Pitfalls**: Not scaling features (KNN is distance-sensitive). Using even k for binary classification. High computational cost for large datasets.
- **Debugging Strategies**: Plot decision boundaries for 2D data. Check neighbor distances with kneighbors(). Use cross-validation for k selection.
- **Parameter Selection**: Start with k=5, tune via CV. Use 'distance' weights for non-uniform densities. Manhattan distance for high-dimensional sparse data.
- **Advanced Optimization**: Computational complexity O(n*d) for fit, O(k*d) per prediction. Use ball_tree or kd_tree for efficiency in low dimensions.

Remember: KNN is interpretable but doesn't scale well - use for small datasets or as a baseline.

---

## Self-Check Questions

1. How does KNN determine class predictions?
2. Why is feature scaling important for KNN?
3. What's the curse of dimensionality and how does it affect KNN?
4. How do you choose the optimal value of k?
5. When would you prefer KNN over other algorithms?

---

## Try This Exercise

> **KNN Exploration**:  
> Load the Breast Cancer dataset from scikit-learn. Compare KNN with different distance metrics (Euclidean, Manhattan) and weighting schemes (uniform, distance). Use cross-validation to find the best k and analyze the confusion matrix.

---

## Builder's Insight

KNN teaches the value of simplicity. It may not be the most advanced algorithm, but its geometric intuition builds your understanding of how ML works in feature space.

Start with the basics like KNN - they'll reveal the essence of machine learning.

---

