---
hide:
  - toc
---

# Chapter 7: Support Vector Machines (SVM)

> "*The best way to have a good idea is to have a lot of ideas.*" - Linus Pauling

---

## Why This Chapter Matters

Support Vector Machines (SVMs) are powerful classifiers that find the optimal hyperplane separating classes with maximum margin. They excel in high-dimensional spaces and handle non-linear data through kernels.

This chapter explores SVM's geometric foundations, kernel tricks, and practical implementation in scikit-learn. You'll learn to tune SVMs for complex datasets and understand their computational trade-offs.

---

## Learning Objectives

By the end of this chapter, you will be able to:

- Understand SVM's margin maximization and support vector concepts
- Implement kernel methods for non-linear classification
- Apply SVMs in scikit-learn with proper parameter tuning
- Visualize decision boundaries and support vectors
- Diagnose overfitting and select appropriate kernels

---

## Intuitive Introduction

Imagine separating two groups of points with a line (or plane in higher dimensions). SVM finds the "best" line that maximizes the gap between classes, making it robust to new data.

For non-linear data, SVM uses kernels to map points to higher dimensions where separation becomes linear. It's like folding the paper so scattered points align perfectly.

---

## Mathematical Development

**Margin Maximization:**	
For linearly separable data, SVM maximizes the margin: 
$$\frac{2}{\|w\|}$$, subject to $$y_i(w \cdot x_i + b) \geq 1$$.

**Dual Formulation:**	
Using Lagrange multipliers $\alpha_i$, the problem becomes:
$$\max \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j (x_i \cdot x_j)$$

**Kernels:**	
For non-linear, use kernel $K(x_i, x_j) = \phi(x_i) \cdot \phi(x_j)$, e.g., RBF: $$K(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2)$$

**Geometry:**	
Support vectors define the margin; decision boundary is hyperplane in feature space.

Web sources: For SVM theory, see [https://en.wikipedia.org/wiki/Support_vector_machine](https://en.wikipedia.org/wiki/Support_vector_machine). For kernels, [https://scikit-learn.org/stable/modules/svm.html](https://scikit-learn.org/stable/modules/svm.html).

---

## Implementation Guide

### SVC API
Key parameters:  
- `C`: float, default=1.0. Regularization parameter (higher = less regularization)  
- `kernel`: str, default='rbf'. 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'  
- `gamma`: str or float, default='scale'. Kernel coefficient for 'rbf', 'poly', 'sigmoid'  
- `degree`: int, default=3. Degree for 'poly' kernel  
- `coef0`: float, default=0.0. Independent term for 'poly' and 'sigmoid'  

Methods: `fit`, `predict`, `predict_proba`, `decision_function`, `support_vectors_`

See: [https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)

Best practices: Scale features. Use 'rbf' as default kernel. Tune C and gamma via grid search.

---

## Practical Applications

Let's train an RBF SVM on the HAR dataset with PCA for dimensionality reduction:

```python
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np

# Load HAR dataset
har = fetch_openml('har', version=1, as_frame=True)
X, y = har.data.iloc[:3000], har.target.iloc[:3000]  # Subset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Pipeline: scale, PCA, SVM
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=50)),  # Reduce dimensions
    ('svm', SVC(kernel='rbf', random_state=42))
])

# Grid search for C and gamma
param_grid = {'svm__C': [0.1, 1, 10], 'svm__gamma': [0.01, 0.1, 1]}
grid = GridSearchCV(pipeline, param_grid, cv=3)
grid.fit(X_train, y_train)

print(f"Best params: {grid.best_params_}")
print(f"Best CV score: {grid.best_score_:.3f}")

# Evaluate
y_pred = grid.predict(X_test)
print(classification_report(y_test, y_pred))

# Visualize decision boundary (2D PCA projection)
X_train_pca = grid.best_estimator_['pca'].transform(grid.best_estimator_['scaler'].transform(X_train))
X_test_pca = grid.best_estimator_['pca'].transform(grid.best_estimator_['scaler'].transform(X_test))

# Fit SVM on 2D PCA
svm_2d = SVC(kernel='rbf', C=grid.best_params_['svm__C'], gamma=grid.best_params_['svm__gamma'])
X_train_2d = X_train_pca[:, :2]
svm_2d.fit(X_train_2d, y_train)

# Plot
x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train, s=20, edgecolor='k')
plt.scatter(svm_2d.support_vectors_[:, 0], svm_2d.support_vectors_[:, 1], s=100, facecolors='none', edgecolors='r')
plt.title('SVM Decision Boundary with Support Vectors')
plt.show()
```

Interpretation: SVM with RBF kernel handles non-linear data. PCA reduces dimensions. Support vectors highlight margin-defining points.

---

## Expert Insights

- **Common Pitfalls**: Not scaling features. Overfitting with high C/low gamma. Choosing wrong kernel.
- **Debugging Strategies**: Plot decision boundaries. Check support vectors. Use cross-validation for tuning.
- **Parameter Selection**: Start with C=1, gamma='scale'. Use 'linear' for high-dim sparse data, 'rbf' otherwise.
- **Advanced Optimization**: Computational complexity $O(n^2)$ for training, use libsvm for efficiency.

Remember: SVMs are powerful but require careful tuning - they're not "set and forget" models.

---

## Self-Check Questions

1. How does SVM maximize the margin?
2. What's the role of kernels in SVM?
3. How do you choose between different kernels?
4. What are support vectors?
5. When would you prefer SVM over other classifiers?

---

## Try This Exercise

> **SVM Kernel Comparison**:  
> Load the Wine dataset from scikit-learn. Train SVMs with 'linear', 'poly', and 'rbf' kernels. Compare accuracies and visualize decision boundaries for 2D projections. Tune parameters and analyze support vectors.

---

## Builder's Insight

SVMs turn geometry into classification power. Their kernel trick opens doors to infinite dimensions.

Master margins and kernels, and you'll handle any separation challenge.

---

