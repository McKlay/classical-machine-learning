---
hide:
  - toc
---

# Chapter 6: Decision Trees

> "*A tree is known by its fruit; a man by his deeds. A good deed is never lost; he who sows courtesy reaps friendship, and he who plants kindness gathers love.*" - Saint Basil

---

## Why This Chapter Matters

Decision trees are among the most interpretable machine learning models, mimicking human decision-making by splitting data based on feature values. They form the foundation for powerful ensemble methods like Random Forests and Gradient Boosting.

This chapter explores how trees recursively partition data, their geometric interpretation, and practical implementation in scikit-learn. You'll learn to build, visualize, and tune trees for both classification and regression.

---

## Learning Objectives

By the end of this chapter, you will be able to:

- Understand the recursive splitting process and impurity measures in decision trees
- Implement decision trees in scikit-learn with proper parameter tuning
- Visualize and interpret tree structures and decision boundaries
- Apply pruning and cross-validation to prevent overfitting
- Analyze feature importance for model interpretability

---

## Intuitive Introduction

Imagine diagnosing a medical condition by asking yes/no questions: "Is the patient over 50? Does their blood pressure exceed 140?" Each question splits patients into groups, narrowing down the diagnosis.

Decision trees work similarly, automatically learning these questions from data. They're like flowcharts that classify or predict by following branches based on feature thresholds. Simple, visual, and powerful.

---

## Mathematical Development

**Entropy and Gini Impurity:**
For classification, trees use impurity measures to choose splits.

Entropy: H(S) = -∑ p_i log₂ p_i

Gini: G(S) = 1 - ∑ p_i²

Lower impurity means purer splits. Information gain = impurity before - weighted impurity after.

**Recursive Splitting:**
Start with root node. Choose feature and threshold that maximize information gain. Split data, repeat on children until stopping criteria (max depth, min samples).

**Geometry:**
In feature space, splits create axis-aligned rectangles. The tree partitions space into regions, each assigned a prediction.

**Regression Trees:**
Use variance reduction: Split minimizes weighted variance of targets in children.

Web sources: For entropy and Gini, see [https://en.wikipedia.org/wiki/Decision_tree_learning](https://en.wikipedia.org/wiki/Decision_tree_learning). For implementation details, https://scikit-learn.org/stable/modules/tree.html.

---

## Implementation Guide

### DecisionTreeClassifier API
Key parameters:
- `criterion`: str, default='gini'. 'gini' or 'entropy' for split quality
- `max_depth`: int, default=None. Maximum tree depth
- `min_samples_split`: int or float, default=2. Minimum samples to split
- `min_samples_leaf`: int or float, default=1. Minimum samples per leaf
- `max_features`: int, float or str, default=None. Features to consider for splits
- `random_state`: int, default=None. For reproducibility

Methods: `fit`, `predict`, `predict_proba`, `score`, `feature_importances_`

See: [https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)

Best practices: Use cross-validation for depth tuning. Visualize trees with plot_tree.

---

## Practical Applications

Let's build a decision tree on the Human Activity Recognition (HAR) dataset, visualizing the tree:

```python
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Load HAR dataset (subset for demo)
har = fetch_openml('har', version=1, as_frame=True)
X, y = har.data.iloc[:5000], har.target.iloc[:5000]  # Subset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train decision tree
tree = DecisionTreeClassifier(max_depth=4, random_state=42)
tree.fit(X_train, y_train)

# Evaluate
y_pred = tree.predict(X_test)
print(classification_report(y_test, y_pred))

# Cross-validation for depth tuning
depths = [3, 5, 7, 10]
cv_scores = []
for depth in depths:
    scores = cross_val_score(DecisionTreeClassifier(max_depth=depth, random_state=42), X_train, y_train, cv=5)
    cv_scores.append(scores.mean())
    print(f"Depth {depth}: CV Accuracy = {scores.mean():.3f}")

# Visualize tree
plt.figure(figsize=(20,10))
plot_tree(tree, feature_names=X.columns, class_names=tree.classes_, filled=True, rounded=True)
plt.show()

# Feature importance
importances = tree.feature_importances_
for name, imp in zip(X.columns, importances):
    if imp > 0.01:  # Top features
        print(f"{name}: {imp:.3f}")
```

Interpretation: Tree splits on key features. Cross-validation prevents overfitting. Visualization shows decision logic.

---

## Expert Insights

- **Common Pitfalls**: Overfitting with deep trees. Not pruning or using CV. Ignoring feature scaling (not needed for trees).
- **Debugging Strategies**: Visualize trees to check splits. Use feature_importances_ for interpretability. Plot learning curves.
- **Parameter Selection**: Start with max_depth=5, tune via CV. Use min_samples_split=10 for robustness. Gini is faster than entropy.
- **Advanced Optimization**: Computational complexity O(n log n) for training. Use random forests for better performance.

Remember: Trees are interpretable but prone to overfitting - ensemble them for power.

---

## Self-Check Questions

1. How does a decision tree choose where to split?
2. What's the difference between entropy and Gini impurity?
3. Why might a deep tree overfit?
4. How do you interpret feature importance in trees?
5. When would you prefer a decision tree over linear models?

---

## Try This Exercise

> **Tree Tuning**:  
> Load the Wine dataset from scikit-learn. Train decision trees with varying max_depth. Use cross-validation to find the optimal depth and visualize the best tree. Compare feature importances.

---

## Builder's Insight

Decision trees turn complexity into clarity. Their branching logic mirrors how we think, making ML accessible.

Master trees, and you'll understand the forest.

---
