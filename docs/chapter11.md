---
hide:
  - toc
---

# Chapter 11: K-Means Clustering

> *"Groups emerge not from similarity of appearance, but from shared patterns of behavior."*

---

## Learning Objectives

By the end of this chapter, you will be able to:

- Understand the mathematical foundations of K-means clustering and the concept of centroids
- Implement K-means clustering in scikit-learn with appropriate parameter selection
- Apply the elbow method and silhouette analysis for determining optimal cluster numbers
- Visualize and interpret clustering results in feature space
- Recognize when K-means is appropriate and its limitations

---

## Intuitive Introduction

Imagine you're organizing a library of books. You don't have predefined categories, but you notice books naturally group together based on their content. Mystery novels cluster near each other, science books form another group, and cookbooks gather in a third area. This is clustering in action—finding natural groupings in data without predefined labels.

K-means clustering works similarly. It finds natural groupings by iteratively assigning data points to the nearest "centroid" (the center of a cluster) and then recalculating those centroids based on the assigned points. It's like placing k pins on a map and letting gravity pull them to the centers of population density.

This algorithm is particularly powerful for customer segmentation, image compression, and anomaly detection. However, it assumes spherical clusters of similar sizes and struggles with non-linear boundaries or varying cluster densities.

---

## Mathematical Development

K-means clustering aims to partition n observations into k clusters where each observation belongs to the cluster with the nearest mean (centroid).

### The Objective Function

The algorithm minimizes the within-cluster sum of squares (WCSS):

$$J = \sum_{i=1}^{k} \sum_{x \in S_i} ||x - \mu_i||^2$$

Where:
- $k$ is the number of clusters
- $S_i$ is the set of points in cluster i
- $\mu_i$ is the centroid (mean) of cluster i
- $||x - \mu_i||^2$ is the squared Euclidean distance

### The Algorithm

1. **Initialization**: Choose k initial centroids (randomly or using k-means++)
2. **Assignment**: Assign each point to the nearest centroid
3. **Update**: Recalculate centroids as the mean of points in each cluster
4. **Repeat**: Steps 2-3 until convergence (centroids don't change significantly)

### K-means++ Initialization

To avoid poor initialization, k-means++ selects initial centroids with probability proportional to their squared distance from existing centroids:

$$P(x) = \frac{D(x)^2}{\sum_{x'} D(x')^2}$$

Where $D(x)$ is the distance to the nearest existing centroid.

### Convergence

The algorithm converges when the centroids stabilize. In practice, we stop when:
- Centroids change by less than a threshold, or
- Maximum iterations reached, or
- No points change cluster assignment

For web sources on K-means mathematics:
- Scikit-learn documentation: [https://scikit-learn.org/stable/modules/clustering.html#k-means](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- Original paper: "k-means++: The advantages of careful seeding" by Arthur and Vassilvitskii

---

## Implementation Guide

Scikit-learn's `KMeans` class provides a robust implementation of the algorithm. Let's explore its usage:

### Basic Usage

```python
from sklearn.cluster import KMeans
import numpy as np

# Create sample data
X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])

# Fit K-means
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)

# Get cluster labels and centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

print("Cluster labels:", labels)
print("Centroids:", centroids)
```

**Key Parameters:**

- `n_clusters`: Number of clusters to form (required)
  - Default: 8
  - Must be specified; no automatic determination

- `init`: Method for centroid initialization
  - `'k-means++'` (default): Smart initialization to avoid poor local minima
  - `'random'`: Random initialization
  - `ndarray`: Pre-specified initial centroids

- `n_init`: Number of random initializations
  - Default: 10
  - Higher values reduce chance of poor local optima but increase computation

- `max_iter`: Maximum iterations per initialization
  - Default: 300
  - Usually converges much faster

- `tol`: Tolerance for convergence
  - Default: 1e-4
  - Stop when centroids move less than this distance

- `random_state`: Random seed for reproducibility
  - Important for consistent results

### Advanced Usage

```python
# With custom initialization
initial_centroids = np.array([[0, 0], [5, 5]])
kmeans = KMeans(n_clusters=2, init=initial_centroids, n_init=1)

# Predicting on new data
new_data = np.array([[2, 3], [3, 1]])
new_labels = kmeans.predict(new_data)

# Getting inertia (within-cluster sum of squares)
inertia = kmeans.inertia_
print(f"Final inertia: {inertia}")
```

**Parameter Interactions:**

- `n_init` and `init`: Use `n_init=1` with custom `init` array
- `max_iter` and `tol`: Balance between accuracy and speed
- `random_state`: Always set for reproducible results

---

## Practical Applications

Let's apply K-means to the classic Iris dataset to demonstrate clustering in action:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load and prepare data
iris = load_iris()
X = iris.data
feature_names = iris.feature_names

# Standardize features (important for distance-based algorithms)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine optimal k using elbow method
inertias = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Plot elbow curve
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(k_range, inertias, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.grid(True)

# Silhouette analysis for k=2,3,4
silhouette_scores = []
k_values = [2, 3, 4]

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)

plt.subplot(1, 3, 2)
plt.bar(k_values, silhouette_scores)
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis')
plt.ylim(0, 1)

# Final clustering with k=3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)
centroids = kmeans.cluster_centers_

# Visualize first two features
plt.subplot(1, 3, 3)
scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', alpha=0.7)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, linewidth=3)
plt.xlabel(f'Standardized {feature_names[0]}')
plt.ylabel(f'Standardized {feature_names[1]}')
plt.title('K-means Clustering (k=3)')
plt.colorbar(scatter)

plt.tight_layout()
plt.show()

# Compare with true labels
from sklearn.metrics import adjusted_rand_score

true_labels = iris.target
ari_score = adjusted_rand_score(true_labels, labels)
print(f"Adjusted Rand Index: {ari_score:.3f}")

# Analyze cluster characteristics
for cluster in range(3):
    cluster_points = X[labels == cluster]
    print(f"\nCluster {cluster}:")
    print(f"  Size: {len(cluster_points)} points")
    print(f"  Mean sepal length: {cluster_points[:, 0].mean():.2f}")
    print(f"  Mean sepal width: {cluster_points[:, 1].mean():.2f}")
    print(f"  Mean petal length: {cluster_points[:, 2].mean():.2f}")
    print(f"  Mean petal width: {cluster_points[:, 3].mean():.2f}")
```

**Interpreting Results:**

The elbow method shows a clear "elbow" at k=3, suggesting this is the optimal number of clusters. The silhouette score peaks at k=2 but remains reasonable at k=3. The clustering achieves an Adjusted Rand Index of ~0.62, indicating moderate agreement with the true species labels.

Each cluster shows distinct characteristics:
- Cluster 0: Small flowers (setosa-like)
- Cluster 1: Large flowers (virginica-like)  
- Cluster 2: Medium flowers (versicolor-like)

---

## Expert Insights

### Choosing K

**Elbow Method**: Plot inertia vs k, look for the "elbow" where adding clusters gives diminishing returns.

**Silhouette Analysis**: Measures how similar points are to their cluster vs other clusters. Values > 0.5 indicate good clustering.

**Domain Knowledge**: Sometimes k is known from business requirements (e.g., customer segments).

### Limitations and Pitfalls

**Assumptions**: K-means assumes:
- Spherical clusters of similar size
- Equal variance in all directions
- Euclidean distance is appropriate

**Common Issues**:
- Sensitive to initialization (use k-means++)
- Can converge to local optima (use multiple `n_init`)
- Struggles with non-convex clusters
- Doesn't handle categorical features well

**Alternatives**:
- **Gaussian Mixture Models**: For elliptical clusters
- **DBSCAN**: For arbitrary-shaped clusters and noise handling
- **Hierarchical Clustering**: For nested cluster structures

### Scaling and Performance

**Computational Complexity**: O(n * k * i * d) where n=samples, k=clusters, i=iterations, d=dimensions

**Scaling Tips**:
- Use mini-batch K-means for large datasets
- Reduce dimensions with PCA first
- Consider approximate methods for very large n

### Validation Metrics

**Internal Metrics** (unsupervised):
- Inertia: Within-cluster sum of squares
- Silhouette Score: Cluster cohesion vs separation
- Calinski-Harabasz Index: Ratio of between-cluster to within-cluster dispersion

**External Metrics** (when true labels available):
- Adjusted Rand Index (ARI)
- Adjusted Mutual Information (AMI)
- Homogeneity/Completeness/V-measure

---

## Self-Check Questions

1. Why is it important to standardize features before applying K-means?
2. How does k-means++ initialization improve upon random initialization?
3. What does a high silhouette score indicate about cluster quality?
4. When would you choose DBSCAN over K-means?

---

## Try This Exercise

**Customer Segmentation with K-means**

1. Load a customer dataset (or create synthetic data with features like age, income, spending score)
2. Apply K-means with different values of k (2-8)
3. Use the elbow method and silhouette analysis to determine optimal k
4. Visualize the clusters in 2D (use PCA if high-dimensional)
5. Analyze the characteristics of each customer segment
6. Compare results with different initialization methods

**Expected Outcome**: You'll identify distinct customer segments and understand how initialization and k selection affect clustering results.

---

## Builder's Insight

K-means clustering represents the intersection of mathematical elegance and practical utility. Its simplicity belies its power—most clustering problems can benefit from starting with K-means as a baseline.

Remember: Clustering is often more art than science. The "right" number of clusters depends on your application. Use multiple validation techniques and domain knowledge to guide your decisions.

As you build ML systems, clustering will become a fundamental tool in your arsenal. It reveals hidden patterns in data that supervised methods can't discover. Master the mathematics, understand the limitations, and you'll find clustering invaluable for exploratory data analysis and feature engineering.

The key insight: Sometimes the most valuable discoveries come not from predicting known outcomes, but from uncovering unknown structures in your data.