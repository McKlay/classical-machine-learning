---
hide:
  - toc
---

# Chapter 12: Hierarchical Clustering

> *"Hierarchies reveal not just groups, but the relationships between groups—the family tree of data."*

---

## Learning Objectives

By the end of this chapter, you will be able to:

- Understand the principles of hierarchical clustering and different linkage methods
- Interpret dendrograms and determine optimal cluster numbers
- Implement agglomerative clustering in scikit-learn with appropriate parameters
- Compare different linkage methods and their effects on clustering results
- Apply hierarchical clustering to real datasets and visualize cluster hierarchies

---

## Intuitive Introduction

Imagine you're organizing a family reunion. You start by grouping immediate family members, then combine families into larger clans, and finally unite clans into the entire extended family. This hierarchical approach creates a "family tree" showing relationships at different levels.

Hierarchical clustering works similarly. It builds a hierarchy of clusters by either:
- **Agglomerative (bottom-up)**: Start with individual points and merge them into larger clusters
- **Divisive (top-down)**: Start with one big cluster and split it into smaller ones

The result is a tree-like structure called a dendrogram that shows how clusters are nested within each other. This approach doesn't require you to specify the number of clusters upfront and provides rich information about cluster relationships.

---

## Mathematical Development

Hierarchical clustering creates a hierarchy of clusters by iteratively merging or splitting clusters based on their similarity.

### Distance Metrics

The foundation of hierarchical clustering is measuring distance between clusters. Common distance metrics include:

**Single Linkage (Minimum)**: Distance between closest points in different clusters
$$d(C_i, C_j) = \min_{x \in C_i, y \in C_j} ||x - y||$$

**Complete Linkage (Maximum)**: Distance between farthest points in different clusters
$$d(C_i, C_j) = \max_{x \in C_i, y \in C_j} ||x - y||$$

**Average Linkage**: Average distance between all pairs of points
$$d(C_i, C_j) = \frac{1}{|C_i| \cdot |C_j|} \sum_{x \in C_i} \sum_{y \in C_j} ||x - y||$$

**Centroid Linkage**: Distance between cluster centroids
$$d(C_i, C_j) = ||\mu_i - \mu_j||$$

Where $\mu_i$ is the centroid (mean) of cluster $C_i$.

### The Agglomerative Algorithm

1. Start with each data point as its own cluster
2. Find the two closest clusters
3. Merge them into a single cluster
4. Update the distance matrix
5. Repeat until only one cluster remains

### Dendrograms

A dendrogram is a tree diagram showing the hierarchical relationships:
- **Height**: Represents the distance between merged clusters
- **Leaves**: Individual data points
- **Branches**: Show cluster merging history

The y-axis represents the linkage distance, helping determine where to "cut" the tree for optimal clustering.

For web sources on hierarchical clustering:
- Scikit-learn documentation: [https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering)
- Wikipedia: Hierarchical clustering algorithms

---

## Implementation Guide

Scikit-learn provides `AgglomerativeClustering` for hierarchical clustering. Let's explore its usage:

### Basic Usage

```python
from sklearn.cluster import AgglomerativeClustering
import numpy as np

# Sample data
X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])

# Perform clustering
clustering = AgglomerativeClustering(n_clusters=2)
labels = clustering.fit_predict(X)

print("Cluster labels:", labels)
```

**Key Parameters:**

- `n_clusters`: Number of clusters to find (optional)
  - If None, builds full hierarchy
  - Default: 2

- `linkage`: Linkage criterion
  - `'ward'`: Minimize variance of clusters (default, requires Euclidean distance)
  - `'complete'`: Complete linkage (maximum distance)
  - `'average'`: Average linkage
  - `'single'`: Single linkage (minimum distance)

- `affinity`: Distance metric
  - `'euclidean'` (default)
  - `'l1'`, `'l2'`, `'manhattan'`, `'cosine'`, `'precomputed'`
  - Auto-selected based on linkage for Ward

- `distance_threshold`: Linkage distance threshold
  - Alternative to `n_clusters`
  - Clusters below this distance are merged

### Advanced Usage

```python
# Using distance threshold instead of n_clusters
clustering = AgglomerativeClustering(
    n_clusters=None,
    distance_threshold=1.5,
    linkage='complete'
)
labels = clustering.fit_predict(X)

# Get number of clusters found
n_clusters_found = clustering.n_clusters_
print(f"Number of clusters found: {n_clusters_found}")
```

**Parameter Interactions:**

- `linkage='ward'` requires `affinity='euclidean'`
- `distance_threshold` and `n_clusters` are mutually exclusive
- Ward linkage tends to create equally-sized clusters

---

## Practical Applications

Let's apply hierarchical clustering to the Wine dataset to demonstrate the technique:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
from scipy.cluster.hierarchy import dendrogram, linkage

# Load and prepare data
wine = load_wine()
X = wine.data
y_true = wine.target
feature_names = wine.feature_names

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform hierarchical clustering with different linkage methods
linkage_methods = ['single', 'complete', 'average', 'ward']
results = {}

for method in linkage_methods:
    clustering = AgglomerativeClustering(
        n_clusters=3,  # We know there are 3 wine classes
        linkage=method
    )
    labels = clustering.fit_predict(X_scaled)
    ari = adjusted_rand_score(y_true, labels)
    results[method] = {'labels': labels, 'ari': ari}

# Create linkage matrices for dendrograms
linkage_matrices = {}
for method in linkage_methods:
    if method == 'ward':
        # Ward requires Euclidean distance
        linkage_matrices[method] = linkage(X_scaled, method=method, metric='euclidean')
    else:
        linkage_matrices[method] = linkage(X_scaled, method=method)

# Plot dendrograms and compare results
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

for i, method in enumerate(linkage_methods):
    # Dendrogram
    plt.subplot(2, 4, i+1)
    dendrogram(linkage_matrices[method], truncate_mode='level', p=3)
    plt.title(f'{method.capitalize()} Linkage Dendrogram')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    
    # Performance comparison
    plt.subplot(2, 4, i+5)
    labels = results[method]['labels']
    ari = results[method]['ari']
    
    # Scatter plot of first two features colored by clusters
    scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.xlabel(f'Standardized {feature_names[0]}')
    plt.ylabel(f'Standardized {feature_names[1]}')
    plt.title(f'{method.capitalize()} Clustering\nARI: {ari:.3f}')
    plt.colorbar(scatter)

plt.tight_layout()
plt.show()

# Print detailed results
print("Clustering Performance Comparison:")
print("-" * 40)
for method, result in results.items():
    ari = result['ari']
    print(f"{method.capitalize():>10}: ARI = {ari:.3f}")

# Analyze cluster characteristics for best method (Ward)
best_method = max(results.keys(), key=lambda x: results[x]['ari'])
best_labels = results[best_method]['labels']

print(f"\nBest Method: {best_method.capitalize()}")
print("Cluster Analysis:")
for cluster in range(3):
    cluster_points = X[best_labels == cluster]
    print(f"\nCluster {cluster}: {len(cluster_points)} samples")
    print(f"  Mean alcohol: {cluster_points[:, 0].mean():.2f}")
    print(f"  Mean malic acid: {cluster_points[:, 1].mean():.2f}")
    print(f"  Mean ash: {cluster_points[:, 2].mean():.2f}")
```

**Interpreting Results:**

The analysis shows different linkage methods produce varying results:
- **Ward linkage** typically performs best (highest ARI score)
- **Single linkage** creates "stringy" clusters, sensitive to noise
- **Complete linkage** creates compact, equally-sized clusters
- **Average linkage** provides balanced performance

The dendrograms reveal the hierarchical structure, with cutting at different heights producing different cluster numbers.

---

## Expert Insights

### Choosing Linkage Methods

**Single Linkage**:
- **Pros**: Handles non-spherical clusters, good for detecting outliers
- **Cons**: Sensitive to noise, creates "chaining" effect
- **Use when**: Data has arbitrary-shaped clusters

**Complete Linkage**:
- **Pros**: Creates compact clusters, less sensitive to outliers
- **Cons**: Breaks large clusters, sensitive to outliers within clusters
- **Use when**: You want tight, spherical clusters

**Average Linkage**:
- **Pros**: Balanced approach, less sensitive to outliers
- **Cons**: Can be computationally expensive
- **Use when**: General-purpose clustering

**Ward Linkage**:
- **Pros**: Tends to create equally-sized clusters, works well with Euclidean distance
- **Cons**: Assumes spherical clusters, requires Euclidean metric
- **Use when**: You want balanced cluster sizes

### Dendrogram Interpretation

**Cutting the Tree**:
- **Horizontal cuts**: Determine cluster membership
- **Height of cuts**: Indicates cluster similarity
- **Branch lengths**: Show within-cluster vs between-cluster distances

**Optimal Cutting**:
- Look for large gaps in linkage distances
- Consider domain knowledge about expected cluster numbers
- Use silhouette analysis to validate cuts

### Computational Considerations

**Complexity**: O(n²) space and O(n³) time for full hierarchy
- **Advantages**: No need to specify k upfront, rich hierarchical information
- **Limitations**: Doesn't scale well to large datasets (>10,000 samples)
- **Solutions**: Use `distance_threshold` to stop early, or sample data

### When to Use Hierarchical Clustering

**Advantages**:
- No need to specify number of clusters upfront
- Provides hierarchical relationships
- Deterministic results (no random initialization)
- Works with any distance metric

**Disadvantages**:
- Cannot "unmerge" clusters once created
- Computationally expensive for large datasets
- Sensitive to choice of linkage method

**Alternatives**:
- **K-means**: Faster, scales better, but requires k specification
- **DBSCAN**: Handles noise and arbitrary shapes, no k needed
- **Gaussian Mixture Models**: Probabilistic approach with soft assignments

---

## Self-Check Questions

1. What is the difference between agglomerative and divisive hierarchical clustering?
2. Why does Ward linkage require Euclidean distance?
3. How do you determine the optimal number of clusters from a dendrogram?
4. When would you choose single linkage over complete linkage?

---

## Try This Exercise

**Hierarchical Clustering on Customer Data**

1. Create or load customer data with features like age, income, spending score, and recency
2. Apply hierarchical clustering with different linkage methods (single, complete, average, ward)
3. Visualize dendrograms for each method
4. Cut the dendrograms at different heights to create 3-5 clusters
5. Compare cluster characteristics across linkage methods
6. Analyze which linkage method creates the most interpretable customer segments

**Expected Outcome**: You'll understand how linkage methods affect cluster formation and learn to interpret dendrograms for business insights.

---

## Builder's Insight

Hierarchical clustering offers a unique perspective on data relationships. While other clustering methods give you flat partitions, hierarchical methods reveal the nested structure of your data—the "Russian dolls" of groupings within groupings.

This approach is particularly valuable when you need to understand not just what groups exist, but how those groups relate to each other. The dendrogram becomes a map of your data's natural hierarchy, guiding decisions about granularity and relationships.

As you build clustering systems, remember that the "right" linkage method depends on your data's structure and your analytical goals. Ward linkage often works well for business data, while single linkage excels at finding unusual patterns. The key is understanding these trade-offs and choosing the method that best serves your analytical narrative.

Mastering hierarchical clustering adds depth to your analytical toolkit, enabling you to uncover not just clusters, but the stories they tell about your data's underlying structure.