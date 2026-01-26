# Exploratory Analysis Methods

Comprehensive reference for dimensionality reduction and clustering methods.

## Table of Contents
- [Principal Component Analysis (PCA)](#principal-component-analysis-pca)
- [UMAP](#umap-uniform-manifold-approximation-and-projection)
- [t-SNE](#t-sne-t-distributed-stochastic-neighbor-embedding)
- [Hierarchical Clustering](#hierarchical-clustering)
- [K-Means Clustering](#k-means-clustering)
- [DBSCAN](#dbscan-density-based-spatial-clustering)
- [Method Comparison](#method-comparison)

---

(pca)=

## Principal Component Analysis (PCA)

**Purpose**: Linear dimensionality reduction preserving maximum variance

### Theory

PCA finds orthogonal directions (principal components) that capture most variance in data:

```
X = USVᵀ  (Singular Value Decomposition)
```

Where:
- **U**: Left singular vectors (sample scores)
- **S**: Singular values (eigenvalues)
- **V**: Right singular vectors (loadings)

**Mathematical Steps**:
1. Center data: `X_centered = X - mean(X)`
2. Compute covariance matrix: `C = XᵀX / (n-1)`
3. Eigendecomposition: `C = VΛVᵀ`
4. Project data: `scores = X × V`

### Parameters

| Parameter      | Type      | Range            | Default | Description                                            |
| -------------- | --------- | ---------------- | ------- | ------------------------------------------------------ |
| `n_components` | int/float | 2-100 or 0.0-1.0 | 2       | Number of PCs or variance to retain                    |
| `whiten`       | bool      | -                | False   | Divide components by singular values                   |
| `svd_solver`   | str       | -                | 'auto'  | SVD algorithm ('auto', 'full', 'arpack', 'randomized') |
| `random_state` | int       | -                | 42      | Random seed for reproducibility                        |

**Parameter Guide**:
```python
# Visualization (2D/3D)
n_components = 2  # or 3

# Keep 95% variance
n_components = 0.95

# Keep specific number
n_components = 10

# Whitening (for clustering)
whiten = True
```

### Usage Example

```python
from functions.ML import apply_pca

# Apply PCA
pca_result = apply_pca(
    data=preprocessed_spectra,
    n_components=2,
    labels=group_labels
)

# Access results
scores = pca_result['scores']  # (n_samples, n_components)
loadings = pca_result['loadings']  # (n_features, n_components)
explained_var = pca_result['explained_variance_ratio']
```

### Output Components

**1. Scores** (Sample Coordinates):
```python
scores = pca_result['scores']
# Shape: (n_samples, n_components)
# Each row: sample position in PC space
# Use for: Visualization, clustering
```

**2. Loadings** (Feature Contributions):
```python
loadings = pca_result['loadings']
# Shape: (n_features, n_components)
# Each column: wavenumber contributions to PC
# Use for: Identifying important peaks
```

**3. Explained Variance**:
```python
var_ratio = pca_result['explained_variance_ratio']
# Array of variance % for each PC
# Example: [0.65, 0.23, 0.08, ...]
# PC1 explains 65%, PC2 23%, etc.
```

**4. Scree Plot Data**:
```python
cumulative_var = np.cumsum(var_ratio)
# Cumulative variance explained
```

### Interpretation

#### Scores Plot

```
     PC2 (23%)
         ↑
    A    |    C
    A    |   CC
   AAA   | CCC
  ─────────────→ PC1 (65%)
     BBB|
      BB|
       B|
```

**What it Shows**:
- **Separation**: Groups A, B, C are distinct
- **Distance**: Similar samples cluster together
- **Outliers**: Points far from main cluster

**Interpretation**:
- **Tight clusters**: Homogeneous groups
- **Overlapping**: Spectral similarity
- **Trend**: Continuous variation (e.g., concentration)

#### Loadings Plot

```python
# Positive peaks in PC1 loading
# → Increase PC1 score

# Negative peaks in PC1 loading
# → Decrease PC1 score
```

**Example**:
```
Loading PC1:
  ↑
  |     ___
  |    /   \    ← Important peak
  |___/     \___
  └─────────────→ Wavenumber
  
If Group A has high PC1 scores,
they have strong signal at this peak
```

**Key Wavenumbers**:
```python
# Find most important peaks
top_indices = np.argsort(np.abs(loadings[:, 0]))[-10:]
important_peaks = wavenumbers[top_indices]
```

#### Explained Variance

**Scree Plot**:
```
Variance (%)
100 |●
 80 |  ●
 60 |    ●
 40 |      ●
 20 |        ●●●●●
  0 |________________
     1 2 3 4 5 6 7 8  PC
```

**Rules of Thumb**:
- **PC1 + PC2 > 70%**: Good 2D representation
- **PC1 + PC2 < 50%**: Consider 3D or UMAP
- **Elbow point**: # PCs to retain (here: ~3-4)

### Common Use Cases

#### 1. Group Visualization
```python
# 2D scatter plot
plt.scatter(scores[:, 0], scores[:, 1], c=labels)
plt.xlabel(f'PC1 ({var_ratio[0]:.1%})')
plt.ylabel(f'PC2 ({var_ratio[1]:.1%})')
```

#### 2. Outlier Detection
```python
# Hotelling's T² statistic
from scipy import stats
T2 = np.sum((scores / np.std(scores, axis=0))**2, axis=1)
threshold = stats.chi2.ppf(0.95, df=n_components)
outliers = T2 > threshold
```

#### 3. Feature Selection
```python
# Important wavenumbers from PC1
loading_weights = np.abs(loadings[:, 0])
top_features = np.argsort(loading_weights)[-20:]
```

#### 4. Dimensionality Reduction for ML
```python
# Reduce to 95% variance
pca = apply_pca(data, n_components=0.95)
reduced_data = pca['scores']
# Use reduced_data for classification
```

### Troubleshooting

| Issue                 | Cause                   | Solution                   |
| --------------------- | ----------------------- | -------------------------- |
| Poor separation       | Low variance in PC1-2   | Try more PCs, use UMAP     |
| All groups overlap    | No spectral differences | Check preprocessing        |
| PC1 = baseline        | Preprocessing issue     | Better baseline correction |
| One outlier dominates | Extreme spectrum        | Remove outlier, re-run     |
| Scores look random    | Data not standardized   | Check normalization        |

### Assumptions

✓ **Linear relationships**: PCA finds linear combinations  
✓ **Variance = importance**: Assumes max variance = most informative  
✗ **No scaling**: Apply normalization first  
✗ **Not rotation-invariant**: PC axes arbitrary

### When to Use

**Use PCA when**:
- ✓ Visualizing high-dimensional data
- ✓ Reducing dimensions for ML
- ✓ Identifying important features
- ✓ Quick exploratory analysis
- ✓ Data roughly linear

**Consider alternatives when**:
- ✗ Non-linear structure (use UMAP/t-SNE)
- ✗ Preserving distances (use MDS)
- ✗ Only want clustering (use K-means directly)

### Advanced Options

#### Whitening
```python
# Normalize PC variances (useful before clustering)
pca_result = apply_pca(data, whiten=True)
```

#### Incremental PCA
```python
# For very large datasets
from sklearn.decomposition import IncrementalPCA
ipca = IncrementalPCA(n_components=2)
ipca.partial_fit(data_batch_1)
ipca.partial_fit(data_batch_2)
scores = ipca.transform(data)
```

### Reference
Jolliffe & Cadima (2016). "Principal component analysis: a review and recent developments"

---

(mcr-als)=
## MCR-ALS

**Purpose**: Spectral unmixing / component extraction from mixtures.

MCR-ALS (Multivariate Curve Resolution – Alternating Least Squares) aims to decompose a data matrix $X$ into
concentrations $C$ and component spectra $S$:

$$
X \approx C S^T
$$

**When to use**:
- ✓ Each measured spectrum is a mixture of a small number of underlying “pure” components
- ✓ You want interpretable component spectra and relative contributions

**Typical constraints**:
- Non-negativity on $C$ and/or $S$
- Normalization or closure constraints (depending on experiment)

**Practical notes**:
- Sensitive to initialization; try multiple starts.
- Preprocessing (baseline correction, normalization) usually improves results.

---

## UMAP (Uniform Manifold Approximation and Projection)

**Purpose**: Non-linear dimensionality reduction preserving local and global structure


### Theory
UMAP constructs high-dimensional graph, then optimizes low-dimensional representation:

1. **Build k-nearest neighbor graph** in high-dimensional space
2. **Compute fuzzy simplicial complex** (topological structure)
3. **Optimize low-dimensional layout** preserving topology

**Key Difference from PCA**:
- PCA: Linear projection, preserves variance
- UMAP: Non-linear, preserves topology (neighbors stay neighbors)

### Parameters

| Parameter      | Type  | Range      | Default     | Description                   |
| -------------- | ----- | ---------- | ----------- | ----------------------------- |
| `n_neighbors`  | int   | 2 - 200    | 15          | # neighbors for graph         |
| `min_dist`     | float | 0.0 - 0.99 | 0.1         | Minimum distance in embedding |
| `n_components` | int   | 2 - 3      | 2           | Output dimensions             |
| `metric`       | str   | -          | 'euclidean' | Distance metric               |
| `random_state` | int   | -          | 42          | Random seed                   |

**Parameter Guide**:

```python
# Local structure (tight clusters)
n_neighbors = 5-10
min_dist = 0.0

# Balanced (recommended)
n_neighbors = 15
min_dist = 0.1

# Global structure (broader view)
n_neighbors = 50-100
min_dist = 0.5
```

**n_neighbors**:
- **Low (5-10)**: Focus on local structure, tight clusters
- **Medium (15-30)**: Balanced local/global
- **High (50-200)**: Focus on global structure, looser clusters

**min_dist**:
- **0.0**: Densest packing, tight clusters
- **0.1**: Default, good separation
- **0.5+**: Spread out, overview

### Usage Example

```python
from functions.ML import apply_umap

# Apply UMAP
umap_result = apply_umap(
    data=preprocessed_spectra,
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    labels=group_labels
)

# Access results
embedding = umap_result['embedding']  # (n_samples, 2)
```

### PCA vs UMAP Comparison

| Aspect               | PCA             | UMAP                  |
| -------------------- | --------------- | --------------------- |
| **Type**             | Linear          | Non-linear            |
| **Speed**            | Fast            | Slower                |
| **Preserves**        | Variance        | Topology              |
| **Global structure** | Good            | Excellent             |
| **Local structure**  | Poor            | Excellent             |
| **Deterministic**    | Yes             | No (use random_state) |
| **Interpretability** | High (loadings) | Low (no loadings)     |
| **Outliers**         | Sensitive       | Robust                |

**Decision Guide**:
```
Use PCA when:
- Need feature importance (loadings)
- Want speed
- Linear relationships
- Interpretability critical

Use UMAP when:
- PCA shows overlap
- Need better separation
- Non-linear structure
- Visualization priority
```

### Interpretation

**UMAP Embedding**:
```
     Dim 2
         ↑
    A    |    C
    AA   |   CCC
   AAA   | CC
  ─────────────→ Dim 1
     BB |
     BBB|
       B|
```

**What to Look For**:
- **Clusters**: Distinct groups
- **Distance**: Relative, not absolute
- **Shape**: Cluster density and spread
- **Bridges**: Transitional samples

**Warnings**:
⚠️ **Distances not quantitative**: UMAP preserves topology, not exact distances  
⚠️ **Cluster size misleading**: Doesn't reflect true cluster variance  
⚠️ **Different seeds → different layouts**: Use random_state for reproducibility

### Troubleshooting

| Issue               | Cause                | Solution                    |
| ------------------- | -------------------- | --------------------------- |
| Too many clusters   | n_neighbors too low  | Increase to 30-50           |
| All points together | n_neighbors too high | Decrease to 10-15           |
| Unclear structure   | min_dist too high    | Reduce to 0.05-0.1          |
| Overlapping groups  | Inherent similarity  | Try different preprocessing |
| Different results   | Random seed          | Set random_state=42         |

### When to Use

**Use UMAP when**:
- ✓ PCA shows poor separation
- ✓ Suspect non-linear structure
- ✓ Want beautiful visualizations
- ✓ Exploring complex datasets

**Use PCA when**:
- ✓ Need interpretable features
- ✓ Speed critical
- ✓ Publishing quantitative results

### Reference
McInnes et al. (2018). "UMAP: Uniform Manifold Approximation and Projection"

---

## t-SNE (t-Distributed Stochastic Neighbor Embedding)

**Purpose**: Non-linear dimensionality reduction emphasizing local structure

### Theory

t-SNE preserves pairwise similarities between points:
1. Compute pairwise probabilities in high dimensions (Gaussian)
2. Compute pairwise probabilities in low dimensions (t-distribution)
3. Minimize KL divergence between probability distributions

**vs UMAP**: t-SNE focuses more on local structure, UMAP balances local/global

### Parameters

| Parameter            | Type  | Range      | Default | Description             |
| -------------------- | ----- | ---------- | ------- | ----------------------- |
| `perplexity`         | float | 5 - 50     | 30      | # effective neighbors   |
| `n_iter`             | int   | 250 - 5000 | 1000    | Optimization iterations |
| `learning_rate`      | float | 10 - 1000  | 200     | Step size               |
| `early_exaggeration` | float | 1 - 20     | 12      | Initial separation      |
| `random_state`       | int   | -          | 42      | Random seed             |

**Parameter Guide**:

```python
# Small dataset (n < 100)
perplexity = 5-10
n_iter = 1000

# Medium dataset (n = 100-1000)
perplexity = 30
n_iter = 1000-2000

# Large dataset (n > 1000)
perplexity = 50
n_iter = 2000-5000
```

**Perplexity**:
- Rule of thumb: `5 < perplexity < n_samples/3`
- **Low (5-10)**: Focus on local clusters
- **Medium (30)**: Balanced
- **High (50+)**: Focus on global structure

### Usage Example

```python
from functions.ML import apply_tsne

# Apply t-SNE
tsne_result = apply_tsne(
    data=preprocessed_spectra,
    perplexity=30,
    n_iter=1000,
    labels=group_labels
)

# Access results
embedding = tsne_result['embedding']  # (n_samples, 2)
```

### Interpretation

**t-SNE Output**:
```
Similar to UMAP, but:
- Even tighter clusters
- Less emphasis on global distances
- More sensitive to perplexity
```

**Key Points**:
- ⚠️ **Cluster sizes meaningless**: Don't interpret relative sizes
- ⚠️ **Distances not quantitative**: Within-cluster distances OK, between-cluster not
- ⚠️ **Slow**: Much slower than PCA/UMAP for large datasets

### UMAP vs t-SNE

| Aspect               | UMAP          | t-SNE        |
| -------------------- | ------------- | ------------ |
| **Speed**            | Fast          | Slow         |
| **Global structure** | Better        | Worse        |
| **Local structure**  | Good          | Excellent    |
| **Scalability**      | 100k+ samples | <10k samples |
| **Reproducibility**  | Better        | Worse        |
| **General use**      | Preferred     | Specialized  |

**Recommendation**: Use UMAP unless you specifically need t-SNE's extreme local focus

### Troubleshooting

| Issue                  | Cause               | Solution            |
| ---------------------- | ------------------- | ------------------- |
| Blob without structure | Perplexity too high | Reduce perplexity   |
| Many tiny clusters     | Perplexity too low  | Increase perplexity |
| Not converged          | n_iter too low      | Increase to 2000+   |
| Different results      | Random seed         | Set random_state    |
| Very slow              | Large dataset       | Use UMAP instead    |

### When to Use

**Use t-SNE when**:
- ✓ Small-medium datasets (< 5000 samples)
- ✓ Need extreme local structure emphasis
- ✓ Publication requires it (legacy)

**Use UMAP instead when**:
- ✓ Large datasets
- ✓ Need reproducibility
- ✓ Want global structure too
- ✓ Speed matters

### Reference
van der Maaten & Hinton (2008). "Visualizing Data using t-SNE"

---

## Hierarchical Clustering

**Purpose**: Create tree of nested clusters (dendrogram)

### Theory

**Agglomerative (bottom-up)**:
1. Start: Each point is a cluster
2. Repeat: Merge closest clusters
3. Stop: All points in one cluster

**Result**: Dendrogram showing merge history

### Parameters

| Parameter    | Type | Options                         | Default     | Description              |
| ------------ | ---- | ------------------------------- | ----------- | ------------------------ |
| `linkage`    | str  | ward, complete, average, single | 'ward'      | Cluster distance metric  |
| `metric`     | str  | euclidean, cosine, correlation  | 'euclidean' | Distance function        |
| `n_clusters` | int  | 2 - 20                          | None        | Cut tree to get clusters |

**Linkage Methods**:

| Method       | Distance Between Clusters         | Use Case                    |
| ------------ | --------------------------------- | --------------------------- |
| **Ward**     | Minimizes within-cluster variance | **Recommended for most**    |
| **Complete** | Maximum distance                  | Compact clusters            |
| **Average**  | Average distance                  | Balanced                    |
| **Single**   | Minimum distance                  | Can find elongated clusters |

**Recommendation**: Use **Ward** with **Euclidean** distance

### Usage Example

```python
from functions.ML import apply_hierarchical_clustering

# Apply hierarchical clustering
hc_result = apply_hierarchical_clustering(
    data=preprocessed_spectra,
    linkage='ward',
    metric='euclidean',
    n_clusters=3
)

# Access results
clusters = hc_result['clusters']  # Cluster assignments
linkage_matrix = hc_result['linkage']  # For dendrogram
```

### Dendrogram Interpretation

```
Height
  |
 10|          ┌─┐
  |      ┌───┤ ├───┐
  5|    ┌─┤   └─┘   ├─┐
  |  ┌─┤ └─┐     ┌─┘ ├─┐
  0|  └─┘   └─────┘   └─┘
     A₁ A₂  A₃ A₄   B₁ B₂
```

**How to Read**:
- **Horizontal lines**: Clusters
- **Height**: Distance at merge
- **Vertical lines**: Similarity
- **Cut horizontally**: Define clusters

**Cutting the Tree**:
```python
# Cut at height = 7
# → 2 clusters: [A₁,A₂,A₃,A₄] and [B₁,B₂]

# Cut at height = 3
# → 3 clusters
```

**Cophenetic Distance**:
```python
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

c, coph_dists = cophenet(linkage_matrix, pdist(data))
# c > 0.7: Good representation
# c < 0.5: Poor representation
```

### Visualization

```python
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt

# Plot dendrogram
plt.figure(figsize=(10, 7))
dendrogram(
    linkage_matrix,
    labels=sample_names,
    leaf_rotation=90,
    leaf_font_size=8
)
plt.xlabel('Sample')
plt.ylabel('Distance')
plt.title('Hierarchical Clustering Dendrogram')
plt.tight_layout()
plt.show()
```

### Choosing Number of Clusters

**Method 1: Visual Inspection**
- Look for large height jumps in dendrogram
- Cut before major merge

**Method 2: Elbow Method**
```python
from scipy.cluster.hierarchy import fcluster

distances = []
for k in range(2, 11):
    clusters = fcluster(linkage_matrix, k, criterion='maxclust')
    dist = calculate_within_cluster_distance(data, clusters)
    distances.append(dist)

# Plot and find elbow
plt.plot(range(2, 11), distances)
```

**Method 3: Silhouette Score**
```python
from sklearn.metrics import silhouette_score

scores = []
for k in range(2, 11):
    clusters = fcluster(linkage_matrix, k, criterion='maxclust')
    score = silhouette_score(data, clusters)
    scores.append(score)

# Choose k with highest score
best_k = np.argmax(scores) + 2
```

### Troubleshooting

| Issue                      | Cause                 | Solution              |
| -------------------------- | --------------------- | --------------------- |
| Unbalanced clusters        | Single linkage        | Use Ward linkage      |
| Unclear structure          | Wrong distance metric | Try different metrics |
| Chains instead of clusters | Single linkage        | Use Ward/Complete     |
| Dendrogram too complex     | Too many samples      | Truncate or subset    |

### When to Use

**Use Hierarchical Clustering when**:
- ✓ Want to explore cluster structure
- ✓ Don't know # clusters
- ✓ Need hierarchical relationships
- ✓ Small-medium datasets (< 5000 samples)

**Use K-Means when**:
- ✓ Know # clusters
- ✓ Large datasets
- ✓ Speed critical

### Reference
Müllner (2013). "fastcluster: Fast Hierarchical, Agglomerative Clustering"

---

## K-Means Clustering

**Purpose**: Partition data into K non-overlapping clusters

### Theory

**Algorithm**:
1. Initialize K centroids randomly
2. Assign each point to nearest centroid
3. Update centroids to cluster means
4. Repeat 2-3 until convergence

**Objective**: Minimize within-cluster sum of squares (WCSS)

### Parameters

| Parameter      | Type | Range                 | Default     | Description              |
| -------------- | ---- | --------------------- | ----------- | ------------------------ |
| `n_clusters`   | int  | 2 - 20                | 3           | Number of clusters       |
| `init`         | str  | 'k-means++', 'random' | 'k-means++' | Initialization method    |
| `n_init`       | int  | 10 - 100              | 10          | # random initializations |
| `max_iter`     | int  | 100 - 1000            | 300         | Maximum iterations       |
| `random_state` | int  | -                     | 42          | Random seed              |

**Recommendation**: Use **k-means++** initialization (smarter than random)

### Usage Example

```python
from functions.ML import apply_kmeans

# Apply K-means
kmeans_result = apply_kmeans(
    data=preprocessed_spectra,
    n_clusters=3,
    init='k-means++',
    n_init=10
)

# Access results
clusters = kmeans_result['clusters']  # Cluster assignments
centroids = kmeans_result['centroids']  # Cluster centers
inertia = kmeans_result['inertia']  # WCSS
```

### Choosing Number of Clusters (K)

(elbow-method)=
#### Method 1: Elbow Method

```python
inertias = []
K_range = range(2, 11)

for k in K_range:
    result = apply_kmeans(data, n_clusters=k)
    inertias.append(result['inertia'])

# Plot elbow curve
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia (WCSS)')
plt.title('Elbow Method')
plt.show()

# Look for "elbow" point
```

**Elbow Plot**:
```
Inertia
  |●
  | ●
  |  ●
  |   ●___
  |       ●___●___●
  └─────────────────
   2 3 4 5 6 7 8  K
       ↑
     Elbow (K=4)
```

#### Method 2: Silhouette Score

```python
from sklearn.metrics import silhouette_score

silhouette_scores = []
for k in range(2, 11):
    result = apply_kmeans(data, n_clusters=k)
    score = silhouette_score(data, result['clusters'])
    silhouette_scores.append(score)

# Choose K with highest score
best_k = np.argmax(silhouette_scores) + 2
```

**Silhouette Score**:
- Range: [-1, 1]
- **> 0.7**: Strong structure
- **0.5 - 0.7**: Reasonable structure
- **< 0.5**: Weak structure, try different K

#### Method 3: Gap Statistic

```python
from scipy.cluster.vq import kmeans
from sklearn.metrics import pairwise_distances

def gap_statistic(data, K_max=10, n_references=10):
    gaps = []
    for k in range(1, K_max+1):
        # Cluster actual data
        result = apply_kmeans(data, n_clusters=k)
        actual_wcss = result['inertia']
        
        # Cluster random reference data
        reference_wcss = []
        for _ in range(n_references):
            reference = np.random.uniform(
                data.min(), data.max(), 
                size=data.shape
            )
            ref_result = apply_kmeans(reference, n_clusters=k)
            reference_wcss.append(ref_result['inertia'])
        
        # Gap = log(E[WCSS_ref]) - log(WCSS_actual)
        gap = np.log(np.mean(reference_wcss)) - np.log(actual_wcss)
        gaps.append(gap)
    
    return gaps

# Choose K where gap stops increasing
```

### Cluster Validation

#### Silhouette Analysis

```python
from sklearn.metrics import silhouette_samples
import matplotlib.pyplot as plt

# Compute silhouette scores for each sample
silhouette_vals = silhouette_samples(data, clusters)

# Plot silhouette for each cluster
fig, ax = plt.subplots()
y_lower = 10

for i in range(n_clusters):
    cluster_silhouette = silhouette_vals[clusters == i]
    cluster_silhouette.sort()
    
    size = cluster_silhouette.shape[0]
    y_upper = y_lower + size
    
    ax.fill_betweenx(
        np.arange(y_lower, y_upper),
        0, cluster_silhouette,
        alpha=0.7
    )
    y_lower = y_upper + 10

ax.axvline(silhouette_scores[k-2], color="red", linestyle="--")
ax.set_xlabel("Silhouette Coefficient")
ax.set_ylabel("Cluster")
```

**Good Clustering**:
- All clusters above average line
- Similar widths (balanced sizes)
- All positive values

**Poor Clustering**:
- Clusters below average
- Very different widths
- Negative values (misassigned points)

### Troubleshooting

| Issue               | Cause               | Solution                          |
| ------------------- | ------------------- | --------------------------------- |
| Empty clusters      | K too high          | Reduce K                          |
| Unbalanced sizes    | Poor initialization | Use k-means++                     |
| Different results   | Random init         | Set random_state, increase n_init |
| Not converging      | max_iter too low    | Increase max_iter                 |
| Unexpected clusters | Need normalization  | Apply vector norm                 |

### Assumptions and Limitations

**Assumes**:
- ✓ Clusters are spherical (isotropic)
- ✓ Clusters have similar sizes
- ✓ Clusters are equally dense

**Limitations**:
- ✗ Must specify K in advance
- ✗ Sensitive to outliers
- ✗ Can't find non-convex clusters
- ✗ Random initialization (use n_init=10+)

### When to Use

**Use K-Means when**:
- ✓ Know or can estimate K
- ✓ Clusters roughly spherical
- ✓ Large datasets (fast algorithm)
- ✓ Need deterministic results (set random_state)

**Use Hierarchical when**:
- ✓ Don't know K
- ✓ Want dendrogram
- ✓ Small-medium datasets

**Use DBSCAN when**:
- ✓ Arbitrary cluster shapes
- ✓ Noise points present
- ✓ Unknown K

### Reference
Lloyd (1982). "Least squares quantization in PCM"

---

## DBSCAN (Density-Based Spatial Clustering)

**Purpose**: Find arbitrarily-shaped clusters based on density

### Theory

**Concepts**:
- **Core point**: Has ≥ `min_samples` neighbors within `eps`
- **Border point**: Within `eps` of core point
- **Noise point**: Neither core nor border

**Algorithm**:
1. Find all core points
2. Connect core points within `eps`
3. Assign border points to nearby clusters
4. Mark remaining points as noise (-1)

### Parameters

| Parameter     | Type  | Range           | Default     | Description                       |
| ------------- | ----- | --------------- | ----------- | --------------------------------- |
| `eps`         | float | 0.1 - 10.0      | 0.5         | Maximum distance for neighborhood |
| `min_samples` | int   | 3 - 20          | 5           | Minimum points for core point     |
| `metric`      | str   | euclidean, etc. | 'euclidean' | Distance metric                   |

**Parameter Guide**:

**eps (epsilon)**:
```python
# Determine eps using k-distance graph
from sklearn.neighbors import NearestNeighbors

neighbors = NearestNeighbors(n_neighbors=min_samples)
neighbors.fit(data)
distances, indices = neighbors.kneighbors(data)

# Sort and plot distances to min_samples-th neighbor
sorted_distances = np.sort(distances[:, -1])
plt.plot(sorted_distances)
plt.ylabel(f'Distance to {min_samples}-th Neighbor')
plt.xlabel('Points (sorted)')

# eps = value at "elbow" of curve
```

**min_samples**:
- Rule of thumb: `2 × n_dimensions`
- **3-5**: Detect finer clusters
- **10-20**: More robust to noise

### Usage Example

```python
from functions.ML import apply_dbscan

# Apply DBSCAN
dbscan_result = apply_dbscan(
    data=preprocessed_spectra,
    eps=0.5,
    min_samples=5
)

# Access results
clusters = dbscan_result['clusters']  # -1 = noise
n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
n_noise = list(clusters).count(-1)

print(f"Clusters found: {n_clusters}")
print(f"Noise points: {n_noise}")
```

### Advantages

✓ **No need to specify K**: Finds natural clusters  
✓ **Handles arbitrary shapes**: Not limited to spherical  
✓ **Identifies outliers**: Noise points marked as -1  
✓ **Deterministic**: Same results every run

### Disadvantages

✗ **Sensitive to parameters**: eps and min_samples critical  
✗ **Varying densities**: Struggles with clusters of different densities  
✗ **High dimensions**: Distance-based, suffers curse of dimensionality

### Troubleshooting

| Issue                | Cause                | Solution             |
| -------------------- | -------------------- | -------------------- |
| One giant cluster    | eps too large        | Reduce eps           |
| All points are noise | eps too small        | Increase eps         |
| Too many clusters    | min_samples too low  | Increase min_samples |
| Undetected clusters  | min_samples too high | Reduce min_samples   |

### When to Use

**Use DBSCAN when**:
- ✓ Don't know number of clusters
- ✓ Clusters have arbitrary shapes
- ✓ Outliers present
- ✓ Clusters vary in size (but not density)

**Use K-Means when**:
- ✓ Spherical clusters
- ✓ Know K
- ✓ All points should be clustered

### Reference
Ester et al. (1996). "A density-based algorithm for discovering clusters"

---

## Method Comparison

### Quick Reference Table

| Method           | Type          | Speed | Strengths                               | Limitations                | Best For                            |
| ---------------- | ------------- | ----- | --------------------------------------- | -------------------------- | ----------------------------------- |
| **PCA**          | Linear DR     | ⚡⚡⚡   | Fast, interpretable, feature importance | Linear only                | Quick exploration, ML preprocessing |
| **UMAP**         | Non-linear DR | ⚡⚡    | Preserves global+local, beautiful plots | No feature importance      | Complex data visualization          |
| **t-SNE**        | Non-linear DR | ⚡     | Excellent local structure               | Slow, no global structure  | Small datasets, local patterns      |
| **Hierarchical** | Clustering    | ⚡⚡    | Dendrogram, no K needed                 | Slow for large data        | Exploratory, unknown K              |
| **K-Means**      | Clustering    | ⚡⚡⚡   | Fast, simple                            | Need K, spherical clusters | Known K, large datasets             |
| **DBSCAN**       | Clustering    | ⚡⚡    | Arbitrary shapes, finds outliers        | Sensitive to parameters    | Outliers, complex shapes            |

### Decision Tree

```
START: What's your goal?
│
├─ Visualization
│  │
│  ├─ Quick overview → PCA (2D/3D)
│  │
│  ├─ Poor PCA separation → UMAP
│  │
│  └─ Extreme local focus → t-SNE
│
├─ Clustering
│  │
│  ├─ Know # clusters → K-Means
│  │
│  ├─ Don't know # clusters → Hierarchical (dendrogram)
│  │
│  └─ Arbitrary shapes + outliers → DBSCAN
│
└─ Feature reduction for ML
   │
   ├─ Linear data → PCA (keep loadings)
   │
   └─ Non-linear data → UMAP → then ML
```

### Typical Workflow

**1. Initial Exploration**:
```python
# Start with PCA (fast)
pca_result = apply_pca(data, n_components=2)
plot_pca_scores(pca_result)

# Check explained variance
if pca_result['explained_variance_ratio'][:2].sum() < 0.5:
    # Poor separation, try UMAP
    umap_result = apply_umap(data, n_neighbors=15)
```

**2. Clustering Investigation**:
```python
# Try hierarchical first (explore # clusters)
hc_result = apply_hierarchical_clustering(data, linkage='ward')
plot_dendrogram(hc_result)

# Identify optimal K from dendrogram
optimal_k = 3  # from visual inspection

# Apply K-means with optimal K
kmeans_result = apply_kmeans(data, n_clusters=optimal_k)
```

**3. Validation**:
```python
# Validate clustering quality
from sklearn.metrics import silhouette_score, davies_bouldin_score

sil_score = silhouette_score(data, clusters)
db_score = davies_bouldin_score(data, clusters)

print(f"Silhouette: {sil_score:.3f} (higher better)")
print(f"Davies-Bouldin: {db_score:.3f} (lower better)")
```

---

## Validation Metrics

### Silhouette Score

**Formula**:
```
s(i) = (b(i) - a(i)) / max(a(i), b(i))

where:
a(i) = avg distance to points in same cluster
b(i) = avg distance to points in nearest different cluster
```

**Interpretation**:
- **+1**: Perfect clustering
- **0**: On cluster boundary
- **-1**: Misassigned

**Usage**:
```python
from sklearn.metrics import silhouette_score
score = silhouette_score(data, clusters)
```

### Davies-Bouldin Index

**Measures**: Ratio of within-cluster to between-cluster distances

**Interpretation**:
- **Lower is better**
- **0**: Perfect clustering

**Usage**:
```python
from sklearn.metrics import davies_bouldin_score
score = davies_bouldin_score(data, clusters)
```

### Calinski-Harabasz Index

**Measures**: Ratio of between-cluster to within-cluster variance

**Interpretation**:
- **Higher is better**

**Usage**:
```python
from sklearn.metrics import calinski_harabasz_score
score = calinski_harabasz_score(data, clusters)
```

---

## Best Practices

### General Guidelines

1. **Preprocessing First**:
   ```python
   # Always preprocess before analysis
   data = apply_baseline_correction(raw_data)
   data = apply_smoothing(data)
   data = apply_normalization(data)
   ```

2. **Try Multiple Methods**:
   ```python
   # Compare different approaches
   pca_result = apply_pca(data)
   umap_result = apply_umap(data)
   # Choose based on results
   ```

3. **Validate Results**:
   ```python
   # Check quality metrics
   silhouette = silhouette_score(data, clusters)
   if silhouette < 0.5:
       print("Warning: Poor clustering quality")
   ```

4. **Use Domain Knowledge**:
   - Does clustering match expected groups?
   - Are separated groups biologically meaningful?
   - Check against known standards

### Reproducibility

```python
# Always set random state
pca_result = apply_pca(data, random_state=42)
umap_result = apply_umap(data, random_state=42)
kmeans_result = apply_kmeans(data, random_state=42, n_init=10)
```

---

## See Also

- [Analysis User Guide](../user-guide/analysis.md) - Step-by-step tutorials
- [Statistical Methods](statistical.md) - Hypothesis testing
- [Machine Learning Methods](machine-learning.md) - Classification algorithms
- [Best Practices](../user-guide/best-practices.md) - Analysis strategies

---

**Last Updated**: 2026-01-24
