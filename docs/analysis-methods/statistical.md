# Statistical Analysis Methods

Comprehensive reference for hypothesis testing and statistical methods.

## Table of Contents
- [T-Tests](#t-tests)
- [Mann-Whitney U Test](#mann-whitney-u-test)
- [ANOVA](#anova-analysis-of-variance)
- [Kruskal-Wallis Test](#kruskal-wallis-test)
- [Correlation Analysis](#correlation-analysis)
- [Multiple Testing Correction](#multiple-testing-correction)
- [Effect Size Measures](#effect-size-measures)
- [Band Ratio Analysis](#band-ratio-analysis)

---

## T-Tests

**Purpose**: Compare means of two groups

### Types of T-Tests

#### 1. Independent Samples T-Test

**Use When**: Comparing two independent groups

**Assumptions**:
1. ✓ Continuous data
2. ✓ Independent samples
3. ✓ Normally distributed (each group)
4. ✓ Equal variances (for Student's t-test)

**Two Variants**:

| Variant              | Variance Assumption | When to Use             |
| -------------------- | ------------------- | ----------------------- |
| **Student's t-test** | Equal variances     | Classic version         |
| **Welch's t-test**   | Unequal variances   | **Recommended default** |

**Parameters**:

| Parameter     | Type | Default     | Description                   |
| ------------- | ---- | ----------- | ----------------------------- |
| `equal_var`   | bool | False       | Use equal variance assumption |
| `alternative` | str  | 'two-sided' | Alternative hypothesis        |

**Usage Example**:
```python
from scipy import stats

# Welch's t-test (recommended)
t_stat, p_value = stats.ttest_ind(
    group1_data,
    group2_data,
    equal_var=False,  # Welch's
    alternative='two-sided'
)

print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_value:.4f}")

# Interpretation
alpha = 0.05
if p_value < alpha:
    print("Significant difference between groups")
else:
    print("No significant difference")
```

**Alternative Hypotheses**:
```python
# Two-sided (default): Groups are different
alternative='two-sided'

# One-sided: Group 1 > Group 2
alternative='greater'

# One-sided: Group 1 < Group 2
alternative='less'
```

**Checking Assumptions**:

```python
# 1. Normality test (Shapiro-Wilk)
stat1, p1 = stats.shapiro(group1_data)
stat2, p2 = stats.shapiro(group2_data)

if p1 < 0.05 or p2 < 0.05:
    print("Warning: Data not normally distributed")
    print("Consider: Mann-Whitney U test instead")

# 2. Equal variance test (Levene's test)
stat, p = stats.levene(group1_data, group2_data)

if p < 0.05:
    print("Unequal variances detected")
    print("Use: Welch's t-test (equal_var=False)")
else:
    print("Equal variances")
    print("Can use: Student's t-test (equal_var=True)")
```

#### 2. Paired Samples T-Test

**Use When**: Same subjects measured twice (before/after)

**Example**: Same sample measured at two time points

**Usage Example**:
```python
from scipy import stats

# Paired t-test
t_stat, p_value = stats.ttest_rel(
    before_treatment,
    after_treatment
)

print(f"Paired t-statistic: {t_stat:.3f}")
print(f"p-value: {p_value:.4f}")
```

**Assumptions**:
1. ✓ Paired observations (same subjects)
2. ✓ Differences normally distributed
3. ✓ Continuous data

#### 3. One-Sample T-Test

**Use When**: Compare sample mean to known value

**Usage Example**:
```python
from scipy import stats

# Test if mean differs from theoretical value
reference_value = 0

t_stat, p_value = stats.ttest_1samp(
    data,
    reference_value
)
```

### Interpretation

**P-Value**:
- **p < 0.05**: Significant difference (reject null hypothesis)
- **p ≥ 0.05**: No significant difference (fail to reject null)

**T-Statistic**:
- **Large |t|**: Groups are far apart
- **Small |t|**: Groups are similar

**Confidence Interval** (95%):
```python
from scipy import stats
import numpy as np

# Calculate 95% CI for mean difference
diff = np.mean(group1) - np.mean(group2)
se = stats.sem(np.concatenate([group1, group2]))
ci = stats.t.interval(0.95, len(group1)+len(group2)-2, 
                      loc=diff, scale=se)

print(f"Mean difference: {diff:.3f}")
print(f"95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")

# If CI doesn't include 0 → significant difference
```

### Troubleshooting

| Issue                | Cause                 | Solution                             |
| -------------------- | --------------------- | ------------------------------------ |
| Non-normal data      | Assumption violated   | Use Mann-Whitney U                   |
| Unequal variances    | Assumption violated   | Use Welch's t-test (equal_var=False) |
| Small sample size    | Low power             | Increase n or use non-parametric     |
| Outliers present     | Skewed distribution   | Remove outliers or use robust test   |
| Multiple comparisons | Inflated Type I error | Apply Bonferroni correction          |

### When to Use

**Use T-Test when**:
- ✓ Two groups to compare
- ✓ Continuous data
- ✓ Approximately normal distribution
- ✓ Independent observations

**Use alternatives when**:
- ✗ Non-normal data → Mann-Whitney U
- ✗ More than 2 groups → ANOVA
- ✗ Paired data → Paired t-test
- ✗ Categorical outcome → Chi-square test

---

## Mann-Whitney U Test

**Alternative Names**: Wilcoxon rank-sum test

**Purpose**: Non-parametric alternative to independent t-test

### Theory

**How It Works**:
1. Rank all observations (both groups combined)
2. Sum ranks for each group
3. Test if rank sums differ significantly

**Key Advantage**: No normality assumption

### Assumptions

1. ✓ Independent observations
2. ✓ Ordinal or continuous data
3. ✓ Similar distributions (for testing medians)

**Does NOT require**:
- ✗ Normal distribution
- ✗ Equal variances

### Usage Example

```python
from scipy import stats

# Mann-Whitney U test
u_stat, p_value = stats.mannwhitneyu(
    group1_data,
    group2_data,
    alternative='two-sided'
)

print(f"U-statistic: {u_stat:.1f}")
print(f"p-value: {p_value:.4f}")

# Effect size (rank-biserial correlation)
n1, n2 = len(group1_data), len(group2_data)
r = 1 - (2*u_stat) / (n1 * n2)
print(f"Effect size (r): {r:.3f}")
```

### Interpretation

**Null Hypothesis**: Distributions are identical

**P-Value**:
- **p < 0.05**: Distributions differ significantly
- **p ≥ 0.05**: No significant difference

**Effect Size (r)**:
- **r = 0**: No effect
- **r = ±0.1**: Small effect
- **r = ±0.3**: Medium effect
- **r = ±0.5**: Large effect

### When to Use

**Use Mann-Whitney when**:
- ✓ Non-normal data
- ✓ Ordinal data (e.g., Likert scales)
- ✓ Outliers present
- ✓ Small sample sizes
- ✓ T-test assumptions violated

**Use T-Test instead when**:
- ✓ Normal data
- ✓ Large samples (n > 30 per group)
- ✓ Want parametric inference

---

## ANOVA (Analysis of Variance)

**Purpose**: Compare means across 3+ groups

### One-Way ANOVA

**Use When**: One categorical factor, 3+ groups

**Assumptions**:
1. ✓ Independent observations
2. ✓ Normal distribution (each group)
3. ✓ Equal variances (homoscedasticity)

### Usage Example

```python
from scipy import stats

# One-way ANOVA
f_stat, p_value = stats.f_oneway(
    group1_data,
    group2_data,
    group3_data
)

print(f"F-statistic: {f_stat:.3f}")
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print("Significant difference among groups")
    print("Proceed to post-hoc tests")
```

### Post-Hoc Tests

**Why Needed**: ANOVA tells you groups differ, not which ones

**Common Post-Hoc Tests**:

#### 1. Tukey's HSD (Honest Significant Difference)

**Most Conservative**: Controls family-wise error rate

```python
from scipy.stats import tukey_hsd

# Tukey's HSD
result = tukey_hsd(group1, group2, group3)

# Pairwise comparisons
print("Pairwise p-values:")
print(result.pvalue)

# Confidence intervals
print("\nConfidence intervals for mean differences:")
print(result.confidence_interval())
```

#### 2. Bonferroni Correction

**Simple Method**: Divide α by number of comparisons

```python
from scipy import stats

# Number of pairwise comparisons
n_groups = 3
n_comparisons = n_groups * (n_groups - 1) / 2  # 3 comparisons

# Adjusted significance level
alpha_corrected = 0.05 / n_comparisons  # 0.05 / 3 = 0.0167

# Pairwise t-tests
pairs = [
    ('Group1', 'Group2', group1, group2),
    ('Group1', 'Group3', group1, group3),
    ('Group2', 'Group3', group2, group3)
]

for name1, name2, data1, data2 in pairs:
    t_stat, p_value = stats.ttest_ind(data1, data2)
    sig = "***" if p_value < alpha_corrected else "ns"
    print(f"{name1} vs {name2}: p={p_value:.4f} {sig}")
```

#### 3. Holm-Bonferroni Method

**Less Conservative**: Step-down procedure

```python
from statsmodels.stats.multitest import multipletests

# Collect all p-values
p_values = []
for name1, name2, data1, data2 in pairs:
    _, p = stats.ttest_ind(data1, data2)
    p_values.append(p)

# Apply Holm-Bonferroni correction
reject, p_corrected, _, _ = multipletests(
    p_values,
    alpha=0.05,
    method='holm'
)

# Display results
for i, (name1, name2, _, _) in enumerate(pairs):
    sig = "***" if reject[i] else "ns"
    print(f"{name1} vs {name2}: p={p_corrected[i]:.4f} {sig}")
```

### Checking Assumptions

```python
# 1. Normality (each group)
from scipy import stats

for i, group in enumerate([group1, group2, group3]):
    stat, p = stats.shapiro(group)
    print(f"Group {i+1} normality: p={p:.4f}")
    if p < 0.05:
        print("  → Not normal, consider Kruskal-Wallis")

# 2. Homogeneity of variance (Levene's test)
stat, p = stats.levene(group1, group2, group3)
print(f"\nLevene's test: p={p:.4f}")
if p < 0.05:
    print("  → Unequal variances")
    print("  → Consider Welch's ANOVA")
```

### Welch's ANOVA

**When**: Unequal variances detected

```python
# Welch's ANOVA (unequal variances)
# Not directly in scipy, use one-way ANOVA with robust method
from scipy import stats

# Alternative: Use stats library with robust option
# or implement Welch's ANOVA manually
```

### Interpretation

**F-Statistic**:
- **Large F**: Groups differ substantially
- **Small F**: Groups similar

**P-Value**:
- **p < 0.05**: At least one group differs
- **p ≥ 0.05**: No significant differences

**Effect Size (Eta-Squared)**:
```python
# Calculate eta-squared (proportion of variance explained)
SS_between = sum([len(g) * (np.mean(g) - grand_mean)**2 
                  for g in [group1, group2, group3]])
SS_total = sum([(x - grand_mean)**2 
                for g in [group1, group2, group3] 
                for x in g])

eta_squared = SS_between / SS_total
print(f"η² = {eta_squared:.3f}")

# Interpretation:
# η² = 0.01: Small effect
# η² = 0.06: Medium effect
# η² = 0.14: Large effect
```

### When to Use

**Use ANOVA when**:
- ✓ 3+ groups to compare
- ✓ One categorical factor
- ✓ Continuous outcome
- ✓ Assumptions met

**Use alternatives when**:
- ✗ Two groups only → T-test
- ✗ Non-normal data → Kruskal-Wallis
- ✗ Unequal variances → Welch's ANOVA
- ✗ Repeated measures → Repeated-measures ANOVA

---

## Kruskal-Wallis Test

**Purpose**: Non-parametric alternative to one-way ANOVA

### Theory

**How It Works**:
1. Rank all observations across groups
2. Compare rank sums
3. Test if medians differ

**Null Hypothesis**: All groups have identical distributions

### Assumptions

1. ✓ Independent observations
2. ✓ Ordinal or continuous data
3. ✓ Similar distribution shapes

**Does NOT require**:
- ✗ Normal distribution
- ✗ Equal variances

### Usage Example

```python
from scipy import stats

# Kruskal-Wallis test
h_stat, p_value = stats.kruskal(
    group1_data,
    group2_data,
    group3_data
)

print(f"H-statistic: {h_stat:.3f}")
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print("Significant difference among groups")
    print("Proceed to post-hoc tests")
```

### Post-Hoc Tests

**Dunn's Test**: Pairwise comparisons with rank-based approach

```python
from scikit_posthocs import posthoc_dunn

# Dunn's test for pairwise comparisons
import pandas as pd

# Prepare data
data_all = np.concatenate([group1, group2, group3])
labels_all = (['Group1']*len(group1) + 
              ['Group2']*len(group2) + 
              ['Group3']*len(group3))

df = pd.DataFrame({'value': data_all, 'group': labels_all})

# Dunn's test
dunn_result = posthoc_dunn(df, val_col='value', group_col='group', 
                           p_adjust='bonferroni')

print("\nDunn's test (Bonferroni-corrected p-values):")
print(dunn_result)
```

### When to Use

**Use Kruskal-Wallis when**:
- ✓ 3+ groups to compare
- ✓ Non-normal data
- ✓ Ordinal data
- ✓ ANOVA assumptions violated
- ✓ Outliers present

---

## Correlation Analysis

**Purpose**: Measure association between two variables

### Pearson Correlation

**Measures**: Linear relationship strength

**Assumptions**:
1. ✓ Continuous data
2. ✓ Linear relationship
3. ✓ Bivariate normal distribution
4. ✓ No outliers

### Usage Example

```python
from scipy import stats

# Pearson correlation
r, p_value = stats.pearsonr(variable1, variable2)

print(f"Pearson r: {r:.3f}")
print(f"p-value: {p_value:.4f}")

# Interpretation
if p_value < 0.05:
    if r > 0:
        print("Significant positive correlation")
    else:
        print("Significant negative correlation")
```

**Interpretation of r**:
- **r = 1**: Perfect positive correlation
- **r = 0.7-0.9**: Strong correlation
- **r = 0.4-0.6**: Moderate correlation
- **r = 0.1-0.3**: Weak correlation
- **r = 0**: No correlation
- **r = -1**: Perfect negative correlation

**Coefficient of Determination (R²)**:
```python
r_squared = r ** 2
print(f"R² = {r_squared:.3f}")
# R² = proportion of variance explained
```

### Spearman Correlation

**Measures**: Monotonic relationship strength (non-parametric)

**Advantages**:
- ✓ No normality assumption
- ✓ Robust to outliers
- ✓ Detects non-linear monotonic relationships

**Usage Example**:
```python
from scipy import stats

# Spearman correlation
rho, p_value = stats.spearmanr(variable1, variable2)

print(f"Spearman ρ: {rho:.3f}")
print(f"p-value: {p_value:.4f}")
```

**When to Use**:
- ✓ Ordinal data
- ✓ Non-linear but monotonic relationship
- ✓ Outliers present
- ✓ Non-normal data

### Kendall's Tau

**Measures**: Ordinal association (non-parametric)

**Advantages**:
- ✓ Better for small samples
- ✓ More robust than Spearman

**Usage Example**:
```python
from scipy import stats

# Kendall's tau
tau, p_value = stats.kendalltau(variable1, variable2)

print(f"Kendall τ: {tau:.3f}")
print(f"p-value: {p_value:.4f}")
```

### Partial Correlation

**Purpose**: Correlation between X and Y, controlling for Z

```python
from scipy import stats
import numpy as np

def partial_correlation(x, y, z):
    """
    Partial correlation between x and y, controlling for z
    """
    # Correlations
    r_xy = stats.pearsonr(x, y)[0]
    r_xz = stats.pearsonr(x, z)[0]
    r_yz = stats.pearsonr(y, z)[0]
    
    # Partial correlation
    numerator = r_xy - (r_xz * r_yz)
    denominator = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
    r_xy_z = numerator / denominator
    
    # Significance test
    n = len(x)
    t_stat = r_xy_z * np.sqrt((n - 3) / (1 - r_xy_z**2))
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 3))
    
    return r_xy_z, p_value

# Example usage
r_partial, p = partial_correlation(
    peak1_intensity,
    peak2_intensity,
    concentration
)
print(f"Partial correlation: {r_partial:.3f}, p={p:.4f}")
```

### Correlation Matrix

**Purpose**: Correlations between multiple variables

```python
import numpy as np
import pandas as pd
from scipy import stats

# Create correlation matrix
data = pd.DataFrame({
    'Peak1': peak1_intensities,
    'Peak2': peak2_intensities,
    'Peak3': peak3_intensities
})

# Pearson correlation matrix
corr_matrix = data.corr(method='pearson')
print(corr_matrix)

# P-values for correlations
from scipy.stats import pearsonr

n = len(data.columns)
p_values = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        if i != j:
            _, p_values[i, j] = pearsonr(
                data.iloc[:, i],
                data.iloc[:, j]
            )

p_df = pd.DataFrame(p_values, 
                    columns=data.columns,
                    index=data.columns)
print("\nP-values:")
print(p_df)

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', 
            vmin=-1, vmax=1, center=0,
            square=True, linewidths=1)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()
```

### Troubleshooting

| Issue                                  | Cause                | Solution                                  |
| -------------------------------------- | -------------------- | ----------------------------------------- |
| Low correlation but clear relationship | Non-linear           | Use Spearman or visualize                 |
| Significant but r near 0               | Large sample size    | Check practical significance              |
| High correlation, not significant      | Small sample         | Increase n                                |
| Spurious correlation                   | Confounding variable | Use partial correlation                   |
| Correlation ≠ Causation                |                      | Design experiment or use causal inference |

---

## Multiple Testing Correction

**Problem**: Testing multiple hypotheses inflates Type I error (false positives)

**Example**:
```
Test 100 peaks at α=0.05
Expected false positives: 100 × 0.05 = 5
```

### Methods

#### 1. Bonferroni Correction

**Most Conservative**: Divide α by number of tests

**Formula**:
```
α_corrected = α / n_tests
```

**Usage**:
```python
from statsmodels.stats.multitest import multipletests

# Original p-values from multiple tests
p_values = [0.001, 0.03, 0.15, 0.008, 0.12]

# Bonferroni correction
reject, p_corrected, _, _ = multipletests(
    p_values,
    alpha=0.05,
    method='bonferroni'
)

for i, (p_orig, p_corr, is_sig) in enumerate(zip(p_values, p_corrected, reject)):
    print(f"Test {i+1}: p={p_orig:.4f}, p_corr={p_corr:.4f}, sig={is_sig}")
```

**When to Use**:
- ✓ Small number of tests (< 20)
- ✓ Want to minimize false positives
- ✗ Conservative (may miss real effects)

#### 2. Holm-Bonferroni Method

**Less Conservative**: Step-down Bonferroni

**Procedure**:
1. Sort p-values: p₁ ≤ p₂ ≤ ... ≤ pₙ
2. Test p₁ against α/n
3. If reject, test p₂ against α/(n-1)
4. Continue until failure to reject

**Usage**:
```python
reject, p_corrected, _, _ = multipletests(
    p_values,
    alpha=0.05,
    method='holm'
)
```

**Advantages**:
- ✓ More powerful than Bonferroni
- ✓ Still controls family-wise error rate

#### 3. False Discovery Rate (FDR) - Benjamini-Hochberg

**Controls**: Proportion of false positives among rejections

**Less Conservative**: Accepts some false positives for more power

**Usage**:
```python
reject, p_corrected, _, _ = multipletests(
    p_values,
    alpha=0.05,
    method='fdr_bh'  # Benjamini-Hochberg
)
```

**When to Use**:
- ✓ Large number of tests (e.g., all wavenumbers)
- ✓ Exploratory analysis
- ✓ Accept controlled false discovery rate (5%)

**Interpretation**:
```
FDR = 0.05 means:
- 5% of "significant" results may be false positives
- More lenient than Bonferroni
- Better for discovery
```

#### 4. Permutation Testing

**Most Rigorous**: Non-parametric approach

**Procedure**:
1. Compute test statistic on real data
2. Randomly permute labels many times (e.g., 10,000)
3. Compute test statistic for each permutation
4. p-value = proportion of permutations ≥ observed statistic

**Usage**:
```python
import numpy as np
from scipy import stats

def permutation_test(group1, group2, n_permutations=10000):
    """
    Two-sample permutation test
    """
    # Observed test statistic
    observed_stat = np.abs(np.mean(group1) - np.mean(group2))
    
    # Combine data
    combined = np.concatenate([group1, group2])
    n1 = len(group1)
    
    # Permutations
    perm_stats = []
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_group1 = combined[:n1]
        perm_group2 = combined[n1:]
        perm_stat = np.abs(np.mean(perm_group1) - np.mean(perm_group2))
        perm_stats.append(perm_stat)
    
    # P-value
    p_value = np.mean(np.array(perm_stats) >= observed_stat)
    
    return p_value, perm_stats

# Example
p_perm, perm_dist = permutation_test(group1, group2, n_permutations=10000)
print(f"Permutation p-value: {p_perm:.4f}")
```

**Advantages**:
- ✓ No distribution assumptions
- ✓ Exact p-values
- ✓ Flexible (any test statistic)

**Disadvantages**:
- ✗ Computationally intensive
- ✗ Requires sufficient data

### Comparison Table

| Method          | Conservativeness | Power  | Best For                       |
| --------------- | ---------------- | ------ | ------------------------------ |
| **Bonferroni**  | Very high        | Low    | Few tests, minimize FP         |
| **Holm**        | High             | Medium | Few tests, more power          |
| **FDR (B-H)**   | Medium           | High   | Many tests, discovery          |
| **Permutation** | Exact            | High   | Small datasets, no assumptions |

### Decision Guide

```
Number of tests:
│
├─ Few (< 20)
│  ├─ Minimize false positives → Bonferroni
│  └─ More power → Holm
│
├─ Many (20-1000)
│  ├─ Exploratory → FDR (Benjamini-Hochberg)
│  └─ Confirmatory → Holm
│
└─ Very many (> 1000, e.g., all wavenumbers)
   └─ FDR (Benjamini-Hochberg)
```

---

## Effect Size Measures

**Why Important**: Statistical significance ≠ practical significance

**Small p-value** with large n → tiny effect can be "significant"  
**Effect size** → How large is the difference?

### Cohen's d

**Purpose**: Standardized mean difference

**Formula**:
```
d = (mean₁ - mean₂) / pooled_SD
```

**Calculation**:
```python
import numpy as np

def cohens_d(group1, group2):
    """
    Calculate Cohen's d effect size
    """
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    
    # Cohen's d
    d = (mean1 - mean2) / pooled_std
    
    return d

# Example
d = cohens_d(group1, group2)
print(f"Cohen's d: {d:.3f}")
```

**Interpretation**:
- **|d| < 0.2**: Negligible
- **|d| = 0.2**: Small effect
- **|d| = 0.5**: Medium effect
- **|d| = 0.8**: Large effect
- **|d| > 1.2**: Very large effect

### Eta-Squared (η²)

**Purpose**: Proportion of variance explained (ANOVA)

**Formula**:
```
η² = SS_between / SS_total
```

**Calculation**:
```python
def eta_squared(groups):
    """
    Calculate eta-squared for ANOVA
    """
    # Grand mean
    grand_mean = np.mean(np.concatenate(groups))
    
    # Between-group sum of squares
    SS_between = sum([
        len(g) * (np.mean(g) - grand_mean)**2 
        for g in groups
    ])
    
    # Total sum of squares
    SS_total = sum([
        (x - grand_mean)**2 
        for g in groups 
        for x in g
    ])
    
    # Eta-squared
    eta_sq = SS_between / SS_total
    
    return eta_sq

# Example
eta_sq = eta_squared([group1, group2, group3])
print(f"η²: {eta_sq:.3f}")
```

**Interpretation**:
- **η² = 0.01**: Small effect (1% variance explained)
- **η² = 0.06**: Medium effect (6%)
- **η² = 0.14**: Large effect (14%)

### Omega-Squared (ω²)

**Purpose**: Less biased estimate of variance explained

**Preferred over η²** for smaller samples

**Calculation**:
```python
def omega_squared(groups):
    """
    Calculate omega-squared (less biased than eta-squared)
    """
    k = len(groups)  # number of groups
    N = sum([len(g) for g in groups])  # total sample size
    
    grand_mean = np.mean(np.concatenate(groups))
    
    SS_between = sum([
        len(g) * (np.mean(g) - grand_mean)**2 
        for g in groups
    ])
    
    SS_within = sum([
        sum((x - np.mean(g))**2 for x in g)
        for g in groups
    ])
    
    MS_within = SS_within / (N - k)
    
    omega_sq = (SS_between - (k - 1) * MS_within) / (SS_between + SS_within + MS_within)
    
    return max(0, omega_sq)  # Can't be negative

omega_sq = omega_squared([group1, group2, group3])
print(f"ω²: {omega_sq:.3f}")
```

### Reporting Effect Sizes

**Best Practice**: Always report effect size with p-value

**Example Report**:
```
"Group A showed significantly higher intensity than Group B 
(t(58) = 3.42, p = 0.001, d = 0.85), representing a large effect."

"ANOVA revealed significant differences among groups 
(F(2, 87) = 12.3, p < 0.001, η² = 0.22), explaining 22% of variance."
```

---

## Band Ratio Analysis

**Purpose**: Compare relative intensities of specific spectral bands

### Theory

**Rationale**:
- Dimensionality reduction (1 feature from 2 peaks)
- Normalize for concentration/thickness
- Create interpretable biomarkers

**Common Ratios**:
```
Amide I / CH₂:     I₁₆₅₅ / I₁₄₄₅
Protein / Lipid:   I₁₆₅₀ / I₁₃₀₀
Amide III ratio:   I₁₂₉₀ / I₁₂₄₀
Phosphate / Amide: I₉₆₀ / I₁₆₅₀
```

### Calculation

```python
def calculate_band_ratio(spectra, wavenumbers, 
                         band1_range, band2_range):
    """
    Calculate ratio between two spectral bands
    
    Parameters:
    -----------
    spectra : array (n_samples, n_features)
    wavenumbers : array (n_features,)
    band1_range : tuple (start, end) wavenumbers
    band2_range : tuple (start, end) wavenumbers
    
    Returns:
    --------
    ratios : array (n_samples,)
    """
    # Find indices for bands
    idx1 = np.where((wavenumbers >= band1_range[0]) & 
                    (wavenumbers <= band1_range[1]))[0]
    idx2 = np.where((wavenumbers >= band2_range[0]) & 
                    (wavenumbers <= band2_range[1]))[0]
    
    # Integrate (or max) each band
    intensity1 = np.trapz(spectra[:, idx1], wavenumbers[idx1], axis=1)
    intensity2 = np.trapz(spectra[:, idx2], wavenumbers[idx2], axis=1)
    
    # Calculate ratio
    ratios = intensity1 / (intensity2 + 1e-10)  # Avoid division by zero
    
    return ratios

# Example usage
ratios = calculate_band_ratio(
    preprocessed_spectra,
    wavenumbers,
    band1_range=(1645, 1665),  # Amide I
    band2_range=(1440, 1460)   # CH₂
)

# Statistical comparison
from scipy import stats
t_stat, p_value = stats.ttest_ind(
    ratios[labels == 'Control'],
    ratios[labels == 'Disease']
)

print(f"Band ratio test: t={t_stat:.3f}, p={p_value:.4f}")
```

### Integration Methods

**1. Trapezoidal Integration** (Recommended):
```python
intensity = np.trapz(spectrum[idx], wavenumbers[idx])
```

**2. Maximum Intensity**:
```python
intensity = np.max(spectrum[idx])
```

**3. Area Under Curve**:
```python
from sklearn.metrics import auc
intensity = auc(wavenumbers[idx], spectrum[idx])
```

### Statistical Testing

```python
# 1. Compare ratios between groups
from scipy import stats

group1_ratios = ratios[labels == 'GroupA']
group2_ratios = ratios[labels == 'GroupB']

# T-test
t_stat, p_value = stats.ttest_ind(group1_ratios, group2_ratios)

# Effect size
d = cohens_d(group1_ratios, group2_ratios)

print(f"t={t_stat:.3f}, p={p_value:.4f}, d={d:.3f}")

# 2. Correlation with clinical variable
r, p = stats.pearsonr(ratios, clinical_scores)
print(f"Correlation: r={r:.3f}, p={p:.4f}")

# 3. ROC analysis (for classification)
from sklearn.metrics import roc_auc_score, roc_curve

# Binary classification
binary_labels = (labels == 'Disease').astype(int)
auc_score = roc_auc_score(binary_labels, ratios)

fpr, tpr, thresholds = roc_curve(binary_labels, ratios)

print(f"AUC: {auc_score:.3f}")

# Find optimal threshold
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal threshold: {optimal_threshold:.3f}")
```

### Best Practices

1. **Preprocessing**:
   - Always baseline-correct before ratio calculation
   - Use same normalization for all spectra
   - Check for negative values

2. **Band Selection**:
   - Use literature-validated bands
   - Avoid overlapping bands
   - Verify peaks present in your samples

3. **Statistical Reporting**:
   ```
   "The Amide I/CH₂ ratio was significantly elevated in disease 
   samples (mean ± SD: 2.34 ± 0.45) compared to controls 
   (1.87 ± 0.32; t(58) = 4.52, p < 0.001, d = 1.17)."
   ```

---

## Best Practices

### General Workflow

```
1. Check Assumptions
   ├─ Normality (Shapiro-Wilk)
   ├─ Equal variance (Levene's)
   └─ Independence

2. Choose Appropriate Test
   ├─ Parametric (if assumptions met)
   └─ Non-parametric (if violated)

3. Apply Test
   └─ Use two-sided unless justified

4. Multiple Testing Correction
   ├─ Few tests → Bonferroni/Holm
   └─ Many tests → FDR

5. Report Effect Size
   ├─ Cohen's d (t-test)
   ├─ η² or ω² (ANOVA)
   └─ r (correlation)

6. Interpret in Context
   └─ Statistical + Practical significance
```

### Reporting Checklist

✓ **Test used** and why  
✓ **Test statistic** and degrees of freedom  
✓ **P-value** (exact or < 0.001)  
✓ **Effect size** with interpretation  
✓ **Sample sizes** per group  
✓ **Descriptive statistics** (mean ± SD)  
✓ **Multiple testing correction** method  
✓ **Assumptions** checked (state violations if any)

### Common Mistakes to Avoid

❌ **Multiple testing without correction** → Inflated Type I error  
❌ **Using parametric tests on non-normal data** → Invalid p-values  
❌ **Reporting only p-values** → Missing practical significance  
❌ **One-sided tests without justification** → Inflated significance  
❌ **Ignoring assumption violations** → Misleading results  
❌ **Cherry-picking significant results** → Publication bias

---

## See Also

- [Statistical Analysis User Guide](../user-guide/analysis.md#statistical-analysis) - Step-by-step tutorials
- [Exploratory Methods](exploratory.md) - PCA, UMAP, clustering
- [Machine Learning Methods](machine-learning.md) - Classification algorithms
- [Best Practices](../user-guide/best-practices.md) - Analysis strategies

---

**Last Updated**: 2026-01-24
