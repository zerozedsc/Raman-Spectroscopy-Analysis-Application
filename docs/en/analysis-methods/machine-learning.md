# Machine Learning Methods

Comprehensive reference for classification and regression algorithms.

## Table of Contents
- [Support Vector Machines (SVM)](#support-vector-machines-svm)
- [Random Forest](#random-forest)
- [XGBoost](#xgboost)
- [Logistic Regression](#logistic-regression)
- [Model Evaluation](#model-evaluation)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [Feature Importance](#feature-importance)

**Note**: Multi-Layer Perceptron (MLP) and neural networks are planned for future releases.

---

## Support Vector Machines (SVM)

**Purpose**: Binary or multi-class classification using optimal decision boundaries

### Theory

**Core Concept**: Find hyperplane that maximizes margin between classes

**Key Components**:
1. **Support Vectors**: Data points closest to decision boundary
2. **Margin**: Distance between hyperplane and nearest points
3. **Kernel**: Function to transform data to higher dimensions

**Decision Function**:
```
f(x) = sign(Σ αᵢ yᵢ K(xᵢ, x) + b)
```

Where:
- α: Lagrange multipliers
- y: Class labels
- K: Kernel function
- b: Bias term

### Kernel Functions

#### 1. Linear Kernel

**Formula**: `K(x, x') = xᵀx'`

**When to Use**:
- ✓ Linearly separable data
- ✓ High-dimensional data (text, spectra)
- ✓ Large datasets (fast)

**Pros**:
- Fast training and prediction
- Interpretable (feature weights)
- Less prone to overfitting

**Cons**:
- Cannot handle non-linear boundaries

#### 2. RBF (Radial Basis Function) Kernel

**Formula**: `K(x, x') = exp(-γ ||x - x'||²)`

**When to Use**:
- ✓ Non-linear boundaries
- ✓ Default choice for most problems
- ✓ Unknown data structure

**Pros**:
- Handles non-linearity
- Flexible decision boundaries

**Cons**:
- More parameters to tune (C, γ)
- Slower than linear
- Risk of overfitting

**Parameter γ (gamma)**:
- **High γ** (e.g., 0.1): Narrow influence → Complex boundaries (overfitting risk)
- **Low γ** (e.g., 0.001): Wide influence → Smooth boundaries
- **'scale'** (default): γ = 1 / (n_features × variance)
- **'auto'**: γ = 1 / n_features

#### 3. Polynomial Kernel

**Formula**: `K(x, x') = (γ xᵀx' + r)ᵈ`

**Parameters**:
- **d**: Polynomial degree (2 or 3 typical)
- **r**: Coefficient (usually 0)

**When to Use**:
- ✓ Known polynomial relationship
- ✓ Image data
- ✗ Rarely used for Raman spectra

### Hyperparameters

#### C (Regularization Parameter)

**Purpose**: Control trade-off between margin and misclassification

**Effect**:
- **High C** (e.g., 100): Smaller margin, fewer errors → Overfitting risk
- **Low C** (e.g., 0.1): Larger margin, more errors → Underfitting risk

**Typical Range**: 0.1 - 100

**Tuning Strategy**:
```python
# Grid search over C
C_range = [0.1, 1, 10, 100]
```

#### Gamma (γ) - RBF Kernel Only

**Purpose**: Define influence radius of single training example

**Effect**:
- **High γ** (e.g., 1.0): Tight fit → Overfitting
- **Low γ** (e.g., 0.001): Loose fit → Underfitting

**Typical Range**: 0.0001 - 1.0

**Tuning Strategy**:
```python
# Grid search over gamma
gamma_range = [0.0001, 0.001, 0.01, 0.1, 1.0]
```

### Usage Example

```python
from functions.ML.svm import train_svm_model

# Prepare data
X_train, X_test, y_train, y_test = train_test_split(
    preprocessed_spectra,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# Train SVM with RBF kernel
svm_model = train_svm_model(
    X_train, y_train,
    kernel='rbf',
    C=10.0,
    gamma='scale',
    random_state=42
)

# Predictions
y_pred = svm_model.predict(X_test)

# Evaluate
from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
print(classification_report(y_test, y_pred))
```

### Hyperparameter Optimization

```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# Define parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1, 'scale', 'auto'],
    'kernel': ['rbf']
}

# Grid search with cross-validation
grid_search = GridSearchCV(
    SVC(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# Best parameters
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")

# Use best model
best_svm = grid_search.best_estimator_
```

### Interpretation

**Decision Function Values**:
```python
# Get decision function scores
decision_scores = svm_model.decision_function(X_test)

# For binary classification:
# Positive → Class 1
# Negative → Class 0
# Magnitude → Confidence

# For multi-class (one-vs-one):
# Multiple decision functions (one per pair)
```

**Support Vectors**:
```python
# Number of support vectors per class
print(f"Support vectors: {svm_model.n_support_}")

# Indices of support vectors
support_indices = svm_model.support_

# Support vectors themselves
support_vectors = X_train[support_indices]
```

### Troubleshooting

| Issue              | Cause                    | Solution                         |
| ------------------ | ------------------------ | -------------------------------- |
| Poor accuracy      | Wrong kernel             | Try RBF if using linear          |
| Overfitting        | C too high or γ too high | Reduce C, reduce γ               |
| Underfitting       | C too low or γ too low   | Increase C, increase γ           |
| Very slow training | Large dataset with RBF   | Use LinearSVC or subsample       |
| Poor validation    | Data leakage             | Use GroupKFold for patient-level |

### When to Use

**Use SVM when**:
- ✓ Binary or multi-class classification
- ✓ High-dimensional data (spectra work well)
- ✓ Clear margin between classes
- ✓ Small to medium datasets (< 50,000 samples)
- ✓ Need probabilistic outputs (use probability=True)

**Consider alternatives when**:
- ✗ Very large datasets (> 100,000) → Random Forest
- ✗ Need interpretability → Logistic Regression
- ✗ Categorical features → Random Forest/XGBoost
- ✗ Need feature importance → Random Forest

### Class Imbalance

```python
# Handle imbalanced classes
svm_model = SVC(
    kernel='rbf',
    C=10,
    gamma='scale',
    class_weight='balanced',  # Automatic adjustment
    random_state=42
)

# Or specify custom weights
class_weights = {0: 1.0, 1: 3.0}  # Give 3x weight to class 1
svm_model = SVC(class_weight=class_weights)
```

### Reference
Cortes & Vapnik (1995). "Support-Vector Networks"

---

## Random Forest

**Purpose**: Ensemble of decision trees for robust classification/regression

### Theory

**Core Concept**: Combine multiple decision trees to reduce overfitting

**Bagging (Bootstrap Aggregating)**:
1. Create multiple bootstrap samples (random sampling with replacement)
2. Train decision tree on each sample
3. Aggregate predictions (vote for classification, average for regression)

**Random Feature Selection**:
- At each split, consider random subset of features
- Decorrelates trees → Better ensemble

**Out-of-Bag (OOB) Error**:
- Each tree trained on ~63% of data
- Remaining 37% used for validation (no separate test set needed)

### Hyperparameters

#### n_estimators

**Purpose**: Number of trees in forest

**Effect**:
- **More trees** → More stable, but slower
- **Fewer trees** → Faster, but less stable

**Typical Range**: 100 - 1000

**Recommendation**: Start with 100, increase if training loss still decreasing

```python
# Check OOB error vs n_trees
oob_errors = []
for n in range(10, 500, 10):
    rf = RandomForestClassifier(n_estimators=n, oob_score=True)
    rf.fit(X_train, y_train)
    oob_errors.append(1 - rf.oob_score_)

# Plot to find plateau
```

#### max_depth

**Purpose**: Maximum depth of each tree

**Effect**:
- **None** (default): Trees grow until pure leaves
- **Shallow** (e.g., 5-10): Prevents overfitting
- **Deep** (e.g., 20-30): Captures complex patterns

**Typical Range**: 10 - 30, or None

**Tuning**:
```python
# Start with None, then limit if overfitting
max_depth = None  # or 20 for regularization
```

#### min_samples_split

**Purpose**: Minimum samples required to split node

**Effect**:
- **Low** (e.g., 2): More splits → More complex trees
- **High** (e.g., 10-20): Fewer splits → Regularization

**Default**: 2

**Recommendation**: Increase to 5-10 if overfitting

#### min_samples_leaf

**Purpose**: Minimum samples in leaf node

**Effect**:
- **Low** (e.g., 1): Allows pure leaves
- **High** (e.g., 5-10): Smooths predictions

**Default**: 1

**Recommendation**: Set to 2-5 for regularization

#### max_features

**Purpose**: Number of features to consider for each split

**Options**:
- **'sqrt'** (default): √n_features (recommended for classification)
- **'log2'**: log₂(n_features)
- **int**: Specific number
- **float**: Proportion (e.g., 0.3 = 30%)

**Effect**:
- **Fewer features** → More diversity → Better ensemble
- **More features** → Individual trees stronger → Less diversity

### Usage Example

```python
from functions.ML.random_forest import train_rf_model

# Train Random Forest
rf_model = train_rf_model(
    X_train, y_train,
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1  # Use all CPU cores
)

# Predictions
y_pred = rf_model.predict(X_test)

# Prediction probabilities
y_proba = rf_model.predict_proba(X_test)

# Feature importance
importances = rf_model.feature_importances_
```

### Hyperparameter Optimization

```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

# Define parameter distributions
param_dist = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', 0.3]
}

# Random search (faster than grid search)
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=50,  # Number of parameter combinations to try
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

random_search.fit(X_train, y_train)

print(f"Best parameters: {random_search.best_params_}")
print(f"Best CV score: {random_search.best_score_:.3f}")
```

### Feature Importance

**Gini Importance** (Default):
```python
import numpy as np
import matplotlib.pyplot as plt

# Get feature importances
importances = rf_model.feature_importances_

# Sort by importance
indices = np.argsort(importances)[::-1]
top_k = 20

# Plot top features
plt.figure(figsize=(10, 6))
plt.bar(range(top_k), importances[indices[:top_k]])
plt.xlabel('Feature (Wavenumber) Index')
plt.ylabel('Importance')
plt.title('Top 20 Important Features')
plt.tight_layout()
plt.show()

# Map to wavenumbers
top_wavenumbers = wavenumbers[indices[:top_k]]
print(f"Top wavenumbers: {top_wavenumbers}")
```

**Permutation Importance** (More Accurate):
```python
from sklearn.inspection import permutation_importance

# Calculate permutation importance
perm_importance = permutation_importance(
    rf_model,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

# Get importances and standard deviations
perm_importances_mean = perm_importance.importances_mean
perm_importances_std = perm_importance.importances_std

# Sort
indices = np.argsort(perm_importances_mean)[::-1]
top_k = 20

# Plot
plt.figure(figsize=(10, 6))
plt.bar(range(top_k), 
        perm_importances_mean[indices[:top_k]],
        yerr=perm_importances_std[indices[:top_k]])
plt.xlabel('Feature Index')
plt.ylabel('Permutation Importance')
plt.title('Top 20 Features (Permutation Importance)')
plt.tight_layout()
plt.show()
```

(shap-values)=
#### SHAP Values

**Purpose**: Explain model predictions via Shapley-value-based feature attribution.

**When to use**:
- ✓ Per-sample explanations (which wavenumbers drive a prediction)
- ✓ Global importance aggregated from local attributions

**Note**: This typically uses the external `shap` library.

### Out-of-Bag (OOB) Score

```python
# Train with OOB scoring
rf_model = RandomForestClassifier(
    n_estimators=100,
    oob_score=True,
    random_state=42
)
rf_model.fit(X_train, y_train)

# OOB score (similar to cross-validation)
print(f"OOB Score: {rf_model.oob_score_:.3f}")

# OOB predictions
oob_pred = rf_model.oob_decision_function_
```

### Advantages

✓ **Robust**: Less prone to overfitting than single tree  
✓ **Feature importance**: Automatic ranking  
✓ **Handles non-linearity**: No kernel needed  
✓ **Missing values**: Can handle with imputation  
✓ **No scaling required**: Works with raw features  
✓ **Parallelizable**: Fast training with multiple cores

### Limitations

✗ **Black box**: Hard to interpret individual predictions  
✗ **Large models**: Memory intensive for many trees  
✗ **Extrapolation**: Poor performance outside training range  
✗ **Biased**: Toward features with many categories

### When to Use

**Use Random Forest when**:
- ✓ Tabular data with mixed features
- ✓ Need feature importance
- ✓ Don't want to tune many hyperparameters
- ✓ Want robust, reliable performance
- ✓ Have sufficient data (> 1000 samples)

**Consider alternatives when**:
- ✗ Need probabilistic model → Logistic Regression
- ✗ Have sequential/spatial data → Neural networks
- ✗ Need speed → Logistic Regression or LinearSVC
- ✗ Want best accuracy → XGBoost (usually better)

### Reference
Breiman (2001). "Random Forests"

---

## XGBoost

**Purpose**: Gradient boosting for high-performance classification/regression

### Theory

**Core Concept**: Build trees sequentially, each correcting previous errors

**Gradient Boosting**:
1. Start with simple model (e.g., mean prediction)
2. Calculate residuals (errors)
3. Train new tree to predict residuals
4. Add to ensemble with small weight
5. Repeat until convergence

**XGBoost Innovations**:
- Regularization (L1/L2) to prevent overfitting
- Handling missing values automatically
- Parallel tree construction (fast)
- Built-in cross-validation

**Formula**:
```
F(x) = Σ fₖ(x)
where fₖ is k-th tree
```

### Hyperparameters

#### n_estimators

**Purpose**: Number of boosting rounds (trees)

**Effect**:
- **More trees** → Better training fit → Overfitting risk
- **Fewer trees** → Underfitting

**Typical Range**: 100 - 1000

**Best Practice**: Use early stopping to find optimal number

```python
# Early stopping
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,  # Stop if no improvement for 50 rounds
    verbose=False
)
```

#### learning_rate (eta)

**Purpose**: Step size for each tree's contribution

**Effect**:
- **High** (e.g., 0.3): Fast convergence → Overfitting risk
- **Low** (e.g., 0.01): Slow convergence → More trees needed → Better generalization

**Typical Range**: 0.01 - 0.3

**Rule of Thumb**: Lower learning_rate requires more n_estimators

```python
# Conservative approach
learning_rate = 0.1
n_estimators = 500

# Aggressive approach (faster, riskier)
learning_rate = 0.3
n_estimators = 100
```

#### max_depth

**Purpose**: Maximum depth of each tree

**Effect**:
- **Deep** (e.g., 6-10): Captures complex interactions
- **Shallow** (e.g., 3-5): Prevents overfitting

**Typical Range**: 3 - 10

**Default**: 6

#### subsample

**Purpose**: Fraction of samples used for each tree

**Effect**:
- **< 1.0** (e.g., 0.8): Reduces overfitting, speeds up training
- **= 1.0**: Use all samples

**Typical Range**: 0.5 - 1.0

**Recommendation**: 0.8 for robustness

#### colsample_bytree

**Purpose**: Fraction of features used for each tree

**Effect**:
- **< 1.0** (e.g., 0.8): Reduces overfitting, adds diversity
- **= 1.0**: Use all features

**Typical Range**: 0.5 - 1.0

**Recommendation**: 0.8

#### gamma (min_split_loss)

**Purpose**: Minimum loss reduction to make split

**Effect**:
- **Higher**: More conservative splitting → Regularization
- **Lower**: More splits → Risk of overfitting

**Typical Range**: 0 - 5

**Default**: 0

#### lambda (reg_lambda) - L2 Regularization

**Purpose**: L2 penalty on leaf weights

**Effect**:
- **Higher**: Stronger regularization
- **Lower**: Less regularization

**Typical Range**: 0 - 10

**Default**: 1

#### alpha (reg_alpha) - L1 Regularization

**Purpose**: L1 penalty on leaf weights (feature selection)

**Effect**:
- **Higher**: More sparsity (some weights → 0)
- **Lower**: Less sparsity

**Typical Range**: 0 - 10

**Default**: 0

### Usage Example

```python
from functions.ML.xgboost import train_xgboost_model

# Train XGBoost
xgb_model = train_xgboost_model(
    X_train, y_train,
    n_estimators=500,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0,
    reg_lambda=1,
    reg_alpha=0,
    random_state=42,
    n_jobs=-1
)

# Predictions
y_pred = xgb_model.predict(X_test)
y_proba = xgb_model.predict_proba(X_test)
```

### Hyperparameter Optimization

```python
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

# Define parameter distributions
param_dist = {
    'n_estimators': [100, 200, 500, 1000],
    'learning_rate': [0.01, 0.05, 0.1, 0.3],
    'max_depth': [3, 5, 7, 9],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 1, 5],
    'reg_lambda': [0, 1, 10],
    'reg_alpha': [0, 0.1, 1]
}

# Random search
random_search = RandomizedSearchCV(
    XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

random_search.fit(X_train, y_train)

print(f"Best parameters: {random_search.best_params_}")
print(f"Best CV score: {random_search.best_score_:.3f}")
```

### Feature Importance

```python
import matplotlib.pyplot as plt

# Get feature importances
importances = xgb_model.feature_importances_

# Plot top features
import numpy as np
indices = np.argsort(importances)[::-1]
top_k = 20

plt.figure(figsize=(10, 6))
plt.bar(range(top_k), importances[indices[:top_k]])
plt.xlabel('Feature Index')
plt.ylabel('Importance (Gain)')
plt.title('Top 20 Important Features (XGBoost)')
plt.tight_layout()
plt.show()

# Importance types in XGBoost:
# - 'weight': Number of times feature used
# - 'gain': Average gain (default, most useful)
# - 'cover': Average coverage

# Get different importance types
import xgboost as xgb
importance_gain = xgb_model.get_booster().get_score(importance_type='gain')
importance_weight = xgb_model.get_booster().get_score(importance_type='weight')
```

### Learning Curves

```python
# Train with evaluation set to monitor performance
eval_set = [(X_train, y_train), (X_val, y_val)]

xgb_model.fit(
    X_train, y_train,
    eval_set=eval_set,
    eval_metric='logloss',
    verbose=50  # Print every 50 rounds
)

# Get evaluation results
results = xgb_model.evals_result()

# Plot learning curves
plt.figure(figsize=(10, 6))
plt.plot(results['validation_0']['logloss'], label='Train')
plt.plot(results['validation_1']['logloss'], label='Validation')
plt.xlabel('Boosting Round')
plt.ylabel('Log Loss')
plt.legend()
plt.title('XGBoost Learning Curves')
plt.tight_layout()
plt.show()
```

### Advantages

✓ **State-of-the-art accuracy**: Often wins competitions  
✓ **Regularization**: Built-in L1/L2  
✓ **Handles missing data**: Automatically  
✓ **Fast**: Parallel tree construction  
✓ **Early stopping**: Prevents overfitting  
✓ **Feature importance**: Gain, weight, cover

### Limitations

✗ **Many hyperparameters**: Requires tuning  
✗ **Sensitive to overfitting**: With default params  
✗ **Black box**: Hard to interpret  
✗ **Memory intensive**: For large datasets

### When to Use

**Use XGBoost when**:
- ✓ Need best possible accuracy
- ✓ Tabular data
- ✓ Have time to tune hyperparameters
- ✓ Competition or critical application
- ✓ Medium to large datasets

**Consider alternatives when**:
- ✗ Need interpretability → Logistic Regression
- ✗ Limited time → Random Forest (fewer params)
- ✗ Very small data → Logistic Regression or SVM

### Reference
Chen & Guestrin (2016). "XGBoost: A Scalable Tree Boosting System"

---

## Logistic Regression

**Purpose**: Linear model for probabilistic binary/multi-class classification

### Theory

**Core Concept**: Model log-odds as linear combination of features

**Binary Classification**:
```
P(y=1|x) = 1 / (1 + exp(-(β₀ + β₁x₁ + ... + βₙxₙ)))
```

**Multi-class** (one-vs-rest or multinomial):
```
P(y=k|x) = exp(xᵀβₖ) / Σⱼ exp(xᵀβⱼ)
```

**Decision Boundary**: Linear in feature space

### Hyperparameters

#### C (Inverse Regularization)

**Purpose**: Control strength of regularization

**Effect**:
- **High C** (e.g., 100): Weak regularization → Overfitting risk
- **Low C** (e.g., 0.01): Strong regularization → Underfitting risk

**Typical Range**: 0.01 - 100

**Note**: C is inverse of regularization strength (unlike most methods)

#### penalty

**Purpose**: Type of regularization

**Options**:
- **'l2'** (default): Ridge regularization (shrinks coefficients)
- **'l1'**: Lasso regularization (sparse coefficients, feature selection)
- **'elasticnet'**: Mix of L1 and L2
- **'none'**: No regularization

**When to Use**:
- **L2**: Default, works well generally
- **L1**: When want feature selection (many irrelevant features)
- **Elasticnet**: When L1 too aggressive

#### solver

**Purpose**: Optimization algorithm

**Options**:
- **'lbfgs'** (default): Good for small datasets, L2 only
- **'saga'**: Supports all penalties, good for large datasets
- **'liblinear'**: Good for small datasets, supports L1/L2

**Recommendation**: Use 'saga' for flexibility

#### max_iter

**Purpose**: Maximum iterations for convergence

**Default**: 100

**Increase if**: Warning about non-convergence

**Typical Range**: 100 - 10000

### Usage Example

```python
from functions.ML.logistic_regression import train_lr_model

# Train Logistic Regression
lr_model = train_lr_model(
    X_train, y_train,
    C=1.0,
    penalty='l2',
    solver='saga',
    max_iter=1000,
    random_state=42
)

# Predictions
y_pred = lr_model.predict(X_test)
y_proba = lr_model.predict_proba(X_test)

# Get coefficients (feature weights)
coefficients = lr_model.coef_[0]  # For binary classification

# Intercept
intercept = lr_model.intercept_
```

### Hyperparameter Optimization

```python
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

# Define parameter grid
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['saga'],
    'max_iter': [1000]
}

# Grid search
grid_search = GridSearchCV(
    LogisticRegression(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")
```

### Interpretation

**Coefficients** (Feature Weights):
```python
import numpy as np
import matplotlib.pyplot as plt

# Get coefficients
coefs = lr_model.coef_[0]

# Find most important features
indices = np.argsort(np.abs(coefs))[::-1]
top_k = 20

# Plot
plt.figure(figsize=(10, 6))
plt.bar(range(top_k), coefs[indices[:top_k]])
plt.xlabel('Feature Index')
plt.ylabel('Coefficient')
plt.title('Top 20 Feature Coefficients')
plt.axhline(y=0, color='k', linestyle='--')
plt.tight_layout()
plt.show()

# Positive coefficient → Increases probability of class 1
# Negative coefficient → Decreases probability of class 1
```

**Odds Ratios**:
```python
# Convert coefficients to odds ratios
odds_ratios = np.exp(coefs)

# Interpretation:
# OR = 2.0 → One unit increase in feature doubles odds
# OR = 0.5 → One unit increase halves odds
# OR = 1.0 → No effect

print(f"Odds ratios for top features:")
for i in indices[:10]:
    print(f"  Feature {i}: OR = {odds_ratios[i]:.3f}")
```

**Probability Calibration**:
```python
# Check calibration
from sklearn.calibration import calibration_curve

# Predicted probabilities
y_proba = lr_model.predict_proba(X_test)[:, 1]

# Calibration curve
fraction_of_positives, mean_predicted_value = calibration_curve(
    y_test, y_proba, n_bins=10
)

# Plot
plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
plt.plot(mean_predicted_value, fraction_of_positives, 'o-', label='Logistic Regression')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve')
plt.legend()
plt.tight_layout()
plt.show()
```

### Advantages

✓ **Interpretable**: Clear feature weights  
✓ **Probabilistic**: Direct probability estimates  
✓ **Fast**: Training and prediction  
✓ **Well-calibrated**: Reliable probabilities  
✓ **No hyperparameters**: (except C)  
✓ **Linear decision boundary**: Simple

### Limitations

✗ **Linear only**: Cannot capture non-linear relationships  
✗ **Feature engineering**: May need manual feature creation  
✗ **Sensitive to scaling**: Requires standardization  
✗ **Multicollinearity**: Correlated features problematic

### When to Use

**Use Logistic Regression when**:
- ✓ Need interpretability (coefficients)
- ✓ Want probabilistic outputs
- ✓ Linearly separable data
- ✓ Baseline model
- ✓ Small datasets
- ✓ Need fast predictions

**Consider alternatives when**:
- ✗ Non-linear boundaries → SVM (RBF) or Random Forest
- ✗ Many irrelevant features → Random Forest (feature importance)
- ✗ Complex interactions → XGBoost or Neural Networks

### Reference
Cox (1958). "The Regression Analysis of Binary Sequences"

---

## Multi-Layer Perceptron (MLP)

**Purpose**: Neural network for non-linear classification/regression

### Theory

**Architecture**: Input → Hidden Layers → Output

**Neuron Computation**:
```
output = activation(Σ wᵢxᵢ + b)
```

**Activation Functions**:
- **ReLU**: f(x) = max(0, x) [Default, recommended]
- **Tanh**: f(x) = tanh(x)
- **Logistic**: f(x) = 1/(1 + e⁻ˣ)

**Training**: Backpropagation with gradient descent

### Hyperparameters

#### hidden_layer_sizes

**Purpose**: Architecture of hidden layers

**Format**: Tuple (layer1_size, layer2_size, ...)

**Examples**:
```python
# Single hidden layer with 100 neurons
hidden_layer_sizes = (100,)

# Two hidden layers (100, 50)
hidden_layer_sizes = (100, 50)

# Three hidden layers (200, 100, 50)
hidden_layer_sizes = (200, 100, 50)
```

**Rules of Thumb**:
- Start with (100,) or (100, 50)
- More neurons → More capacity → Overfitting risk
- More layers → Can learn complex patterns

#### activation

**Purpose**: Non-linear activation function

**Options**:
- **'relu'** (default): Recommended, works well
- **'tanh'**: Alternative, slower convergence
- **'logistic'**: Sigmoid, rarely used

**Recommendation**: Use 'relu'

#### alpha

**Purpose**: L2 regularization strength

**Effect**:
- **Higher** (e.g., 0.01): Strong regularization
- **Lower** (e.g., 0.0001): Weak regularization

**Typical Range**: 0.0001 - 0.01

**Default**: 0.0001

#### learning_rate_init

**Purpose**: Initial learning rate

**Effect**:
- **Higher** (e.g., 0.01): Faster convergence → Instability risk
- **Lower** (e.g., 0.0001): Slower convergence → More stable

**Typical Range**: 0.0001 - 0.01

**Default**: 0.001

#### max_iter

**Purpose**: Maximum epochs (training iterations)

**Default**: 200

**Typical Range**: 200 - 1000

**Increase if**: Training loss still decreasing

#### early_stopping

**Purpose**: Stop training when validation score stops improving

**Recommended**: True

**Parameters**:
- **validation_fraction**: Fraction for validation (default 0.1)
- **n_iter_no_change**: Patience (default 10)

### Usage Example

```python
from functions.ML.mlp import train_mlp_model

# Train MLP
mlp_model = train_mlp_model(
    X_train, y_train,
    hidden_layer_sizes=(100, 50),
    activation='relu',
    alpha=0.0001,
    learning_rate_init=0.001,
    max_iter=500,
    early_stopping=True,
    random_state=42
)

# Predictions
y_pred = mlp_model.predict(X_test)
y_proba = mlp_model.predict_proba(X_test)
```

### Hyperparameter Optimization

```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier

# Define parameter distributions
param_dist = {
    'hidden_layer_sizes': [(50,), (100,), (100, 50), (200, 100)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate_init': [0.0001, 0.001, 0.01],
    'max_iter': [500]
}

# Random search
random_search = RandomizedSearchCV(
    MLPClassifier(early_stopping=True, random_state=42),
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)

print(f"Best parameters: {random_search.best_params_}")
print(f"Best CV score: {random_search.best_score_:.3f}")
```

### Learning Curves

```python
# Access loss history
train_loss = mlp_model.loss_curve_

# Plot
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(train_loss)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('MLP Training Loss')
plt.tight_layout()
plt.show()

# Check for convergence:
# - Should decrease steadily
# - Flatten at end
# - If still decreasing → increase max_iter
```

### Advantages

✓ **Non-linear**: Can learn complex patterns  
✓ **Flexible**: Arbitrary architectures  
✓ **Universal approximator**: Theoretically can learn any function  
✓ **Feature learning**: Automatic feature extraction

### Limitations

✗ **Black box**: Hard to interpret  
✗ **Hyperparameters**: Many to tune  
✗ **Convergence**: Can be slow or unstable  
✗ **Scaling required**: Sensitive to feature scales  
✗ **Random initialization**: Different results per run

### When to Use

**Use MLP when**:
- ✓ Non-linear, complex patterns
- ✓ Large datasets (> 10,000 samples)
- ✓ Don't need interpretability
- ✓ Have time to tune
- ✓ Sufficient data to prevent overfitting

**Consider alternatives when**:
- ✗ Need interpretability → Logistic Regression
- ✗ Small data → SVM or Random Forest
- ✗ Want speed → Random Forest
- ✗ Tabular data → XGBoost (usually better)

### Reference
Rumelhart et al. (1986). "Learning representations by back-propagating errors"

---

## Model Evaluation

### Classification Metrics

#### Accuracy

**Formula**: (TP + TN) / (TP + TN + FP + FN)

**When to Use**: Balanced classes

**Limitation**: Misleading for imbalanced data

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
```

#### Precision

**Formula**: TP / (TP + FP)

**Interpretation**: Of predicted positives, how many are correct?

**When to Use**: Cost of false positives high (e.g., spam detection)

```python
from sklearn.metrics import precision_score

precision = precision_score(y_test, y_pred, average='weighted')
print(f"Precision: {precision:.3f}")
```

#### Recall (Sensitivity)

**Formula**: TP / (TP + FN)

**Interpretation**: Of actual positives, how many detected?

**When to Use**: Cost of false negatives high (e.g., disease screening)

```python
from sklearn.metrics import recall_score

recall = recall_score(y_test, y_pred, average='weighted')
print(f"Recall: {recall:.3f}")
```

#### F1-Score

**Formula**: 2 × (Precision × Recall) / (Precision + Recall)

**Interpretation**: Harmonic mean of precision and recall

**When to Use**: Balance precision and recall

```python
from sklearn.metrics import f1_score

f1 = f1_score(y_test, y_pred, average='weighted')
print(f"F1-Score: {f1:.3f}")
```

#### ROC-AUC

**Purpose**: Measure discrimination ability across all thresholds

**Range**: 0.5 (random) to 1.0 (perfect)

**When to Use**: Imbalanced classes, need threshold-independent metric

```python
from sklearn.metrics import roc_auc_score, roc_curve

# Binary classification
y_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC: {auc:.3f}")

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.tight_layout()
plt.show()
```

#### Confusion Matrix

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()
```

#### Classification Report

```python
from sklearn.metrics import classification_report

report = classification_report(y_test, y_pred, target_names=class_names)
print(report)

# Example output:
#               precision    recall  f1-score   support
#
#      Class A       0.92      0.95      0.93        20
#      Class B       0.88      0.85      0.87        20
#
#     accuracy                           0.90        40
#    macro avg       0.90      0.90      0.90        40
# weighted avg       0.90      0.90      0.90        40
```

### Cross-Validation

#### K-Fold Cross-Validation

```python
from sklearn.model_selection import cross_val_score

# 5-fold CV
scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring='accuracy'
)

print(f"CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
```

#### Stratified K-Fold

**Use when**: Imbalanced classes

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=skf,
    scoring='accuracy'
)

print(f"Stratified CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
```

(groupkfold)=
#### Group K-Fold (Patient-Level)

**Critical for Raman data**: Prevent data leakage from same patient

```python
from sklearn.model_selection import GroupKFold, cross_val_score

# groups: Patient IDs for each sample
gkf = GroupKFold(n_splits=5)

scores = cross_val_score(
    model,
    X_train,
    y_train,
    groups=patient_ids,  # Ensure all samples from same patient in same fold
    cv=gkf,
    scoring='accuracy'
)

print(f"Group CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
```

---

## See Also

- [Machine Learning User Guide](../user-guide/machine-learning.md) - Step-by-step tutorials
- [Best Practices](../user-guide/best-practices.md) - ML strategies
- [Preprocessing Methods](preprocessing.md) - Data preparation
- [Statistical Methods](statistical.md) - Hypothesis testing

---

**Last Updated**: 2026-01-24
