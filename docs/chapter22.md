---
hide:
  - toc
---

# Chapter 19: Pipelines and Workflows

> *"A well-designed pipeline is the backbone of reproducible and maintainable machine learning systems."*

---

## Learning Objectives

By the end of this chapter, you will be able to:

- Understand the importance of ML pipelines for reproducible workflows
- Build and use scikit-learn's `Pipeline` class for end-to-end ML workflows
- Apply `ColumnTransformer` for preprocessing different column types
- Create custom transformers and estimators for specialized preprocessing
- Implement best practices for pipeline design and debugging

---

## Intuitive Introduction

Imagine you're cooking a complex meal. You don't just throw all ingredients into one pot—you follow a systematic process: chop vegetables, marinate meat, cook components separately, then combine them. Machine learning pipelines work the same way.

Instead of manually applying preprocessing steps, training models, and making predictions in separate code blocks, pipelines chain these operations together. This ensures:

- **Reproducibility**: Same preprocessing applied to training and new data
- **Maintainability**: Changes to one step don't break others
- **Efficiency**: No risk of forgetting preprocessing steps
- **Safety**: Prevents data leakage between training and validation

Pipelines transform your ad-hoc ML code into a professional, production-ready workflow.

---

## Mathematical Development

While pipelines themselves don't introduce new mathematical concepts, they ensure mathematical transformations are applied consistently. Consider a typical ML pipeline:

1. **Feature Scaling**: Apply standardization or normalization
2. **Feature Selection**: Select k best features or remove correlated ones
3. **Model Training**: Fit the chosen algorithm
4. **Prediction**: Apply same transformations to new data

Mathematically, if we have preprocessing functions f₁, f₂, ..., fₖ and model g, the pipeline becomes:

**Training**: g(fₖ(...f₂(f₁(X_train))...)) = ŷ_train

**Prediction**: g(fₖ(...f₂(f₁(X_new))...)) = ŷ_new

This ensures identical transformations for training and inference, preventing the common mistake of applying different preprocessing to new data.

For web sources on pipeline design patterns:
- Scikit-learn Pipeline documentation: https://scikit-learn.org/stable/modules/compose.html
- ML Engineering best practices (Google, Microsoft)

---

## Implementation Guide

Scikit-learn provides powerful tools for building ML pipelines. Let's explore them systematically:

### Basic Pipeline Construction

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Create a simple pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Step 1: Feature scaling
    ('classifier', LogisticRegression(random_state=42))  # Step 2: Model
])

# Fit the pipeline (applies all steps sequentially)
pipeline.fit(X, y)

# Make predictions (applies all transformations automatically)
predictions = pipeline.predict(X)
probabilities = pipeline.predict_proba(X)

print(f"Pipeline score: {pipeline.score(X, y):.3f}")
```

**Pipeline Parameters:**	
- `steps`: List of (name, transformer/estimator) tuples
- `memory`: Cache fitted transformers (useful for large datasets)
- `verbose`: Print progress information

### Accessing Pipeline Components

```python
# Access individual steps
scaler = pipeline.named_steps['scaler']
classifier = pipeline.named_steps['classifier']

# Get feature names after transformation (if applicable)
print(f"Scaler mean: {scaler.mean_}")
print(f"Classifier coefficients shape: {classifier.coef_.shape}")

# Replace a step
pipeline.set_params(classifier__C=0.1)  # Access nested parameters
```

### ColumnTransformer for Mixed Data Types

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

# Create sample mixed data
data = pd.DataFrame({
    'age': [25, 30, np.nan, 45, 50],
    'income': [50000, 60000, 70000, 80000, 90000],
    'city': ['NYC', 'LA', 'NYC', 'Chicago', 'LA'],
    'target': [0, 1, 0, 1, 1]
})

X = data.drop('target', axis=1)
y = data['target']

# Define column groups
numeric_features = ['age', 'income']
categorical_features = ['city']

# Create preprocessing pipelines for each column type
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first', sparse_output=False))
])

# Combine with ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ]
)

# Create full pipeline
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42))
])

# Fit and predict
full_pipeline.fit(X, y)
predictions = full_pipeline.predict(X)

print(f"Pipeline score: {full_pipeline.score(X, y):.3f}")
```

**ColumnTransformer Parameters:**	
- `transformers`: List of (name, transformer, columns) tuples
- `remainder`: What to do with unspecified columns ('drop', 'passthrough', or transformer)
- `sparse_threshold`: Threshold for returning sparse matrices

### Custom Transformers

```python
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class OutlierRemover(BaseEstimator, TransformerMixin):
    """Custom transformer to remove outliers using IQR method"""
    
    def __init__(self, factor=1.5):
        self.factor = factor
        self.lower_bounds_ = None
        self.upper_bounds_ = None
    
    def fit(self, X, y=None):
        # Calculate IQR bounds for each feature
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1
        
        self.lower_bounds_ = Q1 - self.factor * IQR
        self.upper_bounds_ = Q3 + self.factor * IQR
        
        return self
    
    def transform(self, X):
        # Remove outliers
        mask = np.all((X >= self.lower_bounds_) & (X <= self.upper_bounds_), axis=1)
        return X[mask]

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer for feature engineering"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_new = X.copy()
        # Add polynomial features
        X_new['age_squared'] = X_new['age'] ** 2
        # Add interaction features
        X_new['age_income_ratio'] = X_new['age'] / X_new['income']
        return X_new

# Use custom transformers in pipeline
custom_pipeline = Pipeline([
    ('outlier_remover', OutlierRemover(factor=1.5)),
    ('feature_engineer', FeatureEngineer()),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(random_state=42))
])
```

**Custom Transformer Requirements:**	
- Inherit from `BaseEstimator` and `TransformerMixin`
- Implement `fit(X, y=None)` method
- Implement `transform(X)` method
- Return `self` from `fit`
- Handle pandas DataFrames and numpy arrays

### Pipeline with Cross-Validation

```python
from sklearn.model_selection import cross_validate, GridSearchCV

# Pipeline with hyperparameter tuning
param_grid = {
    'preprocessor__num__imputer__strategy': ['mean', 'median'],
    'classifier__C': [0.1, 1.0, 10.0]
}

grid_search = GridSearchCV(
    full_pipeline,
    param_grid,
    cv=5,
    scoring='accuracy'
)

grid_search.fit(X, y)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")
```

---

## Practical Applications

Let's build a comprehensive pipeline for the California housing dataset:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load California housing data
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target

# Add categorical feature for demonstration
X['ocean_proximity'] = np.random.choice(['<1H OCEAN', 'INLAND', 'NEAR OCEAN'], size=len(X))

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify column types
numeric_features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
categorical_features = ['ocean_proximity']

# Create preprocessing pipelines
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first', sparse_output=False))
])

# Combine preprocessors
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ]
)

# Create full pipeline
housing_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Fit pipeline
housing_pipeline.fit(X_train, y_train)

# Evaluate
train_pred = housing_pipeline.predict(X_train)
test_pred = housing_pipeline.predict(X_test)

print(f"Train RMSE: {np.sqrt(mean_squared_error(y_train, train_pred)):.3f}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, test_pred)):.3f}")
print(f"Test R²: {r2_score(y_test, test_pred):.3f}")

# Cross-validation
cv_scores = cross_validate(housing_pipeline, X_train, y_train, cv=5, scoring=['neg_mean_squared_error', 'r2'])
print(f"CV RMSE: {np.sqrt(-cv_scores['test_neg_mean_squared_error'].mean()):.3f}")
print(f"CV R²: {cv_scores['test_r2'].mean():.3f}")

# Feature importance analysis
feature_names = (numeric_features + 
                housing_pipeline.named_steps['preprocessor']
                .named_transformers_['cat']
                .named_steps['encoder']
                .get_feature_names_out(categorical_features).tolist())

importances = housing_pipeline.named_steps['regressor'].feature_importances_

# Plot feature importances
plt.figure(figsize=(12, 6))
plt.barh(feature_names, importances)
plt.xlabel('Feature Importance')
plt.title('California Housing Pipeline - Feature Importances')
plt.tight_layout()
plt.show()
```

**Key Insights from the Example:**	
- ColumnTransformer handles mixed data types seamlessly
- Pipeline ensures identical preprocessing for training and testing
- Cross-validation provides robust performance estimates
- Feature importance analysis works through the pipeline

---

## Expert Insights

### Pipeline Design Best Practices

- **Modular Design**: Each step should have a single responsibility
- **Parameter Naming**: Use descriptive names for pipeline steps
- **Error Handling**: Implement proper error handling in custom transformers
- **Memory Management**: Use `memory` parameter for large datasets
- **Version Control**: Track pipeline versions for reproducibility

### Common Pitfalls and Solutions

- **Data Leakage**: Ensure transformations fit only on training data
- **Inconsistent Preprocessing**: Always use pipelines, never manual steps
- **Debugging Difficulty**: Use `verbose=True` and intermediate predictions
- **Performance Issues**: Cache fitted transformers with `memory` parameter

### Advanced Pipeline Patterns

- **Feature Union**: Combine multiple parallel pipelines
- **Conditional Processing**: Use `FunctionTransformer` for conditional logic
- **Pipeline Persistence**: Save/load pipelines with joblib
- **Hyperparameter Tuning**: Tune entire pipeline parameters

### Performance Considerations

- **Computational Cost**: Pipelines add minimal overhead
- **Memory Usage**: Cache intermediate results when possible
- **Parallel Processing**: Use `n_jobs` in grid search
- **Scalability**: Pipelines work well with large datasets

### Integration with ML Workflow

- **Experiment Tracking**: Log pipeline parameters and results
- **Model Deployment**: Pipelines simplify model serving
- **A/B Testing**: Compare different pipeline configurations
- **Monitoring**: Track pipeline performance in production

---

## Self-Check Questions

1. Why are pipelines essential for reproducible ML workflows?
2. How does ColumnTransformer handle different data types?
3. What are the requirements for creating custom transformers?
4. How do pipelines prevent data leakage issues?

---

## Try This Exercise

**Build a Complete ML Pipeline**

1. Load a dataset with mixed data types (numerical and categorical)
2. Create a ColumnTransformer for preprocessing different column types
3. Build a Pipeline with scaling, feature selection, and a classifier
4. Implement cross-validation and hyperparameter tuning
5. Add a custom transformer for feature engineering
6. Evaluate the pipeline's performance and analyze feature importances

**Expected Outcome**: You'll have a production-ready ML pipeline that handles real-world data preprocessing challenges.

---

## Builder's Insight

Pipelines aren't just convenient—they're the foundation of professional machine learning. Without them, you're building on shifting sand.

Think of pipelines as the assembly line of machine learning: each step feeds cleanly into the next, ensuring quality and consistency. A well-designed pipeline transforms chaotic experimentation into systematic, reproducible workflows.

As you advance, you'll find that the most sophisticated ML systems often differ from simpler ones not in their algorithms, but in their pipeline design. Master pipelines, and you'll master the art of building ML systems that work reliably in the real world.

The difference between a prototype and a product often lies in the pipeline.

