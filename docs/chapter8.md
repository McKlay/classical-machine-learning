---
hide:
  - toc
---

# Chapter 8: Naive Bayes Classifiers

## Learning Objectives

By the end of this chapter, you will be able to:
- Understand the probabilistic foundation of Naive Bayes classifiers and their assumptions
- Apply Bayes' theorem to classification problems with conditional independence
- Implement Gaussian Naive Bayes for continuous features using scikit-learn
- Compare different Naive Bayes variants and their appropriate use cases
- Handle common issues like zero probabilities through smoothing techniques
- Evaluate and tune Naive Bayes models for real-world datasets

## Intuitive Introduction

Imagine you're trying to determine if an email is spam or not. You look at words like "free," "win," or "urgent" – these are strong indicators. But you also consider the overall context: the sender, the subject line, and how these words combine. Naive Bayes classifiers work similarly, treating each feature (like a word in an email) as independently contributing to the probability of a class (spam or not spam).

The "naive" part comes from the simplifying assumption that all features are conditionally independent given the class. In reality, features might correlate – "free" and "win" often appear together in spam – but this assumption often works surprisingly well in practice, especially for text classification. It's like assuming that the presence of "free" doesn't affect the probability of "win" appearing, given that it's spam.

This approach is computationally efficient and requires less training data than many other algorithms. It's particularly effective for high-dimensional data like text, where the independence assumption is more plausible, and it serves as a strong baseline for classification tasks.

## Mathematical Development

Naive Bayes classifiers are rooted in Bayes' theorem, which provides a way to update our beliefs about the probability of a hypothesis given new evidence. For classification, we want to find the class $c$ that maximizes the posterior probability given the features $\mathbf{x} = (x_1, x_2, \dots, x_d)$.

The fundamental equation is:

\[
P(c|\mathbf{x}) = \frac{P(\mathbf{x}|c) P(c)}{P(\mathbf{x})}
\]

Since $P(\mathbf{x})$ is the same for all classes, we can focus on maximizing $P(\mathbf{x}|c) P(c)$. The "naive" assumption decomposes the likelihood:

$$P(\mathbf{x}|c) = \prod_{i=1}^d P(x_i|c)$$

This assumes conditional independence of features given the class. For continuous features, we typically assume a Gaussian distribution:

$$P(x_i|c) = \frac{1}{\sqrt{2\pi\sigma_c^2}} \exp\left(-\frac{(x_i - \mu_c)^2}{2\sigma_c^2}\right)$$

Where $\mu_c$ and $\sigma_c^2$ are the mean and variance of feature $i$ for class $c$, estimated from the training data.

For discrete features (like word counts in text), we use multinomial or Bernoulli distributions. The multinomial variant models word frequencies, while Bernoulli considers presence/absence.

To handle zero probabilities (when a feature-class combination doesn't appear in training), we apply Laplace smoothing:

$$P(x_i|c) = \frac{\text{count}(x_i, c) + \alpha}{\text{count}(c) + \alpha \cdot |V|}$$

Where $\alpha$ is the smoothing parameter and $|V|$ is the vocabulary size.

Web sources for further reading:
- [https://en.wikipedia.org/wiki/Naive_Bayes_classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)
- [https://scikit-learn.org/stable/modules/naive_bayes.html](https://scikit-learn.org/stable/modules/naive_bayes.html)

## Implementation Guide

Scikit-learn provides several Naive Bayes implementations in the `sklearn.naive_bayes` module. The most common is `GaussianNB` for continuous features, but we'll also cover `MultinomialNB` for discrete counts and `BernoulliNB` for binary features.

### GaussianNB API

```python
from sklearn.naive_bayes import GaussianNB

# Initialize the classifier
gnb = GaussianNB()

# Key parameters:
# - priors: array-like of shape (n_classes,), default=None
#   Prior probabilities of the classes. If None, priors are inferred from data.
# - var_smoothing: float, default=1e-9
#   Portion of the largest variance added to variances for numerical stability.
```

The `fit` method estimates class priors and feature means/variances:

```python
gnb.fit(X_train, y_train)
```

`predict` and `predict_proba` work as expected:

```python
y_pred = gnb.predict(X_test)
y_proba = gnb.predict_proba(X_test)
```

### Other Variants

- `MultinomialNB`: For discrete features (e.g., word counts). Key parameter: `alpha` for smoothing.
- `BernoulliNB`: For binary features. Key parameter: `alpha` for smoothing, `binarize` threshold.

All variants follow the same API pattern, making them interchangeable in pipelines.

## Practical Applications

Let's apply Gaussian Naive Bayes to the Iris dataset for species classification:

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the model
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Make predictions
y_pred = gnb.predict(X_test)

# Evaluate
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Examine learned parameters
print("Class priors:", gnb.class_prior_)
print("Feature means per class:")
for i, class_name in enumerate(iris.target_names):
    print(f"{class_name}: {gnb.theta_[i]}")
```

This code demonstrates a complete workflow: loading data, training, prediction, and evaluation. The model achieves around 97% accuracy on the Iris dataset, showing its effectiveness for this type of data.

For text classification, we can use MultinomialNB with TF-IDF features:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Sample text data
texts = ["This is a positive review", "This is negative", "Great product", "Terrible quality"]
labels = [1, 0, 1, 0]

# Create pipeline
text_clf = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB(alpha=0.1))
])

# Train and predict
text_clf.fit(texts, labels)
predictions = text_clf.predict(["Amazing service", "Poor experience"])
print(predictions)  # [1, 0]
```

## Expert Insights

Naive Bayes often outperforms more complex models on small datasets due to its simplicity and resistance to overfitting. However, the independence assumption can be violated in practice, leading to suboptimal performance when features are highly correlated.

Common pitfalls include:
- Zero probabilities: Always use smoothing (alpha > 0) to avoid division by zero
- Feature scaling: Not needed for Naive Bayes, unlike distance-based methods
- Class imbalance: The algorithm can be sensitive to uneven class distributions

For model tuning:
- Adjust `alpha` in MultinomialNB/BernoulliNB to control smoothing
- Use `priors` to incorporate domain knowledge about class probabilities
- Compare variants: Gaussian for continuous, Multinomial for counts, Bernoulli for binary

When probabilities seem unreliable, consider calibration techniques. Naive Bayes probabilities can be well-calibrated for some datasets but may need adjustment for others.

Computational complexity is linear in the number of features and classes, making it suitable for high-dimensional data. It's often used as a baseline before trying more complex models.

## Self-Check Questions

1. Why is the "naive" assumption in Naive Bayes considered "naive," and when does it still work well?
2. How does Laplace smoothing prevent zero probability issues in text classification?
3. What are the key differences between GaussianNB, MultinomialNB, and BernoulliNB?
4. When would you choose Naive Bayes over more complex classifiers like SVM or Random Forest?

## Try This Exercise

Implement a spam email classifier using MultinomialNB:

1. Use the 20 Newsgroups dataset (subset to 'sci.space' vs 'talk.politics.misc' for binary classification)
2. Create a pipeline with TfidfVectorizer and MultinomialNB
3. Experiment with different alpha values (0.01, 0.1, 1.0, 10.0)
4. Evaluate using precision, recall, and F1-score
5. Compare performance with and without stop word removal

This exercise will demonstrate Naive Bayes' effectiveness for text classification and the impact of smoothing.

## Builder's Insight

Naive Bayes represents the power of probabilistic thinking in machine learning. While its assumptions are often violated, it provides a computationally efficient and interpretable baseline that can surprise you with its performance. As you build more complex systems, always start with Naive Bayes – it might just be good enough, saving you time and computational resources. Remember, in the real world, simple models that work are often better than complex ones that don't.

