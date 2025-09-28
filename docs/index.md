---
hide:
  - toc
---

## **Classical Machine Learning**  
### *A Builder’s Guide to Mastering Traditional Algorithms with scikit-learn*

---

### **Contents**

---

#### 📖 [Preface](Preface.md)

- [Why This Book Exists](Preface.md#why-this-book-exists)

- [Who Should Read This](Preface.md#who-should-read-this)

- [From Abstraction to Understanding: How This Book Was Born](Preface.md#from-abstraction-to-understanding-how-this-book-was-born)

- [What You’ll Learn (and What You Won’t)](Preface.md#what-youll-learn-and-what-you-wont)

- [How to Read This Book](Preface.md#how-to-read-this-book)

---

#### Part I – [PART I — Foundations](PartI_overview.md)

&nbsp;&nbsp;&nbsp;&nbsp; Chapter 1: [What Is Machine Learning?](chapter1.md)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.1 Supervised vs Unsupervised Learning  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.2 Types of models (classification, regression, clustering)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.3 Typical ML pipeline  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.4 Role of `scikit-learn`

&nbsp;&nbsp;&nbsp;&nbsp; Chapter 2: [Anatomy of scikit-learn](chapter2.md)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.1 How `fit`, `predict`, `transform`, `score` work  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.2 Pipelines and cross-validation  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.3 Hyperparameters vs parameters  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.4 API consistency

---

#### Part II – [Core Algorithms (Supervised Learning)](PartII_overview.md)

&nbsp;&nbsp;&nbsp;&nbsp; Chapter 3: [Dummy Classifiers — The Baselinec](chapter3.md)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.1 Math Intuition: No math—random or majority voting.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.2 Code Walkthrough: Implement on Iris dataset; compare strategies.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.3 Parameter Explanations: Strategy options (most_frequent, stratified).  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.4 Your personal growth and career alignment.   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.4 Source Code Dissection of DummyClassifier.    

&nbsp;&nbsp;&nbsp;&nbsp; Chapter 4: [Logistic & Linear Regression](chapter4.md)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4.1 Math Intuition + Geometry: Sigmoid function, log-odds, decision boundary.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4.2 Code Walkthrough: Binary/multi-class on Wine dataset.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4.3 Parameter Explanations: C (regularization), solvers, multi_class.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4.4 Model Tuning + Diagnostics: Grid search C; check coefficients for interpretability.    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4.5 Source Code Dissection of LogisticRegression.  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4.6 Math Intuition + Geometry: Least squares, hyperplanes; Ridge/Lasso penalties.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4.7 Code Walkthrough: Predict Boston Housing prices; compare OLS vs Ridge.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4.8 Parameter Explanations: Alpha for regularization, degree for polynomial.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4.9 Model Tuning + Diagnostics: Cross-validate alpha; plot residuals.    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4.10 Source Code Dissection of LinearRegression.  

&nbsp;&nbsp;&nbsp;&nbsp; Chapter 5: [K-Nearest Neighbors (KNN)](chapter5.md)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;5.1 Math Intuition + Geometry: Distance metrics (Euclidean), voting in feature space.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;5.2 Code Walkthrough: Classify on Iris dataset with varying k.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;5.3 Parameter Explanations: n_neighbors, weights, metric.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;5.4 Model Tuning + Diagnostics: Elbow plot for k; curse of dimensionality.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;5.5 Source Code Dissection of KNeighborsClassifier.  

&nbsp;&nbsp;&nbsp;&nbsp; Chapter 6: [Decision Trees](chapter6.md)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;6.1 Math Intuition + Geometry: Entropy/ Gini, recursive splitting.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;6.2 Code Walkthrough: Build on HAR dataset; visualize tree.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;6.3 Parameter Explanations: max_depth, min_samples_split, criterion.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;6.4 Model Tuning + Diagnostics: Prune with CV; feature importance.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;6.5 Source Code Dissection of DecisionTreeClassifier.  

&nbsp;&nbsp;&nbsp;&nbsp; Chapter 7: [Support Vector Machines (SVM)](chapter7.md)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;7.1 Math Intuition + Geometry: Margins, kernels, Lagrange multipliers.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;7.2 Code Walkthrough: RBF SVM on HAR dataset with PCA.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;7.3 Parameter Explanations: C, gamma, kernel types.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;7.4 Model Tuning + Diagnostics: Grid search; plot decision boundaries.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;7.5 Deep Dive: Advanced kernel math.    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;7.6 Source Code Dissection of SVC.  

&nbsp;&nbsp;&nbsp;&nbsp; Chapter 8: [Naive Bayes Classifiers](chapter8.md)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;8.1 Math Intuition + Geometry: Bayes theorem, conditional independence.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;8.2 Code Walkthrough: Text classification on a simple dataset.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;8.3 Parameter Explanations: Alpha (smoothing), priors.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;8.4 Model Tuning + Diagnostics: Handle zero probabilities; compare variants.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;8.5 Source Code Dissection of GaussianNB.  

&nbsp;&nbsp;&nbsp;&nbsp; Chapter 9: [Random Forests and Bagging](chapter9.md)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;9.1 Math Intuition + Geometry: Bootstrap aggregating, ensemble voting.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;9.2 Code Walkthrough: Random Forest on Wine dataset.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;9.3 Parameter Explanations: n_estimators, max_features, bootstrap.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;9.4 Model Tuning + Diagnostics: OOB score; feature importance.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;9.5 Source Code Dissection of RandomForestClassifier.  

&nbsp;&nbsp;&nbsp;&nbsp; Chapter 10: [Gradient Boosting (HistGradientBoostingClassifier)](chapter10.md)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;10.1 Math Intuition + Geometry: Gradient descent on residuals, additive trees.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;10.2 Code Walkthrough: Boost on HAR dataset.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;10.3 Parameter Explanations: learning_rate, max_depth, early_stopping.    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;10.4 Model Tuning + Diagnostics: Monitor loss; avoid overfitting.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;10.5 Deep Dive: XGBoost comparison.  

---

#### Part III – [Core Algorithms (Unsupervised Learning)](PartIII_overview.md)

&nbsp;&nbsp;&nbsp;&nbsp; Chapter 11: [K-Means Clustering](chapter11.md)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;11.1 Math Intuition + Geometry: Centroids, within-cluster sum of squares.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;11.2 Code Walkthrough: Cluster Iris dataset; elbow method for k.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;11.3 Parameter Explanations: n_clusters, init, n_init.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;11.4 Model Tuning + Diagnostics: Silhouette scores; visualize clusters.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;11.5 Source Code Dissection of KMeans.  

&nbsp;&nbsp;&nbsp;&nbsp; Chapter 12: [Hierarchical Clustering](chapter12.md)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;12.1 Math Intuition + Geometry: Dendrograms, linkage methods.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;12.2 Code Walkthrough: Agglomerative clustering on Wine dataset.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;12.3 Parameter Explanations: linkage, affinity, n_clusters.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;12.4 Model Tuning + Diagnostics: Cut dendrogram; compare linkages.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;12.5 Source Code Dissection of AgglomerativeClustering.  

&nbsp;&nbsp;&nbsp;&nbsp; Chapter 13: [DBSCAN and Density-Based Clustering](chapter13.md)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;13.1 Math Intuition + Geometry: Core points, density reachability.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;13.2 Code Walkthrough: Detect clusters in noisy data.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;13.3 Parameter Explanations: eps, min_samples.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;13.4 Model Tuning + Diagnostics: Handle noise; parameter sensitivity.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;13.5 Source Code Dissection of DBSCAN.  

---

#### Part IV – [Model Evaluation & Tuning](PartIV_overview.md)

&nbsp;&nbsp;&nbsp;&nbsp; Chapter 14: [Model Evaluation Metrics](chapter14.md)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;14.1 Accuracy, precision, recall, F1  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;14.2 Confusion Matrix, ROC, PR Curves  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;14.3 When metrics disagree

&nbsp;&nbsp;&nbsp;&nbsp; Chapter 15: [Cross-Validation & StratifiedKFold](chapter15.md)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;15.1 Why we need CV  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;15.2 KFold vs Stratified  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;15.3 `cross_validate`, `GridSearchCV`, `RandomizedSearchCV`

&nbsp;&nbsp;&nbsp;&nbsp; Chapter 16: [Hyperparameter Tuning](chapter16.md)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;16.1 Grid search vs random search  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;16.2 Search space design  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;16.3 Practical examples with SVM and RF

&nbsp;&nbsp;&nbsp;&nbsp; Chapter 17: [Probability Calibration](chapter17.md)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;17.1 Why predicted probabilities can lie  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;17.2 Platt scaling (sigmoid), isotonic regression  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;17.3 `CalibratedClassifierCV` explained

&nbsp;&nbsp;&nbsp;&nbsp; Chapter 18: [Choosing Decision Thresholds](chapter18.md)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;18.1 Predicting probabilities vs predicting classes  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;18.2 Optimizing for F1, cost-sensitive thresholds  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;18.3 Manual threshold tuning with plots

---

#### Part V – [Data Engineering & Preprocessing](PartV_overview.md)

&nbsp;&nbsp;&nbsp;&nbsp; Chapter 19: [Feature Scaling and Transformation](chapter19.md)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;19.1 StandardScaler, MinMaxScaler  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;19.2 When to scale and why  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;19.3 Scaling inside pipelines

&nbsp;&nbsp;&nbsp;&nbsp; Chapter 20: [Dimensionality Reduction](chapter20.md)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;20.1 PCA: Math and scikit-learn usage  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;20.2 Using PCA with pipelines  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;20.3 Visualization  

&nbsp;&nbsp;&nbsp;&nbsp; Chapter 21: [Dealing with Imbalanced Datasets](chapter21.md)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;21.1 What is imbalance?  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;21.2 SMOTE and oversampling  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;21.3 Class weights vs resampling  

---

#### Part VI – [Advanced Topics](PartVI_overview.md)

&nbsp;&nbsp;&nbsp;&nbsp; Chapter 22: [Pipelines and Workflows](chapter22.md)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;22.1 Building maintainable ML pipelines  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;22.2 `Pipeline`, `ColumnTransformer`, custom steps  

&nbsp;&nbsp;&nbsp;&nbsp; Chapter 23: [Under the Hood of scikit-learn](chapter23.md)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;23.1 How `fit` is structured  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;23.2 Estimator base classes  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;23.3 Digging into the source 

---

#### [Appendices & Templates](appendices.md)

A. Glossary of ML terms  
B. scikit-learn cheat sheet  
C. Tips for debugging models  
D. Further reading and learning roadmap

---
