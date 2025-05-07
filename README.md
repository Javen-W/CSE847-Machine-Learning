# CSE847: Machine Learning Coursework

This repository showcases my graduate-level coursework for **CSE847: Machine Learning** at Michigan State University, completed in Fall 2024. This portfolio demonstrates my expertise in machine learning theory, algorithm implementation, and model evaluation through three homework assignments (HW1–HW3). Each assignment blends mathematical derivations with Python implementations, highlighting my readiness for a **Machine Learning Engineer** role.

## Table of Contents
- [Overview](#overview)
- [Homework 1: Bayesian Inference and Classification](#homework-1-bayesian-inference-and-classification)
- [Homework 2: SVM, AdaBoost, and KNN](#homework-2-svm-adaboost-and-knn)
- [Homework 3: GMM, Graphical Models, and K-Means](#homework-3-gmm-graphical-models-and-k-means)
- [Skills Demonstrated](#skills-demonstrated)

## Overview
This repository contains three homework assignments from CSE847, covering a broad spectrum of machine learning concepts:
- **HW1**: Bayesian inference, Naïve Bayes classifiers, logistic regression, and model evaluation on the banknote authentication dataset.
- **HW2**: Support Vector Machines (SVM), AdaBoost, and K-Nearest Neighbors (KNN) for digit classification on the MNIST dataset.
- **HW3**: Gaussian Mixture Models (GMM) with EM algorithm, graphical models, and K-Means clustering on the breast cancer Wisconsin dataset.

All implementations are written from scratch in Python, using only basic libraries (e.g., NumPy, Pandas) for linear algebra and data handling, showcasing my ability to build and evaluate ML models without high-level frameworks like scikit-learn.

## Homework 1: Bayesian Inference and Classification

### Overview
HW1 explores probabilistic machine learning through five problems, combining theoretical proofs with practical implementations. I derived estimators, implemented classifiers from scratch, and evaluated models on the [banknote authentication dataset](http://archive.ics.uci.edu/ml/datasets/banknote+authentication).

### Tasks and Solutions
1. **Bayes Classifier**:
   - **Task**: Proved the MLE estimator for variance in a Gaussian distribution is biased and derived the MAP estimator for the mean with a Gaussian prior.
   - **Solution**: Demonstrated bias via expectation calculations (E[σ̂²_MLE] = ((N-1)/N)σ² ≠ σ²) and derived μ̂_MAP = (∑x_i/σ² + θ/λ) / (N/σ² + 1/λ).

2. **Parameter Estimation**:
   - **Task**: Computed the MLE estimator for a Poisson distribution’s parameter λ and its expectation.
   - **Solution**: Derived λ̂_MLE = (1/N)∑k_i (sample mean) and showed E[X] = λ using the Poisson PMF.

3. **Naïve Bayes Classifier**:
   - **Task**: Analyzed a categorical apple dataset to determine parameters, estimate MLE values, and predict class probability for a new apple (Small, Red, Circle).
   - **Solution**: Identified 7 independent parameters (e.g., P(Y=yes), P(Size=small|Y=yes)). Estimated probabilities (e.g., P(Y=yes)=0.4) and predicted P(Y=no|x)=0.182, classifying as “Yes”.

4. **Logistic Regression**:
   - **Task**: Trained a logistic regression classifier on a synthetic dataset with two initial weight vectors, analyzing convergence.
   - **Solution**: Implemented gradient ascent, achieving final weights [0, 0.0166, 0.0166, 0.0166] (log-likelihood -4.1098) and [0, 0.0166, 1.0089, 0.0166] (-3.3616). Noted convergence differences due to iteration limits.

5. **Gaussian Naïve Bayes and Logistic Regression**:
   - **Task**: Implemented both classifiers from scratch, evaluated on the banknote dataset using 3-fold cross-validation, plotted learning curves, and analyzed generative modeling.
   - **Solution**:
     - **Implementation**: Built Gaussian Naïve Bayes (assuming conditional independence) and logistic regression (gradient ascent) using NumPy. Evaluated with 3-fold CV.
     - **Learning Curves**: Plotted accuracy vs. training sizes ([0.01, 0.02, 0.05, 0.1, 0.625, 1.0]) over 5 runs. Logistic regression outperformed Naïve Bayes (e.g., 0.956 vs. 0.841 accuracy at full size).
     - **Generative Modeling**: Generated 400 samples (class y=1) using Naïve Bayes. Compared mean/variance with training data, noting slight variance discrepancies (e.g., 3.33–3.89 generated vs. 3.54 training) due to Gaussian assumptions.

### Tools and Technologies
- **Python**: Core implementation language.
- **NumPy**: Linear algebra and probability calculations.
- **Pandas**: Data manipulation.
- **Matplotlib**: Learning curve visualization.
- **ucimlrepo**: Dataset fetching.
- **scikit-learn**: Used only for `KFold` and `confusion_matrix`.

### Results
- **Logistic Regression (Problem 4)**:
  - Initial weights [0,0,0,0]: Log-likelihood -4.1098.
  - Initial weights [0,0,1,0]: Log-likelihood -3.3616.
- **Model Comparison (Problem 5)**:
  - **Gaussian Naïve Bayes**: Accuracy 0.840, F1-score 0.814 (3-fold CV).
  - **Logistic Regression**: Accuracy 0.953, F1-score 0.945 (3-fold CV).
  - **Learning Curves**: Logistic regression showed higher accuracy across all training sizes (e.g., 0.932 at 1% to 0.956 at 100%).
  - **Generative Modeling**: Generated samples closely matched training means but had lower variances, highlighting Gaussian assumption limitations.

## Homework 2: SVM, AdaBoost, and KNN

### Overview
HW2 investigates Support Vector Machines (SVM), AdaBoost, and K-Nearest Neighbors (KNN) through theoretical analysis and practical implementation. I explored SVM margins, computed AdaBoost weights, and built a KNN classifier from scratch for digit classification on a subset of the [MNIST dataset](https://archive.ics.uci.edu/ml/datasets/MNIST) (6,000 training samples, 1,000 test samples).

### Tasks and Solutions
1. **Support Vector Machines**:
   - **Task**: Compared hard-margin and soft-margin SVMs and categorized samples by slack variables.
   - **Solution**: Explained that hard-margin SVMs assume linear separability and are sensitive to outliers, while soft-margin SVMs handle noisy data via slack variables (ξ_i). Categorized samples as:
     - ξ_i = 0: Correct side of margin.
     - 0 < ξ_i < 1: Correct side of hyperplane, wrong side of margin.
     - ξ_i ≥ 1: Misclassified.
     - Noted that removing a support vector (ξ_i = 0, on margin) could shift the decision boundary.

2. **AdaBoost**:
   - **Task**: Analyzed weak classifier requirements and computed weights for a binary classification dataset over 6 iterations.
   - **Solution**:
     - **Requirements**: Weak classifiers must perform better than random guessing (>50% accuracy for binary classification) and focus on different data subsets. Classifiers with <50% accuracy degrade performance unless predictions are inverted.
     - **Weight Calculation**: Implemented AdaBoost with a threshold (θ=2.5), computing weights for 10 samples. Weights converged after 4 iterations (e.g., [0.06, ..., 0.1667]), with no improvement from iterations 5–6, indicating optimal classifier combination.

3. **K-Nearest Neighbors Classifier**:
   - **Task**: Analyzed online learning requirements and implemented a KNN classifier for MNIST digit classification, evaluating error rates for varying k.
   - **Solution**:
     - **Online Learning**: Noted that SVM requires retraining for new samples (due to margin optimization), while Naïve Bayes and KNN update incrementally. KNN has the highest inference complexity (O(n·d) for n samples, d features) due to distance calculations.
     - **Implementation**: Built KNN from scratch with Euclidean distance, using subroutines for distance calculation, neighbor selection, and majority voting. Evaluated with k=[1, 9, 19, ..., 99] over 5 runs.
     - **Error Curves**: Plotted training and test error rates vs. k. Achieved lowest test error (0.740) at k=1, with errors converging to ~0.848 at k=99. Training error was 0.0 at k=1, as expected.

### Tools and Technologies
- **Python**: Core implementation language.
- **NumPy**: Distance calculations and matrix operations.
- **Pandas**: Data analysis for results.
- **Matplotlib**: Error curve visualization.
- **Keras**: MNIST dataset loading.

### Results
- **AdaBoost**: Weights stabilized after 4 iterations (e.g., [0.06, ..., 0.1667]), confirming no benefit from additional classifiers.
- **KNN**:
  - Test error: 0.740 at k=1, increasing to 0.848 at k=99.
  - Training error: 0.0 at k=1, rising to 0.848 at k=99.
  - Error curves showed optimal performance at low k, with high errors at large k due to over-smoothing.

## Homework 3: GMM, Graphical Models, and K-Means

### Overview
HW3 focuses on unsupervised learning and probabilistic modeling through Gaussian Mixture Models (GMM), graphical models, and K-Means clustering. I implemented algorithms from scratch and applied them to a small 1D dataset and the [breast cancer Wisconsin dataset](https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original) (699 samples, 9 features).

### Tasks and Solutions
1. **Gaussian Mixture Model and EM Algorithm**:
   - **Task**: Determined the number of independent parameters in a GMM with k=2 and computed parameters after one EM iteration for a 1D dataset {-67, -48, 6, 8, 14, 16, 23, 24}.
   - **Solution**:
     - **Parameters**: Calculated 5 independent parameters (3k-1 for k=2: 2 means, 2 variances, 1 weight constrained by sum-to-1).
     - **EM Implementation**: Implemented E-step (responsibilities via Gaussian PDF) and M-step (updated means, variances, weights). After one iteration with initial means [-67, 24], variances [100, 100], weights [0.5, 0.5], obtained:
       - Weights: [0.25, 0.75].
       - Means: [-57, 15].
       - Variances: [90, 46].

2. **Graphical Models**:
   - **Task**: Derived the joint distribution for a given graphical model with variables Y, X1–X6.
   - **Solution**: Factored the joint distribution as P(Y,X1,X2,X3,X4,X5,X6) = P(Y|X2,X5)·P(X1)·P(X2)·P(X3|X2,X6)·P(X4|X3)·P(X5|X1,X6)·P(X6), reflecting the model’s conditional dependencies.

3. **K-Means Clustering**:
   - **Task**: Applied K-Means to a 2D dataset {(3,3), (7,9), (9,7), (5,3)} with initial centroids (6,5), (6,6), and implemented K-Means on the breast cancer dataset for k=2–8.
   - **Solution**:
     - **Small Dataset**: Tracked cluster memberships and centroids over iterations, converging with a loss of 6.0 (Euclidean distance sum). Noted ambiguity about squaring distances (loss 10.0 if squared).
     - **Implementation**: Built K-Means from scratch with a `Cluster` class for centroid updates, handling empty clusters by splitting the largest cluster. Used Euclidean distance and ran for k=2–8.
     - **Breast Cancer Dataset**: Plotted loss vs. k, observing diminishing returns beyond k=6. Recommended k=6 for balance between low loss and model simplicity.

### Tools and Technologies
- **Python**: Core implementation language.
- **NumPy**: Matrix operations and distance calculations.
- **Pandas**: Data handling and result analysis.
- **Matplotlib**: Loss curve visualization.
- **ucimlrepo**: Dataset fetching.

### Results
- **GMM EM**:
  - Post-iteration parameters: Weights [0.25, 0.75], Means [-57, 15], Variances [90, 46].
- **K-Means (Small Dataset)**:
  - Loss: 6.0 (or 10.0 if distances squared).
  - Final centroids: [(4,3), (8,8)].
- **K-Means (Breast Cancer)**:
  - Loss decreased with increasing k, with k=6 identified as optimal due to diminishing returns (visualized in loss curve).

## Skills Demonstrated
- **Machine Learning Theory**: Derived MLE/MAP estimators, SVM slack variables, GMM parameters, and graphical model distributions, showcasing deep understanding of probabilistic and clustering models.
- **Algorithm Implementation**: Built Gaussian Naïve Bayes, logistic regression, KNN, GMM with EM, and K-Means from scratch, mastering gradient ascent, boosting, distance-based methods, and iterative optimization.
- **Model Evaluation**: Conducted k-fold cross-validation (HW1), error rate analysis (HW2), and loss curve analysis (HW3), ensuring robust performance assessment.
- **Data Analysis**: Compared generative model outputs (HW1), error trends (HW2), and clustering loss (HW3), extracting actionable insights from statistical and visual results.
- **Mathematical Rigor**: Solved problems involving Gaussian, Poisson, Bernoulli, and mixture model distributions, with clear derivations for estimators, weights, and joint probabilities.
- **Scientific Computing**: Leveraged NumPy, Pandas, and Matplotlib for efficient data processing, visualization, and algorithm development.

