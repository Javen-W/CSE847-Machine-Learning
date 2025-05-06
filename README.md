# CSE847: Machine Learning Coursework

This repository contains my coursework for CSE847: Machine Learning, a graduate-level course at Michigan State University. It includes three homework assignments (HW1–HW3) covering foundational machine learning concepts through theoretical proofs and practical implementations. The assignments focus on Bayesian inference, parameter estimation, Naïve Bayes classifiers, logistic regression, and model evaluation. The assignments combine mathematical derivations with Python implementations, emphasizing understanding of probabilistic models and optimization.

## Table of Contents
- [CSE847: Machine Learning Coursework](#cse847-machine-learning-coursework)
  - [Homework 1: Bayesian Inference and Classification](#homework-1-bayesian-inference-and-classification)
  - [Skills Demonstrated](#skills-demonstrated)


## Homework 1: Bayesian Inference and Classification

### Overview
Homework 1 includes five problems:
1. **Bayes Classifier**: Proving the bias of the MLE estimator for variance in a Gaussian distribution and deriving the MAP estimator for the mean.
2. **Parameter Estimation**: Computing the MLE estimator for a Poisson distribution and its expectation.
3. **Naïve Bayes Classifier**: Analyzing parameters and estimating probabilities for a categorical dataset of apples.
4. **Logistic Regression**: Investigating the effect of initial weights on convergence in gradient ascent.
5. **Gaussian Naïve Bayes and Logistic Regression**: Comparing implementations on the banknote authentication dataset, including learning curves and generative modeling.

### Problem 1: Bayes Classifier
- **Task**: Prove that the MLE estimator for variance $\hat{\sigma}^2$ in a Gaussian distribution is biased and derive the MAP estimator for the mean $\hat{\mu}_{MAP}$ given a Gaussian prior.
- **Solution**: Demonstrated that $\mathbb{E}[\hat{\sigma}^2] = \frac{N-1}{N}\sigma^2$, confirming bias. Derived $\hat{\mu}_{MAP} = \frac{N\sigma^2\theta + \lambda\sum x_i}{N\sigma^2 + \lambda}$ using the posterior distribution.

### Problem 2: Parameter Estimation
- **Task**: Find the MLE estimator for the parameter $\lambda$ of a Poisson distribution and compute the expectation $\mathbb{E}[X]$.
- **Solution**: Derived $\hat{\lambda}_{MLE} = \frac{1}{N}\sum k_i$ by maximizing the log-likelihood. Proved $\mathbb{E}[X] = \lambda$ using the Poisson PMF.

### Problem 3: Naïve Bayes Classifier
- **Task**: Analyze a dataset of apples with features (size, color, shape) to determine the number of parameters, estimate their MLE values, and predict the class probability for a new apple.
- **Solution**: Identified 14 independent parameters (class prior and conditional probabilities). Estimated probabilities via MLE (e.g., $P(\text{Size=Small}|\text{Good=Yes})$ ). Predicted $P(\text{Good=No}|\text{Small, Red, Circle}) = 0.66$, leading to a "No" prediction.

### Problem 4: Logistic Regression
- **Task**: Train a logistic regression classifier on a synthetic dataset with two initial weight vectors and analyze convergence.
- **Solution**: Implemented gradient ascent with learning rate \(\eta = 0.001\). Showed that different initial weights (\([0, 0, 0, 0]\), \([0, 0, 1, 0]\)) converge to different solutions due to the non-convex nature of the log-likelihood in this dataset. Final weights: \([0, 0.0166, 1.0089, 0.0166]\) and \([0, 0.0166, 0.0166, 0.0166]\).

### Problem 5: Gaussian Naïve Bayes and Logistic Regression
- **Task**: Implement Gaussian Naïve Bayes and logistic regression from scratch, evaluate on the banknote authentication dataset, plot learning curves, and analyze generative modeling.
- **Solution**:
  - **Implementation**: Wrote Gaussian Naïve Bayes (assuming conditional independence) and logistic regression (gradient ascent) without using ML libraries. Used 3-fold cross-validation.
  - **Learning Curves**: Plotted accuracy vs. training set size ([0.01, 0.02, 0.05, 0.1, 0.625, 1.0]) over 5 runs. Logistic regression outperformed Naïve Bayes (e.g., 0.956 vs. 0.841 accuracy at full training size).
  - **Generative Modeling**: Generated 400 samples from Naïve Bayes (class \(y=1\)) and compared mean/variance with training data. Observed slight discrepancies in variance due to Gaussian assumptions not perfectly matching the dataset’s distribution.

### Tools and Technologies
- **Python**: Core language for implementations.
- **NumPy**: Linear algebra and probability calculations.
- **Pandas**: Data manipulation and analysis.
- **Matplotlib**: Plotting learning curves.
- **ucimlrepo**: Fetching the banknote authentication dataset.
- **scikit-learn**: Used only for `KFold` and `confusion_matrix` utilities, not for model implementations.

### Results
- **Problem 4 (Logistic Regression)**:
  - Initial weights \([0, 0, 0, 0]\): Final weights \([0, 0.0166, 0.0166, 0.0166]\), log-likelihood \(-4.1098\).
  - Initial weights \([0, 0, 1, 0]\): Final weights \([0, 0.0166, 1.0089, 0.0166]\), log-likelihood \(-3.3616\).
- **Problem 5 (Model Comparison)**:
  - **Gaussian Naïve Bayes**: Mean accuracy 0.840, F1-score 0.814 (3-fold CV).
  - **Logistic Regression**: Mean accuracy 0.953, F1-score 0.945 (3-fold CV).
  - **Learning Curves**: Logistic regression consistently outperformed Naïve Bayes, with accuracy increasing with training size (e.g., 0.932 at 1% to 0.956 at 100% for LR).
  - **Generative Modeling**: Generated samples had similar means but slightly lower variances compared to training data (e.g., variance for `variance` feature: 3.33–3.89 generated vs. 3.54 training).

## Skills Demonstrated
- **Machine Learning Theory**: Derived MLE and MAP estimators, proved estimator bias, and analyzed model parameters.
- **Algorithm Implementation**: Built Gaussian Naïve Bayes and logistic regression from scratch, including gradient ascent and probability calculations.
- **Model Evaluation**: Conducted 3-fold cross-validation, computed metrics (accuracy, precision, recall, F1), and plotted learning curves.
- **Data Analysis**: Analyzed generative model outputs and compared statistical properties with training data.
- **Mathematical Rigor**: Solved problems involving Gaussian, Poisson, and Bernoulli distributions, with clear derivations.
- **Scientific Computing**: Leveraged NumPy and Pandas for efficient data processing and visualization.

