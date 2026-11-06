# Model Output & What-If Verification

This document logs the runtime results and parameters verified during training and validation.

## Training Logs
```text
Loading dataset...
Training Decision Tree...
Training K-Nearest Neighbours (KNN)...
Training Support Vector Machine (SVM)...
Training Naïve Bayes...
```

## What-If Experiment Log
Changing the SVM kernel hyperparameter from `linear` to `rbf`:

```text
Running What-If Experiment (SVM with RBF Kernel vs Linear Kernel)...
Linear SVM F1: 0.8578
RBF SVM F1: 0.8759
```

### Discussion on Parameter Tuning
The non-linear RBF kernel yielded a slight improvement of ~1.8% in macro F1-score. This indicates that while text classification under TF-IDF representation is largely linearly separable, there are subtle, complex non-linear structures present in specific classes that benefit from projection to an infinite-dimensional space.
