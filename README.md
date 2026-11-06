# Comparative Study on Text Classification Classifiers

This project presents a comparative analysis of four machine learning algorithms (**Decision Tree, K-Nearest Neighbours, Support Vector Machine, and Naïve Bayes**) for a text-classification task using the classic 20 Newsgroups dataset subset.

## 📌 Task Overview
- **Dataset**: 20 Newsgroups (Subset: `comp.graphics`, `sci.med`, `sci.space`, `talk.politics.mideast`)
- **Features**: TF-IDF (Term Frequency-Inverse Document Frequency) vectorization (2,500 features)
- **Models Evaluated**: Decision Tree, KNN, SVM, Naïve Bayes
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score
- **What-If Experiment**: SVM Kernel Parameter Analysis (Linear vs. RBF)

---

## 📊 Results Summary

| Algorithm | Accuracy | Precision (Macro) | Recall (Macro) | F1-score (Macro) |
| :--- | :---: | :---: | :---: | :---: |
| **Naïve Bayes (Multinomial)** | **87.40%** | **87.57%** | **87.44%** | **87.43%** |
| **Support Vector Machine (Linear)** | **85.73%** | **85.99%** | **85.71%** | **85.78%** |
| **Decision Tree** | 76.22% | 77.01% | 76.26% | 76.46% |
| **K-Nearest Neighbours (KNN)** | 35.99% | 41.76% | 35.79% | 33.67% |

---

## 📈 Visualizations

### Performance Comparison
![Performance Comparison](performance_comparison.png)

### Confusion Matrices
![Confusion Matrices](confusion_matrices.png)

---

## 🛠️ Installation & Execution
To replicate this study, install the required packages:
```bash
pip install -r requirements.txt
```
Run the primary classification script:
```bash
python main.py
```
