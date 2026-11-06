# Text Classification Comparative Study

This repository contains my implementation for the Text Classification comparative assignment. I evaluated four classification models: Decision Tree, K-Nearest Neighbours (KNN), Support Vector Machine (SVM), and Naïve Bayes on a subset of the 20 Newsgroups text dataset.

## Dataset and Setup
- **Source**: Scikit-Learn 20 Newsgroups
- **Target Categories**: `comp.graphics`, `sci.med`, `sci.space`, `talk.politics.mideast`
- **Text Preprocessing**: Removed headers, footers, quotes, and English stop words. Calculated TF-IDF representations (2,500 maximum features).
- **Validation Plan**: Stratified 80/20 train/test split.

## Metrics Comparison

| Algorithm | Accuracy | Precision (Macro) | Recall (Macro) | F1-score (Macro) |
| :--- | :---: | :---: | :---: | :---: |
| **Naïve Bayes (Multinomial)** | **87.40%** | **87.57%** | **87.44%** | **87.43%** |
| **Support Vector Machine (Linear)** | **85.73%** | **85.99%** | **85.71%** | **85.78%** |
| **Decision Tree** | 76.22% | 77.01% | 76.26% | 76.46% |
| **K-Nearest Neighbours (KNN)** | 35.99% | 41.76% | 35.79% | 33.67% |

## Visualizations
All generated execution charts can be found in the repository:
- `performance_comparison.png` (Accuracy, Precision, Recall, F1 comparison plot)
- `confusion_matrices.png` (Confusion Matrix heatmaps)

## How to run the project
Ensure dependencies are installed:
```bash
pip install -r requirements.txt
```
Run the execution file:
```bash
python main.py
```
