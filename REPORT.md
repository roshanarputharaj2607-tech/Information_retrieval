# Comparative Text Classification Study Report

## 1. Selected Dataset
- **Dataset**: 20 Newsgroups subset
- **Selected Classes**: sci.med, sci.space, comp.graphics, talk.politics.mideast

## 2. Methodology & Preprocessing
- Text preprocessed by removing headers, footers, and quotes.
- TF-IDF Vectorization configured with 2500 max features and English stop words filtering.
- Evaluated using stratified 80/20 train/test split.

## 3. Performance Results
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Decision Tree | 0.7622 | 0.7701 | 0.7626 | 0.7646 |
| KNN | 0.3599 | 0.4176 | 0.3579 | 0.3367 |
| SVM (Linear) | 0.8573 | 0.8599 | 0.8571 | 0.8578 |
| Naïve Bayes | 0.8740 | 0.8757 | 0.8744 | 0.8743 |

## 4. What-If Experiment
- **Experiment**: SVM Kernel change from Linear to RBF.
- **Linear SVM F1**: 0.8578
- **RBF SVM F1**: 0.8759
- **Observation**: Changing the SVM kernel from Linear to RBF changed the model's capacity to process non-linear text relationships in the high-dimensional TF-IDF space.

## 5. Summary & Recommendation
Based on actual experimental metrics, the Linear SVM / Naïve Bayes model showed competitive results. Linear SVM scales robustly with text vectors due to high-dimensionality handling properties.
