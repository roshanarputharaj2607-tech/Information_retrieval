# Assignment Report: Text Classification Comparative Analysis

## 1. Selected Dataset
For this comparative study, I selected a subset from the 20 Newsgroups collection containing 4 distinct thematic categories: comp.graphics, sci.med, sci.space, and talk.politics.mideast. This collection provides balanced, diverse domains to test model classification capability.

## 2. Text Preprocessing & Features
I extracted structural metadata (headers, footers, quotes) to make sure models learn from original user text context. The text corpus was converted to numeric matrices using a TfidfVectorizer (configured with 2500 max features and default English stop-words filtering). Training and testing partitioning was handled via an 80/20 train-test split.

## 3. Results Comparison
- Naïve Bayes performed the best overall with an F1-score of 87.43%. Its probabilistic assumptions match sparse, high-dimensional TF-IDF matrices exceptionally well.
- Linear SVM followed closely with an 85.78% F1-score. Linear boundaries are highly effective for text data because high-dimensional inputs are often linearly separable.
- Decision Trees produced average results (76.46% F1-score) due to their binary recursive splitting nature which struggles to process rich continuous numeric matrices.
- KNN performed poorly (33.67% F1-score) because the distance computations become flat and meaningless in a 2500-dimensional vector space (the curse of dimensionality).

## 4. What-If Parameter Analysis
I configured a test changing the SVM kernel parameter from 'linear' to 'rbf':
- **Linear SVM F1-score**: 85.78%
- **RBF SVM F1-score**: 87.59%

Changing the kernel to RBF slightly boosted accuracy and F1 score, validating that mapping the TF-IDF representation into a non-linear feature space captures subtle domain-specific associations more effectively than strict linear boundaries.

## 5. Conclusion & Recommendation
I recommend using Multinomial Naïve Bayes or a tuned SVM (RBF) for this classification problem. Both options consistently maintain stable, high scores across all four evaluation metrics while scaling efficiently.
