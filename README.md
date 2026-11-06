# Product Search and Retrieval System (Information Retrieval)

A standalone **Information Retrieval (IR)** project that searches product titles and ranks the most similar products.

> This repository is separate from the Machine Learning classification assignment. It does not train Decision Tree, KNN, SVM, or Naïve Bayes classifiers.

## Project objective

Given a product-title query, retrieve and rank similar product records from an e-commerce product collection.

## IR techniques

1. Text normalization (lowercase, punctuation/extra-space cleanup).
2. TF-IDF vectorization of product titles (unigrams and bigrams).
3. Cosine similarity to rank documents for a query.
4. Relevance judgement using the dataset's `Cluster ID`: another record is relevant when it belongs to the same product cluster as the query record.
5. Evaluation using Precision@5, Recall@5 and Mean Average Precision at 5 (MAP@5).

## Dataset

UCI Product Classification and Clustering (ID 837), containing 35,311 product offers across 10 product categories. The dataset is automatically fetched by `main.py` if `data/pricerunner_aggregate.csv` is not available.

Dataset page: https://archive.ics.uci.edu/dataset/837/product+classification+and+clustering

## Project structure

```text
IR_Product_Search_Retrieval_Project/
├── data/
│   └── README.md
├── results/
│   └── .gitkeep
├── .gitignore
├── OUTPUT.md
├── README.md
├── REPORT.md
├── main.py
└── requirements.txt
```

## Installation and execution

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install dependencies and run:
```bash
pip install -r requirements.txt
python main.py
```

The script creates actual result files in `results/` and updates `OUTPUT.md` using the run's measured values.

## Example interactive search

After execution, use the command-line prompt to enter a product title. The system displays the top 5 matching product records with their cosine similarity scores.

## Important academic note

Run the program yourself before submission. The `OUTPUT.md` file initially contains no fabricated performance values. Report the values generated on your own run, inspect the generated CSV/graph, and be prepared to explain the relevance definition and evaluation metrics.
