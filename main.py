"""
Product Search and Retrieval System - Information Retrieval project.
Uses TF-IDF + cosine similarity. No ML classifiers are trained.
"""

from pathlib import Path
import re
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
DATA_PATH = DATA_DIR / "pricerunner_aggregate.csv"
TOP_K = 5
MAX_EVAL_QUERIES = 300
RANDOM_STATE = 42


def normalize_text(value):
    """Normalize a product title for consistent matching."""
    value = str(value).lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def load_dataset():
    """Load local CSV or fetch UCI dataset 837 and save it locally."""
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    else:
        print("Dataset not found. Fetching UCI dataset 837...")
        try:
            from ucimlrepo import fetch_ucirepo
            dataset = fetch_ucirepo(id=837)
            parts = []
            if dataset.data.features is not None:
                parts.append(dataset.data.features.reset_index(drop=True))
            if dataset.data.targets is not None:
                parts.append(dataset.data.targets.reset_index(drop=True))
            df = pd.concat(parts, axis=1)
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            df.to_csv(DATA_PATH, index=False)
            print(f"Dataset saved to: {DATA_PATH}")
        except Exception as exc:
            raise RuntimeError(
                "Could not fetch the dataset. Check your internet connection, "
                "or download pricerunner_aggregate.csv from the UCI dataset page "
                f"and place it in {DATA_DIR}."
            ) from exc

    # Normalize column names for robust matching.
    df.columns = [str(c).strip() for c in df.columns]
    title_col = next((c for c in df.columns if c.lower() == "product title"), None)
    cluster_col = next((c for c in df.columns if c.lower() == "cluster id"), None)
    if title_col is None or cluster_col is None:
        raise ValueError(
            "Dataset must contain 'Product Title' and 'Cluster ID' columns. "
            f"Found: {list(df.columns)}"
        )

    df = df.copy()
    df["__title"] = df[title_col].fillna("").map(normalize_text)
    df["__cluster"] = df[cluster_col].astype(str)
    df = df[df["__title"].str.len() > 0].reset_index(drop=True)
    return df, title_col


def average_precision_at_k(relevant_flags, total_relevant, k):
    if total_relevant <= 0:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, is_relevant in enumerate(relevant_flags[:k], start=1):
        if is_relevant:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / min(total_relevant, k)


def evaluate_retrieval(df, matrix):
    """Evaluate a deterministic sample of queries, excluding each query itself."""
    eligible = df.groupby("__cluster").filter(lambda group: len(group) > 1).index.to_numpy()
    if len(eligible) == 0:
        raise ValueError("No clusters contain multiple records; cannot evaluate retrieval.")

    rng = np.random.default_rng(RANDOM_STATE)
    chosen = rng.choice(eligible, size=min(MAX_EVAL_QUERIES, len(eligible)), replace=False)

    rows = []
    for query_idx in chosen:
        sims = cosine_similarity(matrix[query_idx], matrix).ravel()
        sims[query_idx] = -1.0
        ranked = np.argsort(-sims)[:TOP_K]
        query_cluster = df.at[query_idx, "__cluster"]
        relevant_total = int((df["__cluster"] == query_cluster).sum() - 1)
        flags = [df.at[int(i), "__cluster"] == query_cluster for i in ranked]
        hits = int(sum(flags))
        precision = hits / TOP_K
        recall = hits / relevant_total if relevant_total else 0.0
        ap = average_precision_at_k(flags, relevant_total, TOP_K)
        rows.append({
            "query_index": int(query_idx),
            "query_title": df.at[query_idx, "__title"],
            "cluster_id": query_cluster,
            "relevant_in_top_5": hits,
            "total_relevant_in_collection": relevant_total,
            "precision_at_5": precision,
            "recall_at_5": recall,
            "average_precision_at_5": ap,
        })

    detail = pd.DataFrame(rows)
    metrics = {
        "evaluation_queries": int(len(detail)),
        "precision_at_5": float(detail["precision_at_5"].mean()),
        "recall_at_5": float(detail["recall_at_5"].mean()),
        "map_at_5": float(detail["average_precision_at_5"].mean()),
    }
    return metrics, detail


def search(query, df, vectorizer, matrix, top_k=TOP_K):
    query_clean = normalize_text(query)
    if not query_clean:
        return pd.DataFrame()
    q_vector = vectorizer.transform([query_clean])
    scores = cosine_similarity(q_vector, matrix).ravel()
    ranked = np.argsort(-scores)[:top_k]
    out = df.iloc[ranked].copy()
    out["similarity_score"] = scores[ranked]
    cols = [c for c in ["Product ID", "Product Title", "Category Label", "Cluster Label"] if c in out.columns]
    cols += ["similarity_score"]
    return out[cols].reset_index(drop=True)


def write_outputs(df, metrics, detail, title_col):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    detail.to_csv(RESULTS_DIR / "query_evaluation.csv", index=False)
    summary = pd.DataFrame([metrics])
    summary.to_csv(RESULTS_DIR / "retrieval_metrics.csv", index=False)

    plt.figure(figsize=(8, 5))
    labels = ["Precision@5", "Recall@5", "MAP@5"]
    values = [metrics["precision_at_5"], metrics["recall_at_5"], metrics["map_at_5"]]
    bars = plt.bar(labels, values)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Product Retrieval Evaluation")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, value + 0.015, f"{value:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "retrieval_metrics.png", dpi=160)
    plt.close()

    sample = detail.head(10)
    sample_md = sample[["query_title", "relevant_in_top_5", "precision_at_5", "recall_at_5", "average_precision_at_5"]].to_markdown(index=False)

    output = f"""# Product Search and Retrieval — Actual Experimental Output

## 1. Execution summary

- Dataset: `data/pricerunner_aggregate.csv` (UCI ID: 837)
- Records after removing empty titles: **{len(df):,}**
- Document field: `{title_col}`
- Relevance field: `Cluster ID` (same cluster = relevant product match)
- Representation: TF-IDF word unigrams and bigrams
- Ranking: cosine similarity
- Evaluation queries: **{metrics['evaluation_queries']}**
- Evaluation protocol: deterministic query sample, random seed {RANDOM_STATE}; query record excluded from its own results.

## 2. Retrieval performance

| Metric | Measured score |
|---|---:|
| Precision@5 | {metrics['precision_at_5']:.4f} |
| Recall@5 | {metrics['recall_at_5']:.4f} |
| MAP@5 | {metrics['map_at_5']:.4f} |

## 3. Sample query evaluation

{sample_md}

## 4. Generated files

- `results/retrieval_metrics.csv`
- `results/query_evaluation.csv`
- `results/retrieval_metrics.png`

## 5. Interpretation

Precision@5 measures the proportion of the first five results that are relevant. Recall@5 measures the proportion of all known relevant products retrieved in the first five. MAP@5 rewards relevant products appearing earlier in the ranked list.

Use these measured values in the final discussion. Results depend on the exact dataset version and software environment.
"""
    (ROOT / "OUTPUT.md").write_text(output, encoding="utf-8")
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    df, title_col = load_dataset()
    print(f"Loaded {len(df):,} product records.")

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=50000,
        sublinear_tf=True,
        norm="l2",
    )
    matrix = vectorizer.fit_transform(df["__title"])
    print(f"TF-IDF index built: {matrix.shape[0]:,} documents, {matrix.shape[1]:,} terms.")

    print("Evaluating retrieval quality...")
    metrics, detail = evaluate_retrieval(df, matrix)
    write_outputs(df, metrics, detail, title_col)

    print("\n=== RETRIEVAL EVALUATION ===")
    print(f"Precision@5: {metrics['precision_at_5']:.4f}")
    print(f"Recall@5:    {metrics['recall_at_5']:.4f}")
    print(f"MAP@5:       {metrics['map_at_5']:.4f}")
    print(f"\nResults saved in: {RESULTS_DIR}")
    print("OUTPUT.md has been updated with this run's actual results.")

    while True:
        try:
            query = input("\nEnter a product title to search (or type 'exit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting search.")
            break
        if query.lower() in {"exit", "quit", "q"}:
            break
        results = search(query, df, vectorizer, matrix)
        if results.empty:
            print("Please enter a non-empty query.")
        else:
            print("\nTop matching products:")
            print(results.to_string(index=False))


if __name__ == "__main__":
    main()
