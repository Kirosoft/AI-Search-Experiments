# Retrieval Experiment Metrics

This document explains each metric reported in `Advanced_RAG_using_new_hybrid_swarm_reranker_approach.ipynb`, how it is computed, and how to interpret it when comparing retrieval variants.

## Notation
- `Q` – set of evaluation queries.
- `k` – cutoff rank (e.g., 1, 5, 10, 20).
- `R_q` – ranked list returned for query `q`.
- `G_q` – ground-truth relevant document set for query `q`.
- `rel(q, i)` – indicator (1 or 0) that the document at rank `i` in `R_q` is relevant to `q`.
- `latency(q)` – wall-clock duration required to process query `q` (in seconds).

## Hit@k
- **Definition**: Fraction of queries for which at least one relevant document appears in the top `k` positions.
- **Formula**: \(\text{Hit@k} = \frac{1}{|Q|} \sum_{q \in Q} \mathbf{1}(\exists\, i \le k : rel(q, i) = 1)\).
- **Interpretation**: Measures the ability to surface any correct answer quickly. Higher is better.

## Mean Reciprocal Rank (MRR)
- **Definition**: Average reciprocal of the rank position of the first relevant document.
- **Formula**: \(\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{rank}_q}\), where `rank_q` is the position of the first relevant hit (undefined queries contribute 0).
- **Interpretation**: Captures early precision. Sensitive to the exact rank of the first relevant result.

## Recall@20
- **Definition**: Proportion of the ground-truth set recovered within the first 20 results.
- **Formula**: \(\text{Recall@20} = \frac{1}{|Q|} \sum_{q \in Q} \frac{|R_q^{20} \cap G_q|}{|G_q|}\), where `R_q^{20}` denotes the top 20 results.
- **Interpretation**: Indicates how completely the retriever covers relevant material at a fixed depth. Higher is better.

## De-duplicate Recall@20 (DR@20)
- **Definition**: Recall@20 computed after collapsing multiple documents that satisfy the same ground-truth target (e.g., near duplicates or documents representing the same paper).
- **Computation**: The notebook projects retrieved documents onto unique ground-truth identifiers, then applies the Recall@20 formula.
- **Interpretation**: Rewards breadth of distinct answers and penalizes redundant results.

## Diversity Count
- **Definition**: Average number of unique ground-truth targets surfaced in the top 20 results per query.
- **Formula**: \(\text{Diversity Count} = \frac{1}{|Q|} \sum_{q \in Q} |\text{unique}(R_q^{20} \cap G_q)|\).
- **Interpretation**: Highlights systems that retrieve many different correct perspectives instead of repeating similar answers.

## Latency
- **Definition**: Mean time taken to execute each retrieval pipeline, measured in seconds.
- **Formula**: \(\text{Latency} = \frac{1}{|Q|} \sum_{q \in Q} latency(q)\).
- **Interpretation**: Lower values indicate faster response. Useful for weighing accuracy gains against serving costs.

## Statistical Significance (Paired t-tests)
- **Purpose**: Evaluate whether metric improvements over the GraphFlow baseline are likely to be genuine.
- **Procedure**: For each metric (`MRR`, `DR@20`) the notebook performs paired t-tests between per-query scores of the method and GraphFlow.
- **Interpretation**:
  - `p < 0.001` (***): highly significant improvement.
  - `0.001 ≤ p < 0.01` (**): strong evidence of improvement.
  - `0.01 ≤ p < 0.05` (*): moderate evidence.
  - `p ≥ 0.05` (`ns`): difference is not statistically significant at the 5% level.

## Putting Metrics Together
- **Accuracy vs. Diversity**: MRR and Hit@k stress ranking accuracy; DR@20 and Diversity Count capture coverage of distinct relevant items.
- **Efficiency**: Latency adds the serving-cost dimension, enabling trade-off analysis.
- **Holistic Assessment**: High-performing systems balance early precision (MRR), depth (Recall@20), and variety (DR@20/Diversity Count) without incurring prohibitive latency.
