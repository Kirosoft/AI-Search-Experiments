## Advanced RAG Using Hybrid Swarm Reranker

### 1. Purpose
- Investigate whether combining graph-based retrieval and swarm-style re-ranking improves recall, diversity, and accuracy for Retrieval-Augmented Generation (RAG).
- Benchmark the recently proposed GraphFlow baseline against multiple hybrid semantic-swarm retrievers on synthetic scholarly content.
- Quantify trade-offs between accuracy (MRR), diversity (DR@20), and latency while validating statistical significance of observed gains.

### 2. Experiment Setup
- **Notebook**: `Advanced_RAG_using_new_hybrid_swarm_reranker_approach.ipynb`
- **Environment**: Python 3.12.2 with dependencies auto-installed when configuring the notebook (see pip list in kernel metadata).
- **Synthetic Knowledge Base**:
	- 500 pseudo-academic documents across 10 topics.
	- Each document includes metadata (topic, year, relevance, citations, authors) and a graph structure capturing inter-document relations (citations, co-authorship, topic adjacency, hub links).
- **Query Generation**: 50 diverse queries with multiple ground-truth answers to stress-test diversity and ranking.
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` for document and query embeddings.
- **Evaluation Metrics**: Hit@k, MRR, Recall@20, De-duplicate Recall@20 (DR@20), Diversity Count, per-query latency.

### 3. Retrieval Variants Compared
1. **GraphFlow**: Flow-guided traversal on the knowledge graph (baseline).
2. **Semantic RAG**: Pure semantic similarity search over embeddings.
3. **Router Semantic RAG**: Semantic retrieval followed by router-based vote re-ranking.
4. **Enhanced SwarmRAG**: Swarm exploration with adaptive hop selection.
5. **Swarm ReRank**: Semantic recall plus swarm voting reranker on candidate pool.
6. **Leaping Semantic RAG**: Two-stage hybrid (semantic expansion + router swarm re-ranking).

### 4. Running the Notebook
- Launch Jupyter (already configured via VS Code kernel).
- Execute cells sequentially:
	1. Initialize libraries, helpers, and reproducibility seed.
	2. Build the synthetic knowledge base (graph + documents).
	3. Generate evaluation queries.
	4. Embed documents and queries with SentenceTransformer.
	5. Instantiate each retrieval strategy.
	6. Run retrieval loops to collect per-query metrics.
	7. Aggregate metrics, perform paired t-tests vs. GraphFlow baseline.
	8. Render plots and export artifacts to the repository root.
- Artifacts are emitted to the workspace root:
	- `full_experiment_results.png`
	- `complexity_analysis_all_models.png`
	- `full_experiment_results.csv` (if notebook execution saved intermediate data).

### 5. Results Summary
Average metrics (higher is better except latency), ordered by MRR:

| Method | Hit@1 | Hit@5 | Hit@10 | Hit@20 | MRR | Recall@20 | DR@20 | Diversity Count | Latency (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Leaping Semantic RAG | 0.4333 | 0.8000 | 0.9000 | 0.9000 | **0.6033** | 0.3003 | 0.3003 | 2.6000 | 0.0309 |
| Swarm ReRank | 0.3667 | **0.8333** | **0.9667** | **1.0000** | 0.5576 | **0.6792** | **0.6792** | **5.7333** | 0.0235 |
| Enhanced_SwarmRAG | 0.4000 | 0.7333 | 0.8000 | 0.8333 | 0.5339 | 0.2517 | 0.2517 | 2.3000 | 0.0216 |
| Router Semantic RAG | 0.2000 | 0.5667 | 0.8333 | 0.9333 | 0.3822 | 0.3094 | 0.3094 | 2.6667 | 0.0396 |
| Semantic RAG | 0.1667 | 0.5333 | 0.7667 | 1.0000 | 0.3615 | 0.4289 | 0.4289 | 3.5333 | **0.0110** |
| GraphFlow (Baseline) | 0.2667 | 0.3333 | 0.4667 | 0.6667 | 0.3230 | 0.1233 | 0.1233 | 1.1333 | 0.1071 |

- **Statistical Significance**: Paired t-tests vs. GraphFlow show Leaping Semantic RAG and Swarm ReRank deliver significant gains in MRR (p < 0.01) and DR@20 (p < 0.001). Diversity-oriented methods dramatically increase unique relevant hits without sacrificing latency.
- **Latency**: All hybrid methods are faster than the graph-only baseline; Semantic RAG is the fastest but underperforms on ranking.

### 6. Visual Results
![Full RAG Experiment Results Comparison](full_experiment_results.png)
*Grouped overview of Hit@k accuracy, MRR performance (with significance stars vs. GraphFlow), DR@20 diversity, unique correct-document counts, latency comparisons, and per-query MRR distributions. Highlights Leaping Semantic RAG as top in accuracy and Swarm ReRank as the diversity champion while showcasing latency advantages over GraphFlow.*

![Performance by Query Complexity (All Models)](complexity_analysis_all_models.png)
*Breakdown of diversity (DR@20) and accuracy (MRR) across simple (≤3 GT), medium (4–6 GT), and complex (>6 GT) queries. Demonstrates Swarm ReRank’s consistent superiority on diversity, Leaping Semantic RAG’s strength on medium/complex accuracy, and GraphFlow’s deficits as query difficulty increases.*

### 7. Key Conclusions
- Hybrid semantic + swarm reranking (Leaping Semantic RAG) yields the best overall accuracy, with an 86.8% MRR lift over GraphFlow while staying sub-40 ms per query.
- Swarm ReRank offers the strongest diversity improvements, achieving more than 5× unique relevant documents compared to the baseline and maintaining competitive latency.
- Pure Semantic RAG spreads recall broadly but struggles to rank, validating the need for router/swarm re-ranking layers.
- Graph-only approaches are both slower and less accurate, indicating that graph traversal alone cannot match the combined semantic + swarm strategies.

### 8. Next Steps
- Integrate real-world corpora to validate synthetic findings.
- Explore adaptive weighting between semantic expansion and swarm voting to balance diversity vs. precision dynamically.
- Extend the evaluation with generation quality metrics (e.g., answer faithfulness) to connect retrieval performance with downstream RAG outputs.
