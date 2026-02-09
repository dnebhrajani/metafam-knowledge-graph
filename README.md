# MetaFam Knowledge Graph Project

This repository contains an exploratory analysis of the **MetaFam** family knowledge graph dataset for the PreCog recruitment task. The work uses graph-theoretic analysis, intra-family community detection, rule discovery, and link prediction (TransE) with a neurosymbolic evaluation layer.

---

## Directory Structure

```text
metafam-knowledge-graph/
├── task1_exploration.ipynb       # Task 1: Dataset exploration + importance measures
├── task2_communities.ipynb       # Task 2: Community detection + KINDRED relatedness metric
├── task3_rule_mining.ipynb       # Task 3: Rule mining + discovery + failure analysis
├── task4_link_prediction.ipynb   # Task 4: TransE + stress tests + rule-enhanced re-ranking
├── train.txt                     # Training triples (13,821)
├── test.txt                      # Test triples (590)
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## Dataset Snapshot (from notebooks)

- **Triples (train):** 13,821  
- **Triples (test):** 590  
- **Entities:** 1,316  
- **Relations:** 28  
- **Disconnected families (components):** 50  
- **Articulation points:** 95 (~7.2% of nodes)

---

## How to Run

```bash
git clone https://github.com/dnebhrajani/metafam-knowledge-graph.git
cd metafam-knowledge-graph

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

# Open notebooks
jupyter lab
# or
jupyter notebook
```

Then run notebooks top-to-bottom:

- `task1_exploration.ipynb`
- `task2_communities.ipynb`
- `task3_rule_mining.ipynb`
- `task4_link_prediction.ipynb`

---

## Dependencies

- pandas
- numpy
- networkx
- matplotlib
- seaborn
- community (python-louvain)
- scikit-learn
- scipy
- tqdm

---

## Approach (by task)

### Task 1: Dataset Exploration
- Load `train.txt` and build a directed multi-relational graph.
- Compute core graph metrics and structural properties.
- Define and compare multiple “importance” notions:
  - PageRank, Degree, Betweenness, Closeness
  - Vocabulary Diversity (relation-type diversity)
  - Counterfactual Importance (causal impact if removed)
  - Story-Theoretic Importance (frequency in shortest-path explanations)
- Identify **95 articulation points** (structural bridges across family branches).

### Task 2: Community Detection + Relatedness (KINDRED)
- Run Louvain / Label Propagation globally, then critique why results look “perfect”:
  - With **50 disconnected families** and effectively no inter-family edges, global community detection mostly reduces to connected-component detection (not useful for analysis).
- Apply community detection **within each connected component** to recover meaningful intra-family structure (nuclear families, lineage branches, cohorts).
- Define and implement **KINDRED**, a relatedness metric beyond hop count with components:
  - Weighted path semantics (e.g., parent=1.0, sibling=1.5, grandparent=2.0, cousin=3.0)
  - Intra-family community agreement
  - Ancestor-set overlap (Jaccard)
  - Narrative Overlap Score (NOS): co-occurrence in shortest explanation paths

### Task 3: Rule Mining
- Mine Horn-clause rules using support/confidence/coverage.
- Key findings:
  - **10 composition rules at 100% confidence** (recovering the KG’s construction axioms)
  - Structured incompleteness via inverse rules (e.g., **motherOf=60.6%**, **fatherOf=83.0%** inverse materialization)
  - Unexpected rules found by exploration, e.g.:
    - `auntOf ∘ motherOf → greatAuntOf`
    - `uncleOf ∘ fatherOf → greatUncleOf`
- Debug failed “obvious rules” by correcting directionality / variable binding (e.g., the “grandmother–sister” formulation).

### Task 4: Link Prediction (TransE) + Rule-Enhanced Evaluation
- Implement **TransE from scratch** with negative sampling and filtered evaluation.
- Stress-test predictions for biological/logic consistency (self-loop, parent uniqueness, transitivity violations).
- Apply Task 3 rules as a neurosymbolic post-processing layer:
  - Filter impossible candidates (no self-loops, no duplicate parents)
  - Re-rank remaining TransE candidates
  - Compare metrics and violation rates before/after filtering

---

## Reproducibility

- All experiments, outputs, and plots live inside the notebooks.
- Results are reproduced by running notebooks from top to bottom.
- No post-processing is done outside notebooks.

---
