# MetaFam Knowledge Graph: Final Report

**Author:** Durga Nebhrajani  
**Institution:** IIIT Hyderabad  
**Repository:** https://github.com/dnebhrajani/metafam-knowledge-graph

---

## 1. Task Completion Summary

**What was done:**
- All four tasks (exploration, community detection, rule mining, link prediction) were completed in full, with all code, results, and analysis transparently logged in Jupyter notebooks.
- All experiments are reproducible: every notebook can be re-executed from scratch, and all results are visible in the output cells.
- All code and notebooks are in the GitHub repository, with a clear README and requirements.txt for setup.

**What was not done:**
- No GNN-based link prediction (only TransE and rule-based methods were implemented, as GNNs were not required and not feasible within the time constraints).
- No temporal or weighted analysis (the dataset is static and unweighted).
- No cross-family relationship analysis (the dataset contains 50 disconnected families, so this is not possible).

---

## 2. Methodology Overview

- **Task 1:** Dataset exploration using NetworkX, pandas, numpy. Computed all standard graph metrics, 7 centrality/importance measures, generation detection, articulation points, and anomaly tests. All results are mathematically verified.
- **Task 2:** Community detection using Louvain and Label Propagation. Evaluated with modularity, NMI, ARI, coverage, conductance. Created and validated the Family Relatedness Score (FRS) metric. Honest critique: perfect scores are due to trivial disconnected structure.
- **Task 3:** Rule mining via path enumeration. Discovered 10 perfect composition rules, 4 inverse rules, 3 multi-hop rules. All rules are logged with support, confidence, and concrete examples. Failure analysis and improvement strategies included.
- **Task 4:** Link prediction with TransE (from scratch, NumPy) and rule-based baseline. Full training, evaluation, error analysis, and stress tests. Rule-enhanced (neurosymbolic) evaluation implemented. All metrics (MRR, Hits@K) are computed and compared to random baseline.

---

## 3. Key Findings & Insights

- The MetaFam dataset is synthetic, with 50 disconnected families and uniform component sizes. All metrics are computed on the whole graph, not per-family.
- Centrality and importance are multi-dimensional: only 2 nodes are top-10 in 4+ measures.
- Community detection is trivial (perfect NMI/ARI) due to disconnected structure, but Label Propagation reveals meaningful subfamilies.
- Rule mining finds 10 perfect composition rules (100% confidence), but inverse rules are only ~36% confident due to missing reciprocals.
- TransE outperforms random baseline by 75x (MRR), but fails on symmetric and sparse relations. Rule-enhanced evaluation improves logical consistency.
- All code is modular, well-documented, and reproducible. Mathematical verification is included for all key results.

---

## 4. Directory Structure

```
metafam-knowledge-graph/
├── task1_exploration.ipynb      # Task 1: Dataset Exploration
├── task2_communities.ipynb      # Task 2: Community Detection
├── task3_rule_mining.ipynb      # Task 3: Rule Mining
├── task4_link_prediction.ipynb  # Task 4: Link Prediction
├── train.txt                    # Training data (13,821 triples)
├── test.txt                     # Test data (590 triples)
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
├── final_report.md              # This report
└── TECHNICAL_REPORT.md          # Full technical details
```

---

## 5. How to Run / Reproduce

1. **Clone the repository:**
   ```bash
   git clone https://github.com/dnebhrajani/metafam-knowledge-graph.git
   cd metafam-knowledge-graph
   ```
2. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Open any notebook in Jupyter:**
   ```bash
   jupyter notebook
   # or
   jupyter lab
   ```
5. **Run all cells in each notebook to reproduce results.**

---

## 6. Libraries Used
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

## 7. Approach & Reproducibility
- All experiments are in Jupyter notebooks, with code, results, and markdown explanations in each cell.
- All outcomes are transparent, reproducible, and traceable.
- No results are hidden or post-processed outside the notebooks.
- The codebase is modular and well-commented for clarity.

---

## 8. Interview Readiness
- All code and results are in the GitHub repository.
- The README and this report explain the approach, structure, and findings.
- All notebooks are ready to be demonstrated and discussed in an interview setting.

---

## 9. References
- See TECHNICAL_REPORT.md for full references, formulas, and detailed analysis.
