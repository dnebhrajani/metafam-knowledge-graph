# MetaFam Knowledge Graph Project

## Project Overview

This repository contains a comprehensive exploratory analysis of the MetaFam family knowledge graph dataset as part of the Precog recruitment task. The analysis uses graph theory techniques to uncover patterns in family structures, identify important family members, and understand hierarchical relationships.

---

## Directory Structure

```
metafam-knowledge-graph/
│
├── task1_exploration.ipynb       # Task 1: Dataset exploration
├── task2_communities.ipynb       # Task 2: Community detection
├── task3_rule_mining.ipynb       # Task 3: Rule mining
├── task4_link_prediction.ipynb   # Task 4: Link prediction
├── train.txt                     # Training dataset (13,821 relationships)
├── test.txt                      # Test dataset (590 relationships)
├── README.md                     # This file
├── requirements.txt              # Python dependencies
└── REPORT.md                     # Project report
```

---

## How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/dnebhrajani/metafam-knowledge-graph.git
   cd metafam-knowledge-graph
   ```
2. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Open the notebooks in Jupyter:**
   ```bash
   jupyter notebook
   # or
   jupyter lab
   ```
5. **Run all cells in each notebook to reproduce results.**

---

## Dependency Libraries
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

## Transparency
- All experiments and results are in Jupyter notebooks, with code, outputs, and markdown explanations in each cell.
- Each task (exploration, community detection, rule mining, link prediction) is a separate notebook.
- All outcomes are transparent, reproducible, and traceable within the notebooks.
- No results are hidden or post-processed outside the notebooks.

---

## Project Highlights
- **Task 1:** Full dataset exploration, graph metrics, 7 centrality/importance measures, generation detection, articulation points, anomaly tests, and mathematical verification.
- **Task 2:** Community detection (Louvain, Label Propagation), evaluation metrics, FRS metric, honest critique of trivial structure, and subfamily analysis.
- **Task 3:** Rule mining (10 perfect composition rules, 4 inverse, 3 multi-hop), support/confidence, concrete examples, failure analysis, and improvement strategies.
- **Task 4:** Link prediction (TransE from scratch, rule-based baseline, neurosymbolic evaluation), full training, evaluation, error analysis, and stress tests.

---

## Reproducibility

- All code and results are in the notebooks.
- All experiments are reproducible by running the notebooks from top to bottom.
- The codebase is modular and well-commented for clarity.


To reproduce:
```bash
git clone https://github.com/dnebhrajani/metafam-knowledge-graph.git
cd metafam-knowledge-graph
pip install -r requirements.txt

# Task 1: Dataset Exploration
jupyter notebook task1_exploration.ipynb
# Run all cells

# Task 2: Community Detection
jupyter notebook task2_communities.ipynb
# Run all cells

# Task 3: Rule Mining
jupyter notebook task3_rule_mining.ipynb
# Run all cells

# Task 4: Link Prediction
jupyter notebook task4_link_prediction.ipynb
# Run all cells (training takes ~5-8 minutes)
```

---

---

## References

# Task 1: Dataset Exploration
jupyter notebook task1_exploration.ipynb

# Task 2: Community Detection
jupyter notebook task2_communities.ipynb

# Task 3: Rule Mining
jupyter notebook task3_rule_mining.ipynb

# Task 4: Link Prediction
jupyter notebook task4_link_prediction.ipynb

# Or use JupyterLab to open all
jupyter lab
```

Then execute all cells sequentially:
- **Menu**: Cell -> Run All
- **Keyboard**: Shift + Enter (run each cell)

### Option 2: Run from Command Line

```bash
# Convert notebook to Python script and run
jupyter nbconvert --to script task1_exploration.ipynb
python task1_exploration.py
```

---

## Methodology & Approach

### Task 1: Dataset Exploration Methodology

### 1. Data Loading & Exploration
- Load train.txt (13,821 edges) using pandas
- Identify unique entities (1,316 people) and relationship types (28 types)
- Analyze relationship distribution

### 2. Graph Construction
- **Choice**: NetworkX DiGraph (directed graph)
- **Rationale**: Family relationships are directional (motherOf != childOf)
- Build graph with edges labeled by relationship type

### 3. Network Analysis

#### Basic Metrics
- **Density**: 0.008 (sparse network)
- **Components**: 50 disconnected families
- **Diameter**: 3 (maximum separation)
- **Average path length**: 1.47 (highly connected)
- **Clustering coefficient**: 0.79 (tight-knit families)

#### Centrality Measures
- **Degree Centrality**: O(n) - Most connected nodes
- **Betweenness Centrality**: O(n³) - Bridge nodes
- **Closeness Centrality**: O(n²) - Central positions
- **PageRank**: O(k·m) - Recursive importance

#### Generation Detection
- Use relationship semantics (parent/grandparent/child)
- Identify 7-generation hierarchy
- Visualize with hierarchical layout

### 4. Anomaly Detection & Hypothesis Testing
Five hypothesis tests performed:
1. **Cycle detection**: Validated DAG structure (except symmetric edges)
2. **Biological constraints**: No >2 parent violations found
3. **Component uniformity**: Chi-square test reveals synthetic data
4. **Missing inverses**: 64.4% lack inverse (single-perspective recording)
5. **Generation consistency**: Many grandparent links lack intermediate nodes

### 5. Visualization Strategy
- **Relationship distribution**: Horizontal bar chart
- **Degree analysis**: Three-panel histogram (in/out/total)
- **Family hierarchy**: Generation-colored network graph
- **Centrality comparison**: Grouped bar chart

### 6. Mathematical Verification
Eight verification tests ensure correctness:
- Handshaking lemma (Σ degrees = 2|E|)
- Degree balance (in + out = total)
- Component partition (disjoint union)
- Diameter bounds (radius ≤ diameter ≤ 2×radius)
- Articulation point validity
- Density formula
- Path length consistency
- Centrality normalization

### Task 2: Community Detection Methodology

#### 1. Algorithm Selection
- **Louvain**: Modularity optimization, O(n log n), hierarchical detection
- **Label Propagation**: Semi-supervised, O(m), parameter-free local propagation
- **Rejected alternatives**: Girvan-Newman (too slow O(m²n)), spectral clustering (requires k), Infomap (wrong model)

#### 2. Evaluation Metrics
- **Modularity**: Standard quality measure for community structure
- **NMI (Normalized Mutual Information)**: Accounts for chance agreement
- **ARI (Adjusted Rand Index)**: Pairwise classification agreement
- **Coverage**: Fraction of edges within communities
- **Conductance**: Community separation quality

#### 3. Analysis Questions
- Q1: How do detected communities align with ground truth families?
- Q2: How many generations are represented in each community?
- Q3: Which individuals serve as bridges between communities?

#### 4. Family Relatedness Score (FRS)
Composite metric with theoretically justified weights:
- **Path distance** (0.4): Most reliable, objective measure
- **Community membership** (0.3): Algorithmic validation
- **Ancestry depth** (0.3): Biological validation

Validated on 5 relationship types, showing successful differentiation.

#### 5. Critical Analysis
- Investigated suspicious "perfect" metrics
- Discovered zero inter-family edges (explains modularity 0.97)
- Analyzed why Label Propagation creates 64 vs 50 communities
- Manual mathematical verification of all metrics

### Task 4: Link Prediction Methodology

#### 1. Problem Formulation
- **Open World Assumption (OWA)**: Missing triples != false (just unobserved)
- **Link prediction task**: Rank entities for incomplete triples (h,r,?) or (?,r,t)
- **Negative sampling reconciliation**: Use corrupted triples as "assumed negative" training signal
- **Filtered evaluation**: Remove all known positives (train ∪ test) to avoid false penalties

#### 2. Negative Sampling
- Corrupt head or tail with random entity replacement
- Filter against all_triple_set to avoid sampling known triples as negatives
- Training uses 1 negative per positive in each batch (efficient)
- Generation is collision-aware with maximum attempt limits

#### 3. TransE Model
- **Core idea**: Model relations as translations in embedding space (h + r ≈ t)
- **Scoring function**: -||h + r - t||₂ (higher score = more plausible)
- **Loss function**: Margin ranking loss (γ=1.0)
- **Training**: Mini-batch SGD (batch=512, lr=0.01, epochs=100)
- **Implementation**: From-scratch NumPy (educational clarity, no external KG libraries)
- **Normalization**: L2-normalize entity embeddings after each update

#### 4. Evaluation Metrics
- **MRR (Mean Reciprocal Rank)**: Average of 1/rank across all predictions
- **Hits@K**: Proportion of correct entities ranked in top-K (K=1,3,10)
- **Filtered ranking**: Standard KG evaluation protocol (removes other known true triples)
- **Both head and tail prediction**: Comprehensive bi-directional evaluation

#### 5. Sanity Checks
- **Loss decrease check**: Verify training convergence
- **Embedding norm tracking**: Detect gradient explosion (should stay ~1.0)
- **Score distribution**: True triples must score higher than negatives on average
- All checks include pass/fail indicators and diagnostic plots

#### 6. Error Analysis Framework
- **Success cases**: Inspect rank=1 predictions (perfect predictions)
- **Failure cases**: Analyze rank>100 predictions (systematic failures)
- **Per-relation analysis**: 28 relation types evaluated separately
- **Hypothesis testing**: Symmetric relations, sparse relations, inverse relations
- **Rule comparison**: Symbolic rules from Task 3 vs learned embeddings

#### 7. Stress Test: Rule Consistency
- **Self-loop violations**: Person predicted as own ancestor
- **Parent constraints**: >2 mothers or >2 fathers
- **Transitivity validation**: Grandmother paths via intermediate parents
- Quantifies % of top predictions violating logical constraints
- Demonstrates need for hybrid symbolic-neural reasoning

---

### Memory Optimization
- Sparse matrix representation (scipy.sparse saves 90% memory)
- Out-of-core processing (HDF5, Parquet for disk-based chunks)
- Graph databases (Neo4j for >10M nodes)
- Compression (gzip, lz4 for edge lists)

---


