# MetaFam Knowledge Graph Analysis

## Project Overview

This repository contains a comprehensive exploratory analysis of the MetaFam family knowledge graph dataset as part of the Precog recruitment task. The analysis uses graph theory techniques to uncover patterns in family structures, identify important family members, and understand hierarchical relationships.

---

## Directory Structure

```
metafam-knowledge-graph/
│
├── Graph_Analysis.ipynb          # Task 1: Dataset exploration
├── task2_communities.ipynb       # Task 2: Community detection
├── train.txt                     # Training dataset (13,821 relationships)
├── test.txt                      # Test dataset
├── README.md                     # This file
├── requirements.txt              # Python dependencies
└── TECHNICAL_REPORT.md           # Comprehensive technical documentation
```

---

## Task Completion Status

### Task 1: Dataset Exploration (COMPLETED)

**What was completed:**
- Comprehensive dataset loading and exploration
- Statistical analysis: 1,316 people, 28 relationship types, 50 family components
- Graph construction and network analysis (density, diameter, clustering, path lengths)
- Multiple centrality measures (Degree, Betweenness, Closeness, PageRank)
- Generation identification (7-level hierarchy detected)
- Articulation point analysis (95 critical connectors)
- Multiple visualizations:
  - Relationship type distribution
  - Degree distributions (in/out/total)
  - Hierarchical family structure graph
  - Centrality comparison charts
- Anomaly detection with 5 hypothesis tests
- Mathematical verification of all graph-theoretic claims
- Comprehensive insights summary (200+ lines)

**Key Findings:**
- Small-world network properties (high clustering 0.84, short paths 1.47)
- 7 generations detected (3 up, 3 down, 1 ego)
- 17.5% relationship redundancy (inverse pairs)
- Suspicious uniform component sizes suggest synthetic/constrained data
- 6 universally important nodes across all centrality measures

### Task 2: Community Detection (COMPLETED)

**What was completed:**
- **Problem framing**: Explicit definition of clustering task, acknowledged genealogical ambiguity
- **Data preparation**: Documented undirected graph conversion, edge collapse strategy
- Two complementary algorithms implemented (Louvain, Label Propagation)
- **Hyperparameter exploration**: Tested 6 resolution values (0.5-2.0), validated default choice
- **Random baseline comparison**: Demonstrated 417,000%+ improvement over random clustering
- Five evaluation metrics (Modularity, NMI, ARI, Coverage, Conductance)
- Algorithm selection justified (complexity analysis, theoretical basis)
- Mathematical verification (manual modularity calculation, graph theory validation)
- **Structural evaluation**: Component-community overlap analysis (100% pure communities)
- **Generation entropy analysis**: Calculated information entropy (~1.72) showing multi-generational families
- **Visual subgraph inspection**: 3 representative communities with generation-colored nodes
- Three analysis questions answered (community-family alignment, generations per community, bridge individuals)
- Family Relatedness Score (FRS) metric created with justified weights
- **FRS superiority demonstration**: Showed why FRS beats simple hop-count with edge cases
- Critical dataset discovery: Zero inter-family edges (explains perfect metrics)
- Comprehensive comparison visualizations

**Key Findings:**
- **Random baseline**: 0.0002 modularity (essentially random) vs 0.9794 Louvain (417,000%+ improvement)
- **Louvain**: 50 communities (modularity 0.9794, NMI 1.0000, ARI 1.0000)
- **Label Propagation**: 64 communities (modularity 0.9652, NMI 0.9844, ARI 0.9576)
- **Hyperparameter robustness**: Resolution γ=0.5 to 2.0 all produce 50 communities (perfect component structure)
- **Perfect community purity**: 100% of communities (both algorithms) stay within single families
- **Perfect Louvain metrics explained**: dataset has ZERO inter-family edges
- **Generation entropy**: 1.72 average entropy indicates multi-generational families (not age cohorts)
- **Generation span**: Average 6-7 generations per community (great-grandparents to great-grandchildren)
- **LP's 14 extra communities** reveal meaningful subfamilies within large families
- **95 bridge individuals** (7.22%) identified as critical connectors
- **FRS metric** successfully differentiates relationship types (parent-child 0.65 vs hop-count 1.0)

---

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/dnebhrajani/metafam-knowledge-graph.git
cd metafam-knowledge-graph
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Dependencies

The project uses the following Python libraries:

### Core Libraries
- **pandas** (2.0+): Data manipulation and analysis
- **numpy** (1.24+): Numerical computing
- **networkx** (3.1+): Graph creation and analysis

### Visualization
- **matplotlib** (3.7+): Plotting and visualization
- **seaborn** (0.12+): Statistical data visualization

### Statistical Analysis
- **scipy** (1.10+): Scientific computing and statistical tests

### Community Detection (Task 2)
- **python-louvain** (0.15+): Louvain modularity optimization
- **scikit-learn** (1.3+): Label Propagation, NMI/ARI metrics

### Optional (for notebook)
- **jupyter** (1.0+): Interactive notebook environment
- **ipykernel**: Jupyter kernel for Python

All dependencies are listed in `requirements.txt` for easy installation.

---

## Running the Analysis

### Option 1: Run Complete Notebook

```bash
# Task 1: Dataset Exploration
jupyter notebook Graph_Analysis.ipynb

# Task 2: Community Detection
jupyter notebook task2_communities.ipynb

# Or use JupyterLab to open both
jupyter lab
```

Then execute all cells sequentially:
- **Menu**: Cell -> Run All
- **Keyboard**: Shift + Enter (run each cell)

### Option 2: Run from Command Line

```bash
# Convert notebook to Python script and run
jupyter nbconvert --to script Graph_Analysis.ipynb
python Graph_Analysis.py
```

### Expected Runtime
- **Task 1 (Graph_Analysis.ipynb)**: ~5-10 seconds
- **Task 2 (task2_communities.ipynb)**: ~3-5 seconds
- **Memory usage**: <100 MB per notebook
- **Output**: All visualizations and statistics printed to notebooks

---

## Methodology & Approach

### 1. Data Loading & Exploration
- Load train.txt (13,821 edges) using pandas
- Identify unique entities (1,316 people) and relationship types (28 types)
- Analyze relationship distribution

### 2. Graph Construction
- **Choice**: NetworkX DiGraph (directed graph)
- **Rationale**: Family relationships are directional (motherOf ≠ childOf)
- Build graph with edges labeled by relationship type

### 3. Network Analysis

#### Basic Metrics
- **Density**: 0.008 (sparse network)
- **Components**: 50 disconnected families
- **Diameter**: 3 (maximum separation)
- **Average path length**: 1.47 (highly connected)
- **Clustering coefficient**: 0.84 (tight-knit families)

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
4. **Missing inverses**: 82.5% lack inverse (single-perspective recording)
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

---

## Key Results

### Important Nodes Identified
**Top 6 universally important individuals** (appear in top-10 of all centrality measures):
- lisa5, isabella11, elias6, nico4, oskar24, selina10

### Network Properties
- **Small-world structure**: High clustering + short paths
- **Scale-free characteristics**: Hub-based family organization
- **95 articulation points**: Critical connectors (7.2% of population)

### Data Quality Insights
- **HIGH-QUALITY**: 17.5% inverse validation shows consistency
- **INCOMPLETE**: 82.5% missing inverses, gaps in intermediate nodes
- **CONSTRAINED**: Uniform component sizes (26-27 people) suggest synthetic generation

### Task 2 Results: Community Detection

**Algorithm Performance:**
- **Louvain**: 50 communities, modularity 0.9794, NMI 1.0000, ARI 1.0000
- **Label Propagation**: 64 communities, modularity 0.9652, NMI 0.9844, ARI 0.9576

**Critical Discovery:**
- Dataset has **ZERO inter-family edges** (completely disconnected components)
- This explains "perfect" Louvain metrics (trivially rediscovers components)
- Label Propagation's 14 extra communities reveal meaningful subfamilies

**Generations per Community:**
- Average: 4.14 generations per community
- Most communities span 3-5 generations
- Confirms multi-generational family structures

**Bridge Individuals:**
- 95 articulation points identified (7.22% of network)
- Primarily cousins (50.5%) and grandparents (31.6%)
- Removing them would disconnect family branches

**FRS Metric Validation:**
- Successfully differentiates relationships: parent-child (0.60) > grandparent-grandchild (0.50) > cousin (0.35)
- Weights justified: path most reliable (0.4), community/ancestry equal secondary (0.3 each)
- Limitations identified: sensitive to community detection errors

---

## Reproducibility

All results in the notebook are fully reproducible:

1. **Random seeds set**: For consistent sampling in verification
2. **Deterministic algorithms**: NetworkX algorithms are deterministic
3. **Logged outputs**: Every computation has printed results
4. **Version controlled**: All code tracked in Git

To reproduce:
```bash
git clone https://github.com/dnebhrajani/metafam-knowledge-graph.git
cd metafam-knowledge-graph
pip install -r requirements.txt

# Task 1: Dataset Exploration
jupyter notebook Graph_Analysis.ipynb
# Run all cells

# Task 2: Community Detection
jupyter notebook task2_communities.ipynb
# Run all cells
```

---

## Performance & Scalability

### Current Performance
- **Nodes**: 1,316
- **Edges**: 13,821
- **Runtime**: <5 seconds
- **Memory**: <100 MB

### Scalability Limits
- **10x scale** (130K edges): NetworkX sufficient
- **100x scale** (1.3M edges): Need betweenness sampling
- **1000x scale** (13M edges): Switch to igraph/graph-tool
- **Web scale** (1B+ edges): Distributed computing (Spark GraphX)

---

## Limitations & Future Work

### Current Limitations
1. **Static analysis**: No temporal dynamics (births/deaths over time)
2. **Binary relationships**: No relationship strength/frequency modeling
3. **Missing data**: 82.5% inverse relationships not recorded
4. **No cross-family links**: Dataset contains only intra-family relationships (no edges between the 50 families)

### Future Extensions
1. **Temporal analysis**: If timestamps available
2. **Probabilistic modeling**: Handle uncertainty in relationships
3. **Advanced visualizations**: Interactive tools and time-series animations

---

## Citation & References

### Dataset
- MetaFam Knowledge Graph (Precog recruitment task)
- Format: Space-separated triples (head, relation, tail)

### Tools & Libraries
- NetworkX: Hagberg, A., Schult, D., & Swart, P. (2008). "Exploring network structure, dynamics, and function using NetworkX."
- Python Scientific Stack: NumPy, pandas, matplotlib, scipy

### Graph Theory Concepts
- Centrality measures: Freeman (1978), Page et al. (1999)
- Small-world networks: Watts & Strogatz (1998)
- Articulation points: Hopcroft & Tarjan (1973)

---

## Scalability Considerations

### Current Performance
- **Dataset**: 1,316 nodes, 13,821 edges
- **Runtime**: <5 seconds total
- **Memory**: ~4 MB
- **Bottleneck**: Betweenness centrality O(n³)

### Scaling to Larger Graphs

**10x Scale (130K edges)**:
- NetworkX remains viable
- Sample betweenness centrality (10% sample, extrapolate)
- Estimated runtime: 5-10 minutes

**100x Scale (1.3M edges)**:
- Switch to igraph (10-100x faster than NetworkX)
- Use approximate algorithms (Brandes approximation, HyperANF)
- Parallel processing with multiprocessing
- Estimated runtime: 10-30 minutes

**1000x Scale (13M edges)**:
- Distributed computing required (Apache Spark GraphX)
- Graph databases for storage (Neo4j)
- Approximate centrality algorithms mandatory
- Graph partitioning (METIS)
- Estimated runtime: 1-3 hours on cluster

### Sampling Strategies
- **Random node sampling**: 10% sample preserves degree distribution
- **BFS sampling**: Preserves local structure from seed nodes
- **Forest Fire sampling**: Captures community structure (recommended)
- **Snowball sampling**: K-hop neighborhoods for ego networks

### Memory Optimization
- Sparse matrix representation (scipy.sparse saves 90% memory)
- Out-of-core processing (HDF5, Parquet for disk-based chunks)
- Graph databases (Neo4j for >10M nodes)
- Compression (gzip, lz4 for edge lists)

---

## Contact & Support

**Author**: Durga Nebhrajani  
**Institution**: IIIT Hyderabad  
**Task**: Precog Recruitment - MetaFam Knowledge Graph Analysis  
**GitHub**: https://github.com/dnebhrajani/metafam-knowledge-graph

For questions or issues:
1. Open a GitHub issue
2. Check the notebook for detailed methodology
3. Review the comprehensive insights section

---

## License

This project is created for educational and recruitment purposes. The dataset is provided by Precog, IIIT Hyderabad.

---

## Acknowledgments

- **Precog, IIIT Hyderabad** for providing the dataset and task structure
- **NetworkX community** for excellent graph analysis tools
- **Scientific Python ecosystem** for robust data analysis capabilities

---

**Last Updated**: January 23, 2026  
**Status**: Task 1 Complete | Task 2 Complete - Ready for Tasks 3-4
