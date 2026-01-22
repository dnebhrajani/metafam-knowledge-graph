# MetaFam Knowledge Graph Analysis

## Project Overview

This repository contains a comprehensive exploratory analysis of the MetaFam family knowledge graph dataset as part of the Precog recruitment task. The analysis uses graph theory techniques to uncover patterns in family structures, identify important family members, and understand hierarchical relationships.

---

## Directory Structure

```
metafam-knowledge-graph/
│
├── Graph_Analysis.ipynb          # Main Jupyter notebook with all analysis
├── train.txt                     # Training dataset (13,821 relationships)
├── test.txt                      # Test dataset
├── README.md                     # This file
└── requirements.txt              # Python dependencies
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

### Optional (for notebook)
- **jupyter** (1.0+): Interactive notebook environment
- **ipykernel**: Jupyter kernel for Python

All dependencies are listed in `requirements.txt` for easy installation.

---

## Running the Analysis

### Option 1: Run Complete Notebook

```bash
# Launch Jupyter Notebook
jupyter notebook Graph_Analysis.ipynb

# Or use JupyterLab
jupyter lab Graph_Analysis.ipynb
```

Then execute all cells sequentially:
- **Menu**: Cell → Run All
- **Keyboard**: Shift + Enter (run each cell)

### Option 2: Run from Command Line

```bash
# Convert notebook to Python script and run
jupyter nbconvert --to script Graph_Analysis.ipynb
python Graph_Analysis.py
```

### Expected Runtime
- **Full notebook execution**: ~5-10 seconds
- **Memory usage**: <100 MB
- **Output**: All visualizations and statistics printed to notebook

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
jupyter notebook Graph_Analysis.ipynb
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
4. **No cross-family links**: Marriages between families not captured

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

**Last Updated**: January 22, 2026  
**Status**: Task 1 Complete - Analysis Ready for Extension
