# MetaFam Knowledge Graph Analysis - Technical Report

**Task**: Precog Recruitment - Graph Analysis Assignment  
**Author**: Durga Nebhrajani  
**Institution**: IIIT Hyderabad  
**Date**: January 22, 2026  
**Repository**: https://github.com/dnebhrajani/metafam-knowledge-graph

---

# TASK 1: DATASET EXPLORATION

**Completed**: Dataset loading (13,821 relationships, 1,316 entities, 28 types), graph construction (NetworkX DiGraph, 50 components), network analysis (density 0.008, diameter 3, clustering 0.84), 4 centrality measures, generation detection (7 levels), articulation points (95), 4 visualizations, 5 anomaly tests, 8 verifications.

---

# 1. DATASET OVERVIEW

**Source**: MetaFam family knowledge graph (Precog task) - 13,821 relationships among 1,316 people across 50 families, 28 relationship types, 7 generations detected.

**Format**: Space-separated triples (head, relation, tail). Files: train.txt (analysis), test.txt (reserved).

---

# 2. METHODOLOGY

**Graph Construction**: NetworkX DiGraph to preserve asymmetric relationships.

**Metrics**: Density, diameter, clustering, 4 centrality measures (degree, betweenness, closeness, PageRank).

**Analysis**: Generation detection (semantic), articulation points (Hopcroft-Tarjan), 5 anomaly tests, 8 mathematical verifications.

---

# 3. RESULTS

## 3.1 Dataset Statistics

**Entities**: 1,316 people | **Edges**: 13,821 | **Types**: 28 | **Components**: 50 (size 18-44)

### 3.1.2 Relationship Distribution

**Top relationships**: siblingOf (2,840, 20.6%), spouseOf (1,498, 10.8%), sonOf (1,314, 9.5%), daughterOf (1,196, 8.7%), fatherOf (1,144, 8.3%), motherOf (1,105, 8.0%). Parent-child relations are balanced; grandparent relations present (7.9%).

## 3.2 Network Properties

**Key Metrics**: Density 0.008 (sparse), diameter 3, avg path 1.47, clustering 0.84, 50 components.

**Small-World Analysis**: High clustering (0.84) with short paths (1.47) confirms small-world properties - tight-knit families with efficient connections.

## 3.3 Important Nodes Analysis

### 3.3.1 Centrality Rankings Summary

**Top 5 nodes by each measure:**
- **Degree**: lisa5 (68), isabella11 (62), oskar24 (60), elias6 (59), nico4 (57)
- **Betweenness**: nico4 (0.0127), elias6 (0.0116), lisa5 (0.0110), isabella11 (0.0103), selina10 (0.0098)
- **Closeness**: nico4 (0.4286), elias6 (0.4211), lisa5 (0.4182), isabella11 (0.4118), selina10 (0.4091)
- **PageRank**: isabella11 (0.0019), lisa5 (0.0018), elias6 (0.0017), nico4 (0.0016), oskar24 (0.0016)

### 3.3.2 Universally Important Nodes

**6 individuals appear in top-10 of ALL four centrality measures:**

1. **lisa5** - #1 degree, #3 betweenness, #3 closeness, #2 PageRank
2. **isabella11** - #2 degree, #4 betweenness, #4 closeness, #1 PageRank
3. **elias6** - #4 degree, #2 betweenness, #2 closeness, #3 PageRank
4. **nico4** - #5 degree, #1 betweenness, #1 closeness, #4 PageRank
5. **oskar24** - #3 degree, #6 betweenness, #6 closeness, #5 PageRank
6. **selina10** - #6 degree, #5 betweenness, #5 closeness, #6 PageRank

**Interpretation**: These individuals are critical across all dimensions:
- Most connections (degree)
- Best bridge positions (betweenness)
- Most central locations (closeness)
- Recursively most important (PageRank)

## 3.4 Hierarchical Structure

**7 generations detected** (-3 to +3): Ego generation largest (30.2%, 398 people), symmetric distribution around ego (3 up: 370 people, 3 down: 548 people). Great-grandparents smallest (43, 3.3%).

## 3.5 Articulation Points

### 3.5.1 Overall Statistics

| Metric | Value |
|--------|-------|
| Total articulation points | 95 |
| Percentage of population | 7.2% |
| Components with articulation points | 45 |
| Components without | 5 |
| Average per component (with APs) | 2.1 |
| Maximum in one component | 6 |

### 3.5.2 Distribution

| AP Count | Components |
|----------|------------|
| 0 | 5 |
| 1 | 12 |
| 2 | 18 |
| 3 | 10 |
| 4 | 3 |
| 5 | 1 |
| 6 | 1 |

**Interpretation**:
- 7.2% of people are critical connectors
- Removing them would fragment families
- Most components have 2-3 articulation points
- These are likely patriarch/matriarch figures connecting family branches

## 3.6 Redundancy & Anomalies

**Inverse Pairs**: 17.5% found (2,420), 82.5% missing (11,401) - suggests single-perspective recording.

**Anomaly Tests**: DAG structure valid, no biological violations, but component sizes suspiciously uniform (χ²=48.12, p=0.39), many generation gaps, 82.5% missing inverses.

**Data Quality**: Structurally valid but incomplete recording and synthetic signals detected.

---

# 4. KEY INSIGHTS

## 4.1 Network Structure Insights

### 4.1.1 Small-World Properties

**Finding**: MetaFam exhibits strong small-world characteristics

**Evidence**:
- High clustering coefficient: 0.84 (vs ~0.0 for random)
- Short average path length: 1.47 (similar to random)
- Tight-knit family groups with efficient connections

**Implications**:
- Information spreads quickly within families
- Family members are close despite network size
- Real-world family networks often show this pattern

### 4.1.2 Scale-Free Characteristics

**Finding**: Degree distribution follows power-law-like pattern

**Evidence**:
- Few highly connected hubs (lisa5, isabella11, oskar24)
- Many moderately connected nodes
- Long tail of less connected individuals

**Implications**:
- Hub-and-spoke family structure
- Key individuals (parents) connect many family members
- Resilient to random node removal, vulnerable to hub removal

## 4.2 Centrality Insights

**Key Findings**: Six individuals (lisa5, isabella11, elias6, nico4, oskar24, selina10) rank in top-10 across all centrality measures, indicating multi-dimensional importance:
- **Degree centrality**: Identifies family hubs (parents with many children)
- **Betweenness**: Reveals bridges connecting family branches
- **Closeness**: Shows central positions for efficient communication
- **PageRank**: Captures recursive importance through network structure

### 4.2.1 Universal Importance

**6 individuals rank in top-10 across ALL measures**:
lisa5, isabella11, elias6, nico4, oskar24, selina10

**Implication**:
- These are truly critical family members
- Multi-dimensional importance (connections, bridges, centrality, influence)
- Likely patriarch/matriarch figures

## 4.3 Generational Insights

**Key Findings**: 7 generations detected (3 up, 1 ego, 3 down) with ego generation largest (30.2%), indicating middle-generation data collection perspective. Many grandparent relationships lack intermediate nodes, suggesting incomplete recording or "shortcut" edges that affect path length calculations.

## 4.4 Network Fragmentation

**Key Findings**: 50 disconnected families with suspiciously uniform sizes (18-44 nodes) and 95 articulation points (7.2% of population). These critical connectors, likely patriarch/matriarch figures, control network integrity across 45 components (avg 2.1 per component). No inter-family marriages recorded.

## 4.5 Data Quality Summary

**Strengths**: Valid DAG structure, 28 relationship types, 17.5% inverse validation.

**Weaknesses**: 82.5% missing inverses, uniform sizes (synthetic signal), no cross-family links.

---

# 5. VISUALIZATIONS

**Four plots created**:
1. **Relationship distribution** (bar chart): siblingOf dominates (20.6%)
2. **Degree distribution** (histograms): Power-law pattern, few hubs (>60 degree)
3. **Family hierarchy** (network graph): 7-generation structure, largest component (44 nodes)
4. **Centrality comparison** (grouped bars): 6 universally important nodes identified

---

# 6. TECHNICAL DETAILS

**Stack**: pandas 2.0+, numpy 1.24+, networkx 3.1+, matplotlib 3.7+, seaborn 0.12+, scipy 1.10+

**Performance**: ~5s total (betweenness: ~2s, O(n³) bottleneck), ~4MB memory

**Scalability**: 10x viable with NetworkX, 100x+ needs sampling/distributed approaches

---

# 7. CONCLUSIONS
| **1000x** | 1.3M | 13.8M | ~54days | 4 GB | Need igraph/Spark |

### Scaling Recommendations

**For 10x scale (130K edges)**:
- Current approach sufficient
- NetworkX handles well
- May need to sample betweenness

**For 100x scale (1.3M edges)**:
- Switch betweenness to sampling
- Use igraph for better performance
- Consider approximate algorithms

**For 1000x scale (13M edges)**:
- Distributed computing (Spark GraphX)
- Approximate centrality algorithms
- Graph databases (Neo4j)

## 6.4 Code Quality

### Best Practices Applied

1. **Modular Structure**
   - Clear cells for each analysis step
   - Reusable functions for metrics
   - Separated visualization code

2. **Documentation**
   - Extensive comments
   - Methodology justification
   - Mathematical formulas included

3. **Verification**
   - 8 mathematical tests implemented
   - Assertion checks for invariants
   - Sanity checks on results

4. **Reproducibility**
   - All dependencies listed
   - Random seeds set (where applicable)
   - Full results logged in notebook

5. **Professional Standards**
   - No emojis (academic formatting)
   - Markdown for text (not print statements)
   - Clean, readable code

---

# 7. LIMITATIONS & FUTURE WORK

## 7.1 Current Limitations

**Data Issues**:
- 82.5% missing inverse relationships (incomplete recording)
- 50 disconnected families (no cross-family marriages)
- Uniform component sizes suggest synthetic generation
- Generation gaps (missing intermediate nodes)

**Methodological Constraints**:
- Static analysis (no temporal dynamics)
- Unweighted edges (all relationships treated equally)
- Binary relationships (no strength/frequency modeling)

**Computational Constraints**:
- O(n³) betweenness limits scalability beyond 10K nodes
- Component-level analysis misses global patterns


---

# 8. CONCLUSIONS

## 8.1 Summary of Findings

This comprehensive analysis of the MetaFam knowledge graph has revealed:

### Structural Properties
- **Small-world network**: High clustering (0.84) + short paths (1.47)
- **Scale-free characteristics**: Hub-based family organization
- **50 disconnected families**: No inter-family marriages recorded
- **7 generations**: Realistic 3-level depth in both directions

### Important Nodes
- **6 universally important individuals**: Rank top-10 in all centrality measures
- **95 articulation points (7.2%)**: Critical family connectors
- **lisa5, isabella11, elias6** emerge as most critical across multiple dimensions

### Data Quality
- **Structurally valid**: DAG structure, no biological violations
- **Incomplete (82.5% missing inverses)**: Single-perspective recording
- **Synthetic signals**: Suspiciously uniform component sizes
- **Usable for ML**: Good size (1,316 nodes), rich types (28 relations)

### Key Insights
- Families exhibit efficient small-world communication structures
- A small percentage of individuals control network integrity
- Data likely collected from middle-generation perspective (ego bias)
- Missing cross-family relationships limit global analysis

## 8.2 Task 1 Summary

**Completed**: Dataset exploration (13,821 relationships, 1,316 entities), graph construction (NetworkX DiGraph), network analysis (density, diameter, clustering, paths), 4 centrality measures, hierarchical generation detection (7 levels), articulation points (95 identified), 4 visualizations, 5 anomaly tests, 8 mathematical verifications.

**Key Contributions**: Multi-centrality analysis identifying 6 universally important nodes, articulation point distribution analysis, small-world property validation, data quality quantification (17.5% redundancy), generation detection algorithm.

## 8.3 Path Forward

Task 1 establishes a comprehensive foundation: network structure understood (small-world, 50 components), important nodes identified (6 universal, 95 articulation points), data quality assessed (17.5% redundancy, synthetic signals detected).

The analysis provides rich features (centrality scores, generation levels, component structure) and insights (missing inverses, uniform distributions) that will inform subsequent tasks. The notebook is fully reproducible, mathematically verified, and ready for extension.

---

# REFERENCES

**Graph Theory**: Watts & Strogatz (1998) small-world networks, Freeman (1978) centrality, Page et al. (1999) PageRank, Hopcroft & Tarjan (1973) articulation points.

**Tools**: NetworkX (Hagberg et al. 2008), pandas (McKinney 2010), NumPy (Harris et al. 2020).

**Dataset**: MetaFam Knowledge Graph, Precog IIIT Hyderabad (2026).

---

**END OF TASK 1 REPORT**

---

**Document Version**: 1.0  
**Last Updated**: January 22, 2026
│   ├── Cells 26-28: Insights
│   └── Cell 29: Mathematical Verification (8 tests)
│
├── train.txt                     # Training data (13,821 edges)
├── test.txt                      # Test data
├── README.md                     # Comprehensive documentation
├── requirements.txt              # Python dependencies
└── TECHNICAL_REPORT.md          # This document

```

---

**END OF TASK 1 REPORT**
