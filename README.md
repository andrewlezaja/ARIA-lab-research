# DRL-CO: Maximizing Boosted Influence Spread — Data Pipeline

> Research project targeting publication at **NeurIPS 2026** — NJIT ARIA Lab

This repository contains the **data ingestion and preprocessing pipeline** for a Deep Reinforcement Learning framework targeting the Boosted Influence Maximization (BIM) problem: given a social network and a budget of edges to add, select edges that maximally increase the spread of influence from a fixed seed set.

---

## Problem Overview

Influence Maximization (IM) is a well-studied combinatorial optimization problem — given a budget of *k* seed nodes, which nodes should you activate to maximize information spread through a network? **Boosted** IM flips the lever: instead of choosing seeds, you choose which *edges* to add to the graph to maximize spread from a pre-fixed seed set.

This is a submodular combinatorial optimization problem that sits beyond the reach of traditional greedy algorithms at scale. The broader project approaches it with Deep Reinforcement Learning — this repo covers the data pipeline that feeds it.

---

## My Contribution

I built the end-to-end preprocessing pipeline responsible for ingesting raw SNAP social network datasets, cleaning and validating them, and producing the structured graph splits and candidate/seed files consumed by the DRL training environment.

**Key responsibilities:**
- Ingested and validated four large-scale SNAP datasets (up to 876K nodes), cross-validating outputs against published paper benchmarks to verify data integrity
- Built a modular pipeline to clean raw edgelists, remap node IDs, deduplicate edges, and remove self-loops
- Implemented candidate edge set generation: sampling a subset of edges to withhold from the graph, forming the action space for the RL agent
- Implemented two seed node selection strategies on the refined graph:
  - **High-degree seeding** — greedy MIA-based influence maximization (following the paper's method)
  - **Random seeding** — uniform random sampling across multiple trials
- Structured all outputs (edgelists, node maps, candidate files, seed files, dataset stats) for direct consumption by the training pipeline
- Validated pipeline outputs against published benchmark numbers to ensure correctness before handoff

---

## Datasets

Four large-scale SNAP social network datasets:

| Dataset | Nodes | Description |
|---|---|---|
| Email-Enron | ~36K | Enron email communication network |
| NetHEPT | ~15K | High-energy physics collaboration network |
| Google Web | ~876K | Google web crawl graph |
| Web-NotreDame | ~326K | Notre Dame web graph |

---

## Pipeline Overview

```
Raw SNAP edgelist
       │
       ▼
  Load & validate
  (remap IDs, drop self-loops, dedup)
       │
       ▼
  Sample candidate edge set C (size 2,000)
  Remove C from graph → refined graph G'
       │
       ├──▶ Seed selection on G'
       │         ├── highdeg (greedy MIA-based, K = 1/10/50/100/200)
       │         └── random  (uniform, K = 1/10/50/100/200, 3 trials each)
       │
       └──▶ Outputs
                 ├── graph.edgelist       (refined graph)
                 ├── node_map.json        (original → remapped IDs)
                 ├── stats.json           (node/edge counts, dataset metadata)
                 ├── seeds/<strategy>_K<k>_t<trial>.txt
                 └── candidates/C_2000_seed<strategy>_K<k>_t<trial>.txt
```

---

## Tech Stack

- **Python** — pipeline orchestration
- **NetworkX** — graph loading, manipulation, and influence-based seed selection
- **NumPy / Pandas** — data validation and cross-checking against benchmarks

---

## Status

Active research — targeting NeurIPS 2026 submission.

---

## Author

Andrew Lezaja — NJIT ARIA Lab
