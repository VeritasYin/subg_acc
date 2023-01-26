# Subgraph Operation Accelerator
The `subg_acc` package is an extension library based on C and openmp to accelerate subgraph operations in subgraph-based graph representation learning (SGRL) with multithreading enabled. Follow the principles of algorithm system co-design in [SUREL](https://arxiv.org/abs/2202.13538)/[SUREL+](https://github.com/VeritasYin/SUREL_Plus/blob/main/manuscript/SUREL_Plus_Full.pdf), query-level subgraphs (of link/motif) (e.g. ego-network in canonical SGRLs) are decomposed into reusable node-level ones. Currently, `subg_acc` consists of the following methods for the realization of scalable SGRLs:

- `run_walk` random-walk based subgraph sampling
- `rpe_encoder` relative positional encoding (localized structural feature construction)
- `sjoin` online subgraph joining that reconstructs the query-level subgraph from node-level ones to serve queries (a set of nodes)

## Requirements
(Other versions may work, but are untested)

- python >= 3.8
- gcc >= 8.4
- cmake >= 3.16
- make >= 4.2

## Installation
```
python setup.py install
```

