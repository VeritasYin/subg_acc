# **SubGACC**: Subgraph Operation Accelerator
<p align="center">
    <a href="https://github.com/VeritasYin/subg_acc/blob/master/LICENSE"><img src="https://img.shields.io/badge/License-BSD%202--Clause-red.svg"></a>
    <a href="https://github.com/VeritasYin/subg_acc/blob/master/setup.py"><img src="https://img.shields.io/badge/Version-v2.0-orange" alt="Version"></a>
    <a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FVeritasYin%2Fsubg_acc&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Hits&edge_flat=false"/></a>
</p>

The `subg_acc` package is an extension library based on C and openmp to accelerate subgraph operations in subgraph-based graph representation learning (SGRL) with multithreading enabled. Follow the principles of algorithm system co-design, query-level subgraphs (of link/motif) (e.g. ego-network in canonical SGRLs) are decomposed into reusable node-level ones (e.g. walk in [SUREL](https://arxiv.org/abs/2202.13538) by `WalkSampler`, set in [SUREL+](https://github.com/VeritasYin/SUREL_Plus/blob/main/manuscript/SUREL_Plus_Full.pdf) by `SetSampler`). Currently, `subg_acc` consists of the following methods for the realization of scalable SGRLs:

- [New] `SetSampler` walk-based set sampling with LP structure encoder
- `WalkSampler` walk-based subgraph sampling with relative positional encoding
- `BatchSampler` walk-based sampling of training queries in batches
- `SubGJoin` online subgraph joining that reconstructs the query-level subgraph from node-level ones to serve queries (a set of nodes)

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

