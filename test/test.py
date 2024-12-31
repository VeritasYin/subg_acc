'''
Author: Haoteng Yin
Date: 2023-02-10 17:10:14
LastEditors: VeritasYin
LastEditTime: 2023-02-25 13:29:51
FilePath: /subg_acc/test/test.py

Copyright (c) 2023 by VeritasYin, All Rights Reserved. 
'''
import subg
import numpy as np
from scipy.sparse import csr_matrix
from time import time


def edge2csr(file='twitter-2010.txt'):
    row, col = np.loadtxt(file, dtype=int).T
    data = np.ones(len(row), dtype=bool)
    nmax = max(row.max(), col.max())
    return csr_matrix((data, (row, col)), shape=(nmax+1, nmax+1))


G_full = edge2csr('test.edgelist')

ptr = G_full.indptr
neighs = G_full.indices

num_walks = 100
num_steps = 4
target = np.arange(G_full.shape[0])
# np.arange(10)
print(subg.__file__)


walks, rpes = subg.walk_sampler(ptr, neighs, target, num_walks=num_walks, num_steps=num_steps, nthread=4, replacement=True)
# walks: list of walks [#target, num_walks*(num_steps+1)]
# rpes: tuple of unique sampled nodes & positional encoding for each root node (#unique_nodes, [#unique_nodes, num_steps+1])

test_nid = np.random.randint(0, len(target))
uidx = np.unique(walks[test_nid])
assert np.abs(uidx - np.sort(rpes[test_nid][0])).sum() == 0

remap = dict(zip(uidx, np.arange(len(uidx))))
enc = np.zeros((len(uidx), num_steps+1), dtype=int)
for i, node in enumerate(walks[test_nid]):
    enc[remap[node], i%(num_steps+1)] += 1

shuffle_idx = [remap[i] for i in rpes[test_nid][0]]
assert np.abs((enc[shuffle_idx] - rpes[test_nid][1])).sum() == 0

indices, encs = zip(*rpes)
len_idx = list(map(len, indices))
enc_idx = np.array_split(np.arange(1, np.sum(len_idx)+1), np.cumsum(len_idx)[:-1])
test_queries = np.random.randint(0, len(target), size=(test_nid, 2))

tic = time()
joint_walk = subg.walk_join(walks, indices, query=test_queries, nthread=4)
# joint_walk: list of joint walks [2, #queries*num_walks*(num_steps+1)*2]
# format: [[x_x, x_y], [y_x, y_y]]
print(f"OpenMP Join Time for #{len(test_queries)} queries: {time()-tic:.2f}s")

assert joint_walk.shape[-1] // test_nid == num_walks*(num_steps+1)*2

tic = time()
idx_map = dict(zip(target, [dict(zip(indices[i], enc_idx[i])) for i in range(len(target))]))
join_x, join_y = [], []
for x, y in test_queries:
    for w in walks[x]:
        join_x.append(idx_map[x].get(w, 0))
        join_x.append(idx_map[y].get(w, 0))
    for w in walks[y]:
        join_y.append(idx_map[x].get(w, 0))
        join_y.append(idx_map[y].get(w, 0))
print(f"Naive Join Time for #{len(test_queries)} queries: {time()-tic:.2f}s")

assert np.abs(joint_walk[0] - join_x).sum() == 0
assert np.abs(joint_walk[1] - join_y).sum() == 0

nsize, remap, enc = subg.gset_sampler(
    ptr, neighs, target, num_walks=num_walks, num_steps=num_steps, nthread=16)
# check the alignment of nsize of remapping
# nsize: list of unique node sizes of each sampled subgraph [#target]
# enc: unique positional encoding [*, num_steps+1]
# remap: remapping sampled nodes to its corresponding encoding []

assert nsize.sum() == remap.shape[1]
# check the boundary of node and encoding indices
try:
    assert (remap.max(axis=1) - [G_full.shape[0]-1, enc.shape[0]-1]).sum() == 0
# check the encoding of root
except:
    import pdb; pdb.set_trace()

assert (enc[remap[1]][:, 0] == num_walks).sum() == len(target)

assert np.abs((enc[remap[1]].sum(axis=0) / G_full.shape[0] - num_walks).sum()) < 1e-10

nsize, remap, enc, raw_enc = subg.gset_sampler(
    ptr, neighs, target, num_walks=num_walks, num_steps=num_steps, debug=1)
assert (raw_enc[:, 0] == num_walks).sum() == len(target)
assert (enc[remap[1]] - raw_enc).sum() == 0
assert (raw_enc.max(axis=0) - num_walks).sum() == 0
print(f"Test Passed.")
