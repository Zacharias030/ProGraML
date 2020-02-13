from pathlib import Path
import pickle

import time, os, json, sys


import numpy as np
#from matplotlib import pyplot as plt
import networkx as nx
#import tqdm
#import torch
#from torch_geometric.data import Data, DataLoader, InMemoryDataset
#import torch_geometric


# make this file executable from anywhere
#if __name__ == '__main__':
full_path = os.path.realpath(__file__)
print(full_path)
repo_root = full_path.rsplit('ProGraML', maxsplit=1)[0] + 'ProGraML'
print(repo_root)
#insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, repo_root)
repo_root = Path(repo_root)


from deeplearning.ml4pl.graphs.unlabelled.llvm2graph import graph_builder

builder = graph_builder.ProGraMLGraphBuilder() #opt='/usr/bin/opt')
builder7 = graph_builder.ProGraMLGraphBuilder(opt='/usr/bin/opt')

file_to_debug = '/mnt/data/llvm/master_thesis_datasets/unsupervised_ncc_data/amd_app_sdk/amd/AtomicCounters.ll'
#with open('/mnt/data/llvm/master_thesis_datasets/unsupervised_ncc_data/amd_app_sdk/amd_ocl/AMDAPPSDK-3.0_samples_bolt_BoxFilterSAT_BoxFilterSAT_Kernels.ll', 'r') as f:
#with open('/mnt/data/llvm/master_thesis_datasets/unsupervised_ncc_data/eigen/eigen_matmul_3/eigen_matmul-266.ll_', 'r') as f:
#with open(repo_root / 'deeplearning/ml4pl/poj104' / '71.ll', 'r') as f:
with open(file_to_debug, 'r') as f:
    ll = f.read()

nx_graph = builder.Build(ll)
nx_graph7 = builder7.Build(ll)











for i in range(5):
    nn = builder.Build(ll)
    print(f"====== {i} =====")
    for n, d in nn.nodes.items():
        print(n, d)
        if 15 >= n and n > 14:
            pass
    print('\n\n\n\n')
    
    
di = []
ddi = []

for n, d in nn.nodes.items():
    match = None
    is_ok = False
    for nn, dd in nx_graph.nodes.items():
        if d == dd:
            #if is_ok:
            #    assert False, f'double match: {match}, double: {nn, dd}'
            is_ok = True
            match = [n, d, nn, dd]
    assert is_ok, f"is not okay! {n} and data {d}"

print('done')

            