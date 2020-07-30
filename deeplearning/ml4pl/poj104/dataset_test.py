#!/usr/bin/env python3

import os
from deeplearning.ml4pl.poj104 import dataset

TEST_PROTO = "programl/proto/example.ProgramGraph.pb"
DEVMAP = os.path.expanduser("~/programl/devmap")

graph = dataset.load(TEST_PROTO)
print("Loaded graph with", len(graph.node), "nodes and", len(graph.edge), "edges")

vocab = dataset.load_vocabulary(dataset.PROGRAML_VOCABULARY)
print("Loaded", len(vocab), "element ProGraML vocabulary")

vocab = dataset.load_vocabulary(dataset.CDFG_VOCABULARY)
print("Loaded", len(vocab), "element CDFG vocabulary")

data = dataset.nx2data(graph, vocab)
print("Created unlabelled graph data", data)

data = dataset.nx2data(graph, vocab, ablate_vocab=dataset.AblationVocab.NO_VOCAB)
print("Ablated vocab", data)

data = dataset.nx2data(graph, vocab, ablate_vocab=dataset.AblationVocab.NODE_TYPE_ONLY)
print("Ablated node type", data)

data = dataset.nx2data(graph, vocab, "poj104_label")
print("Created POJ-104 labelled graph data", data)

graph = dataset.load(TEST_PROTO, cdfg=True)
print("Loaded CDFG graph with", len(graph.node), "nodes and", len(graph.edge), "edges")

if os.path.isdir(DEVMAP):
    dataset.DevmapDataset(root=DEVMAP, split="amd")
    dataset.DevmapDataset(root=DEVMAP, split="amd", cdfg=True)

assert dataset.filename("foo", False, dataset.AblationVocab.NONE) == "foo_data.pt"
assert dataset.filename("foo", True, dataset.AblationVocab.NONE) == "foo_cdfg_data.pt"
assert dataset.filename("foo", True, dataset.AblationVocab.NO_VOCAB) == "foo_cdfg_no_vocab_data.pt"
