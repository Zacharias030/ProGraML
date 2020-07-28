#!/usr/bin/env python3

from deeplearning.ml4pl.poj104 import dataset

TEST_PROTO = "programl/proto/example.ProgramGraph.pb"

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
