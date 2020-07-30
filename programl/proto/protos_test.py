#!/usr/bin/env python3

from programl.proto import program_graph_pb2


TEST_PROTO = "programl/proto/example.ProgramGraph.pb"

graph = program_graph_pb2.ProgramGraph()
with open(TEST_PROTO, 'rb') as f:
    graph.ParseFromString(f.read())
print("Loaded graph with", len(graph.node), "nodes")
