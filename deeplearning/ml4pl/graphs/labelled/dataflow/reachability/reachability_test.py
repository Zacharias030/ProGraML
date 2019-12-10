"""Unit tests for //deeplearning/ml4pl/graphs/labelled/dataflow/reachability."""
import random

import networkx as nx

from deeplearning.ml4pl.graphs import programl
from deeplearning.ml4pl.graphs import programl_pb2
from deeplearning.ml4pl.graphs.labelled.dataflow.reachability import (
  reachability,
)
from deeplearning.ml4pl.testing import random_networkx_generator
from deeplearning.ml4pl.testing import random_programl_generator
from labm8.py import decorators
from labm8.py import test

FLAGS = test.FLAGS

###############################################################################
# Fixtures.
###############################################################################


@test.Fixture(scope="function")
def graph() -> nx.MultiDiGraph:
  """A program graph with linear control flow."""
  builder = programl.GraphBuilder()
  a = builder.AddNode(x=[0])
  b = builder.AddNode(x=[0])
  c = builder.AddNode(x=[0])
  d = builder.AddNode(x=[0])
  builder.AddEdge(a, b)
  builder.AddEdge(b, c)
  builder.AddEdge(c, d)
  return builder.g


@test.Fixture(scope="function")
def annotator() -> reachability.ReachabilityAnnotator:
  return reachability.ReachabilityAnnotator()


@test.Fixture(
  scope="function",
  params=list(random_programl_generator.EnumerateProtoTestSet()),
)
def real_graph(request) -> programl_pb2.ProgramGraph:
  """A test fixture which yields one of 100 "real" graphs."""
  return request.param


###############################################################################
# Tests.
###############################################################################


def test_Annotate_reachable_node_count_D(
  graph: nx.MultiDiGraph, annotator: reachability.ReachabilityAnnotator
):
  annotated = annotator.Annotate(graph, 3)
  assert annotated.data_flow_positive_node_count == 1


def test_Annotate_reachable_node_count_A(
  graph: nx.MultiDiGraph, annotator: reachability.ReachabilityAnnotator
):
  annotated = annotator.Annotate(graph, 0)
  assert annotated.data_flow_positive_node_count == 4


def test_Annotate_data_flow_steps_D(
  graph: nx.MultiDiGraph, annotator: reachability.ReachabilityAnnotator
):
  annotated = annotator.Annotate(graph, 3)
  assert annotated.data_flow_steps == 1


def test_Annotate_data_flow_steps_A(
  graph: nx.MultiDiGraph, annotator: reachability.ReachabilityAnnotator
):
  annotated = annotator.Annotate(graph, 0)
  assert annotated.data_flow_steps == 4


def test_Annotate_node_x(
  graph: nx.MultiDiGraph, annotator: reachability.ReachabilityAnnotator
):
  annotated = annotator.Annotate(graph, 0)
  assert annotated.node[0].x == [0, 1]
  assert annotated.node[1].x == [0, 0]
  assert annotated.node[2].x == [0, 0]
  assert annotated.node[3].x == [0, 0]


def test_Annotate_node_y(
  graph: nx.MultiDiGraph, annotator: reachability.ReachabilityAnnotator
):
  annotated = annotator.Annotate(graph, 1)
  assert annotated.node[0].y == [1, 0]
  assert annotated.node[1].y == [0, 1]
  assert annotated.node[2].y == [0, 1]
  assert annotated.node[3].y == [0, 1]


def test_MakeAnnotated_real_graphs(
  real_graph: programl_pb2.ProgramGraph,
  annotator: reachability.ReachabilityAnnotator,
):
  """Opaque black-box test of reachability annotator."""
  annotated = list(annotator.MakeAnnotated(real_graph, n=100))
  assert len(annotated) <= 100


@decorators.loop_for(seconds=10)
def test_fuzz_MakeAnnotated(annotator: reachability.ReachabilityAnnotator,):
  """Opaque black-box test of reachability annotator."""
  n = random.randint(1, 100)
  proto = random_programl_generator.CreateRandomProto()
  annotated = list(annotator.MakeAnnotated(proto, n=n))
  assert len(annotated) <= n


if __name__ == "__main__":
  test.Main()
