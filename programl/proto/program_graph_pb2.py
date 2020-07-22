# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: programl/proto/program_graph.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from programl.proto import edge_pb2 as programl_dot_proto_dot_edge__pb2
from programl.proto import features_pb2 as programl_dot_proto_dot_features__pb2
from programl.proto import function_pb2 as programl_dot_proto_dot_function__pb2
from programl.proto import module_pb2 as programl_dot_proto_dot_module__pb2
from programl.proto import node_pb2 as programl_dot_proto_dot_node__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='programl/proto/program_graph.proto',
  package='programl',
  syntax='proto3',
  serialized_pb=_b('\n\"programl/proto/program_graph.proto\x12\x08programl\x1a\x19programl/proto/edge.proto\x1a\x1dprograml/proto/features.proto\x1a\x1dprograml/proto/function.proto\x1a\x1bprograml/proto/module.proto\x1a\x19programl/proto/node.proto\"\xb8\x01\n\x0cProgramGraph\x12\x1c\n\x04node\x18\x01 \x03(\x0b\x32\x0e.programl.Node\x12\x1c\n\x04\x65\x64ge\x18\x02 \x03(\x0b\x32\x0e.programl.Edge\x12$\n\x08\x66unction\x18\x04 \x03(\x0b\x32\x12.programl.Function\x12 \n\x06module\x18\x05 \x03(\x0b\x32\x10.programl.Module\x12$\n\x08\x66\x65\x61tures\x18\x06 \x01(\x0b\x32\x12.programl.Features\"^\n\x10ProgramGraphList\x12#\n\x07\x63ontext\x18\x01 \x01(\x0b\x32\x12.programl.Features\x12%\n\x05graph\x18\x02 \x03(\x0b\x32\x16.programl.ProgramGraphB/\n\x0c\x63om.programlB\x11ProgramGraphProtoP\x01Z\nprogramlpbb\x06proto3')
  ,
  dependencies=[programl_dot_proto_dot_edge__pb2.DESCRIPTOR,programl_dot_proto_dot_features__pb2.DESCRIPTOR,programl_dot_proto_dot_function__pb2.DESCRIPTOR,programl_dot_proto_dot_module__pb2.DESCRIPTOR,programl_dot_proto_dot_node__pb2.DESCRIPTOR,])
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_PROGRAMGRAPH = _descriptor.Descriptor(
  name='ProgramGraph',
  full_name='programl.ProgramGraph',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='node', full_name='programl.ProgramGraph.node', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='edge', full_name='programl.ProgramGraph.edge', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='function', full_name='programl.ProgramGraph.function', index=2,
      number=4, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='module', full_name='programl.ProgramGraph.module', index=3,
      number=5, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='features', full_name='programl.ProgramGraph.features', index=4,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=194,
  serialized_end=378,
)


_PROGRAMGRAPHLIST = _descriptor.Descriptor(
  name='ProgramGraphList',
  full_name='programl.ProgramGraphList',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='context', full_name='programl.ProgramGraphList.context', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='graph', full_name='programl.ProgramGraphList.graph', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=380,
  serialized_end=474,
)

_PROGRAMGRAPH.fields_by_name['node'].message_type = programl_dot_proto_dot_node__pb2._NODE
_PROGRAMGRAPH.fields_by_name['edge'].message_type = programl_dot_proto_dot_edge__pb2._EDGE
_PROGRAMGRAPH.fields_by_name['function'].message_type = programl_dot_proto_dot_function__pb2._FUNCTION
_PROGRAMGRAPH.fields_by_name['module'].message_type = programl_dot_proto_dot_module__pb2._MODULE
_PROGRAMGRAPH.fields_by_name['features'].message_type = programl_dot_proto_dot_features__pb2._FEATURES
_PROGRAMGRAPHLIST.fields_by_name['context'].message_type = programl_dot_proto_dot_features__pb2._FEATURES
_PROGRAMGRAPHLIST.fields_by_name['graph'].message_type = _PROGRAMGRAPH
DESCRIPTOR.message_types_by_name['ProgramGraph'] = _PROGRAMGRAPH
DESCRIPTOR.message_types_by_name['ProgramGraphList'] = _PROGRAMGRAPHLIST

ProgramGraph = _reflection.GeneratedProtocolMessageType('ProgramGraph', (_message.Message,), dict(
  DESCRIPTOR = _PROGRAMGRAPH,
  __module__ = 'programl.proto.program_graph_pb2'
  # @@protoc_insertion_point(class_scope:programl.ProgramGraph)
  ))
_sym_db.RegisterMessage(ProgramGraph)

ProgramGraphList = _reflection.GeneratedProtocolMessageType('ProgramGraphList', (_message.Message,), dict(
  DESCRIPTOR = _PROGRAMGRAPHLIST,
  __module__ = 'programl.proto.program_graph_pb2'
  # @@protoc_insertion_point(class_scope:programl.ProgramGraphList)
  ))
_sym_db.RegisterMessage(ProgramGraphList)


DESCRIPTOR.has_options = True
DESCRIPTOR._options = _descriptor._ParseOptions(descriptor_pb2.FileOptions(), _b('\n\014com.programlB\021ProgramGraphProtoP\001Z\nprogramlpb'))
# @@protoc_insertion_point(module_scope)