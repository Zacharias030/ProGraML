# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: programl/proto/program_graph_options.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='programl/proto/program_graph_options.proto',
  package='programl',
  syntax='proto3',
  serialized_pb=_b('\n*programl/proto/program_graph_options.proto\x12\x08programl\"q\n\x13ProgramGraphOptions\x12\x19\n\x11instructions_only\x18\x01 \x01(\x08\x12\x1b\n\x13ignore_call_returns\x18\x02 \x01(\x08\x12\x11\n\topt_level\x18\x04 \x01(\x05\x12\x0f\n\x07ir_path\x18\n \x01(\tB6\n\x0c\x63om.programlB\x18ProgramGraphOptionsProtoP\x01Z\nprogramlpbb\x06proto3')
)
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_PROGRAMGRAPHOPTIONS = _descriptor.Descriptor(
  name='ProgramGraphOptions',
  full_name='programl.ProgramGraphOptions',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='instructions_only', full_name='programl.ProgramGraphOptions.instructions_only', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='ignore_call_returns', full_name='programl.ProgramGraphOptions.ignore_call_returns', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='opt_level', full_name='programl.ProgramGraphOptions.opt_level', index=2,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='ir_path', full_name='programl.ProgramGraphOptions.ir_path', index=3,
      number=10, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
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
  serialized_start=56,
  serialized_end=169,
)

DESCRIPTOR.message_types_by_name['ProgramGraphOptions'] = _PROGRAMGRAPHOPTIONS

ProgramGraphOptions = _reflection.GeneratedProtocolMessageType('ProgramGraphOptions', (_message.Message,), dict(
  DESCRIPTOR = _PROGRAMGRAPHOPTIONS,
  __module__ = 'programl.proto.program_graph_options_pb2'
  # @@protoc_insertion_point(class_scope:programl.ProgramGraphOptions)
  ))
_sym_db.RegisterMessage(ProgramGraphOptions)


DESCRIPTOR.has_options = True
DESCRIPTOR._options = _descriptor._ParseOptions(descriptor_pb2.FileOptions(), _b('\n\014com.programlB\030ProgramGraphOptionsProtoP\001Z\nprogramlpb'))
# @@protoc_insertion_point(module_scope)
