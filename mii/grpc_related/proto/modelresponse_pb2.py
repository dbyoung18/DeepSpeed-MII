# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: modelresponse.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()

DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n\x13modelresponse.proto\x12\rmodelresponse\"&\n\x13SingleStringRequest\x12\x0f\n\x07request\x18\x01 \x01(\t\"9\n\x11SingleStringReply\x12\x10\n\x08response\x18\x01 \x01(\t\x12\x12\n\ntime_taken\x18\x02 \x01(\x02\".\n\tQARequest\x12\x10\n\x08question\x18\x01 \x01(\t\x12\x0f\n\x07\x63ontext\x18\x02 \x01(\t\"\x8c\x01\n\x13\x43onversationRequest\x12\x0c\n\x04text\x18\x01 \x01(\t\x12\x1c\n\x0f\x63onversation_id\x18\x02 \x01(\x03H\x00\x88\x01\x01\x12\x18\n\x10past_user_inputs\x18\x03 \x03(\t\x12\x1b\n\x13generated_responses\x18\x04 \x03(\tB\x12\n\x10_conversation_id\"w\n\x11\x43onversationReply\x12\x17\n\x0f\x63onversation_id\x18\x01 \x01(\x03\x12\x18\n\x10past_user_inputs\x18\x02 \x03(\t\x12\x1b\n\x13generated_responses\x18\x03 \x03(\t\x12\x12\n\ntime_taken\x18\x04 \x01(\x02\x32\xbc\x04\n\rModelResponse\x12X\n\x0eGeneratorReply\x12\".modelresponse.SingleStringRequest\x1a .modelresponse.SingleStringReply\"\x00\x12]\n\x13\x43lassificationReply\x12\".modelresponse.SingleStringRequest\x1a .modelresponse.SingleStringReply\"\x00\x12V\n\x16QuestionAndAnswerReply\x12\x18.modelresponse.QARequest\x1a .modelresponse.SingleStringReply\"\x00\x12W\n\rFillMaskReply\x12\".modelresponse.SingleStringRequest\x1a .modelresponse.SingleStringReply\"\x00\x12\x62\n\x18TokenClassificationReply\x12\".modelresponse.SingleStringRequest\x1a .modelresponse.SingleStringReply\"\x00\x12]\n\x13\x43onversationalReply\x12\".modelresponse.ConversationRequest\x1a .modelresponse.ConversationReply\"\x00\x62\x06proto3'
)

_SINGLESTRINGREQUEST = DESCRIPTOR.message_types_by_name['SingleStringRequest']
_SINGLESTRINGREPLY = DESCRIPTOR.message_types_by_name['SingleStringReply']
_QAREQUEST = DESCRIPTOR.message_types_by_name['QARequest']
_CONVERSATIONREQUEST = DESCRIPTOR.message_types_by_name['ConversationRequest']
_CONVERSATIONREPLY = DESCRIPTOR.message_types_by_name['ConversationReply']
SingleStringRequest = _reflection.GeneratedProtocolMessageType(
    'SingleStringRequest',
    (_message.Message,
     ),
    {
        'DESCRIPTOR': _SINGLESTRINGREQUEST,
        '__module__': 'modelresponse_pb2'
        # @@protoc_insertion_point(class_scope:modelresponse.SingleStringRequest)
    })
_sym_db.RegisterMessage(SingleStringRequest)

SingleStringReply = _reflection.GeneratedProtocolMessageType(
    'SingleStringReply',
    (_message.Message,
     ),
    {
        'DESCRIPTOR': _SINGLESTRINGREPLY,
        '__module__': 'modelresponse_pb2'
        # @@protoc_insertion_point(class_scope:modelresponse.SingleStringReply)
    })
_sym_db.RegisterMessage(SingleStringReply)

QARequest = _reflection.GeneratedProtocolMessageType(
    'QARequest',
    (_message.Message,
     ),
    {
        'DESCRIPTOR': _QAREQUEST,
        '__module__': 'modelresponse_pb2'
        # @@protoc_insertion_point(class_scope:modelresponse.QARequest)
    })
_sym_db.RegisterMessage(QARequest)

ConversationRequest = _reflection.GeneratedProtocolMessageType(
    'ConversationRequest',
    (_message.Message,
     ),
    {
        'DESCRIPTOR': _CONVERSATIONREQUEST,
        '__module__': 'modelresponse_pb2'
        # @@protoc_insertion_point(class_scope:modelresponse.ConversationRequest)
    })
_sym_db.RegisterMessage(ConversationRequest)

ConversationReply = _reflection.GeneratedProtocolMessageType(
    'ConversationReply',
    (_message.Message,
     ),
    {
        'DESCRIPTOR': _CONVERSATIONREPLY,
        '__module__': 'modelresponse_pb2'
        # @@protoc_insertion_point(class_scope:modelresponse.ConversationReply)
    })
_sym_db.RegisterMessage(ConversationReply)

_MODELRESPONSE = DESCRIPTOR.services_by_name['ModelResponse']
if _descriptor._USE_C_DESCRIPTORS == False:

    DESCRIPTOR._options = None
    _SINGLESTRINGREQUEST._serialized_start = 38
    _SINGLESTRINGREQUEST._serialized_end = 76
    _SINGLESTRINGREPLY._serialized_start = 78
    _SINGLESTRINGREPLY._serialized_end = 135
    _QAREQUEST._serialized_start = 137
    _QAREQUEST._serialized_end = 183
    _CONVERSATIONREQUEST._serialized_start = 186
    _CONVERSATIONREQUEST._serialized_end = 326
    _CONVERSATIONREPLY._serialized_start = 328
    _CONVERSATIONREPLY._serialized_end = 447
    _MODELRESPONSE._serialized_start = 450
    _MODELRESPONSE._serialized_end = 1022
# @@protoc_insertion_point(module_scope)
