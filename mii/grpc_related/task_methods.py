# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team


from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union

from google.protobuf.message import Message

from mii.batching.data_classes import Response
from mii.constants import TaskType
from mii.grpc_related.proto import modelresponse_pb2
from mii.utils import kwarg_dict_to_proto, unpack_proto_query_kwargs


def single_string_request_to_proto(self, request_dict, **query_kwargs):
    return modelresponse_pb2.SingleStringRequest(
        request=request_dict["query"],
        query_kwargs=kwarg_dict_to_proto(query_kwargs))


def single_string_response_to_proto(self, response, time_taken, model_time_taken):
    return modelresponse_pb2.SingleStringReply(response=f"{response}",
                                               time_taken=time_taken,
                                               model_time_taken=model_time_taken)


class TaskMethods(ABC):
    @property
    @abstractmethod
    def method(self):
        ...

    @abstractmethod
    def pack_request_to_proto(self, request, **query_kwargs):
        ...

    @abstractmethod
    def unpack_request_from_proto(self, proto_request):
        ...

    @abstractmethod
    def pack_response_to_proto(self, response):
        ...

    @abstractmethod
    def unpack_response_from_proto(self, proto_response):
        ...


class TextGenerationMethods(TaskMethods):
    @property
    def method(self):
        return "GeneratorReply"

    @property
    def method_stream_out(self):
        return "GeneratorReplyStream"

    def pack_request_to_proto(self,
                              prompts: List[str],
                              input_tokens: List[List[int]] = None,
                              uids: List[int] = None,
                              **query_kwargs: Dict[str, Any]) -> Message:
        proto_requests = []
        for i, prompt in enumerate(prompts):
            proto_requests.append(
            modelresponse_pb2.SingleStringRequest(
                uid=uids[i] if uids is not None else None,
                request=prompt,
                prompt_tokens=input_tokens[i] if input_tokens is not None else None,
                query_kwargs=kwarg_dict_to_proto(query_kwargs),
            ))
        return modelresponse_pb2.MultiStringRequest(request=proto_requests, )

    def unpack_request_from_proto(self,
                                  proto_request: Message) -> Union[
                                                                    Tuple[List[str], Dict[str, Any]],
                                                                    Tuple[List[List[int]], Dict[str, Any]]
                                                                    ]:
        requests = [r for r in proto_request.request]
        kwargs = unpack_proto_query_kwargs(requests[0].query_kwargs)
        if requests[0].prompt_tokens is None:
            inputs = [r.request for r in requests]
        else:
            inputs = [r.prompt_tokens for r in requests]
        if requests[0].uid is not None:
            uids = [r.uid for r in requests]
        return inputs, kwargs, uids

    def pack_response_to_proto(self, responses: List[Response]) -> Message:
        proto_responses = []
        for r in responses:
            if r.generated_tokens is None:
                generated_tokens = []
            elif type(r.generated_tokens) == list:  # stream:List[Tensor]
                if len(r.generated_tokens) != 0:
                    generated_tokens = [r.generated_tokens[0].tolist()]
                else:
                    generated_tokens = []
            else:  # Tensor
                generated_tokens = r.generated_tokens.tolist()

            proto_responses.append(
                modelresponse_pb2.SingleGenerationReply(
                    uid=r.uid,
                    response=r.generated_text,
                    finish_reason=str(r.finish_reason.value) if r.finish_reason is not None else "none",
                    prompt_tokens=r.prompt_length,
                    time_taken=-1,
                    model_time_taken=-1,
                    generated_length=r.generated_length,
                    generated_tokens=generated_tokens,
                ))

        return modelresponse_pb2.MultiGenerationReply(response=proto_responses, )

    def unpack_response_from_proto(self, response: Message) -> List[Response]:
        response_batch = []
        for r in response.response:
            response_batch.append(
                Response(
                    uid=r.uid,
                    generated_text=r.response,
                    prompt_length=r.prompt_tokens,
                    finish_reason=r.finish_reason,
                    generated_length=r.generated_length,
                    generated_tokens=r.generated_tokens,
                ))
        return response_batch


TASK_METHODS_DICT = {
    TaskType.TEXT_GENERATION: TextGenerationMethods(),
}
