import numpy as np
import queue
import threading
import torch

from typing import List, Optional, Union


class Request(object):
    def __init__(
        self,
        request_id: int = 0,
        dataset_idx: Optional[int] = 0,
        prompt: Optional[str] = None,
        input_tokens: Union[List[int], torch.Tensor] = None,
        attn_mask: Optional[List[int]] = None,
        input_len: Optional[int] = None,
        padded_len: Optional[int] = None,
        receipt_time: Optional[float] = 0,
        generated_tokens: Optional[List[int]] = None,
    ) -> None:
        self.request_id = request_id
        self.dataset_idx = dataset_idx
        self.prompt = prompt
        self.input_tokens = input_tokens
        self.attn_mask = attn_mask
        self.input_len = input_len
        self.padded_len = padded_len
        self.receipt_time = receipt_time
        self.generated_tokens = generated_tokens

    def __repr__(self) -> str:
        return (
            f"Request("
            f"request_id={self.request_id}"
            f"dataset_idx={self.dataset_idx},"
            f"prompt={self.prompt},"
            f"input_tokens={self.input_tokens},"
            f"attn_mask={self.attn_mask},"
            f"input_len={self.input_len},"
            f"receipt_time={self.receipt_time},"
            f"generated_tokens={self.generated_tokens})"
        )


class RequestGroup(object):
    def __init__(
        self,
        request_ids: List[int] = None,
        dataset_idx: Optional[List[int]] = None,
        prompts: Optional[List[str]] = None,
        input_tokens: Optional[Union[List[List[int]], torch.Tensor]] = None,
        attn_masks: Optional[Union[List[int], torch.Tensor]] = None,
        input_lens: Optional[List[int]] = None,
        padded_lens: Optional[List[int]] = None,
        receipt_time: Optional[List[float]] = 0,
        generated_tokens: Optional[List[np.ndarray]] = np.array([], dtype="int32"),
    ) -> None:
        self.request_ids = request_ids
        self.dataset_idx = dataset_idx
        self.prompts = prompts
        self.input_tokens = input_tokens
        self.attn_masks = attn_masks
        self.input_lens = input_lens
        self.padded_lens = padded_lens
        self.receipt_time = receipt_time
        self.batch_size = len(self.request_ids)
        self.generated_tokens = generated_tokens

    @property
    def output_lens(self):
        if len(self.generated_tokens) != 0:
            return [out.shape[-1] for out in self.generated_tokens]
        else:
            return [0] * self.batch_size

    def __repr__(self) -> str:
        return (
            f"RequestGroup("
            f"request_ids={self.request_ids},"
            f"dataset_idx={self.dataset_idx},"
            # f"prompts={self.prompts},"
            f"input_tokens={self.input_tokens},"
            f"attn_masks={self.attn_masks},"
            f"input_lens={self.input_lens},"
            f"padded_lens={self.padded_lens},"
            f"output_lens={self.output_lens},"
            f"receipt_time={self.receipt_time},"
            f"batch_size={self.batch_size},"
            f"generated_tokens={self.generated_tokens})"
        )


class RequestThread(threading.Thread):
    def __init__(self, queue):
        super().__init__()
        self.queue = queue
        self.setDaemon(True)
        self.start()

    def run(self):
        while True:
            func, args = self.queue.get()
            func(args)
            self.queue.task_done()

    def join(self, timeout=None):
        self.queue.join()


class RequestThreadPool:
    def __init__(self, num):
        self.num = num
        self.queue = queue.Queue()
        for _ in range(self.num):
            RequestThread(self.queue)

    def apply_async(self, func, args):
        self.queue.put((func, args))

    def join(self):
        self.queue.join()
