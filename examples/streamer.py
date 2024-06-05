from mii.constants import GenerationFinishReason
from typing import Dict, List


class TokenStreamer:
    def __init__(self) -> None:
        self.generated_text: Dict[int, str] = {}
        self.generated_tokens: Dict[int, List[int]] = {}
        self.infer_count = 0
        self.max_count = -1

    def put(self, responses: List) -> None:
        for response in responses:
            request_id = response.uid
            generated_tokens = response.generated_tokens
            if request_id not in self.generated_text.keys():
                self.generated_text[request_id] = response.generated_text
                self.generated_tokens[request_id] = list(generated_tokens)
            elif len(generated_tokens) != 0:
                self.generated_text[request_id] += response.generated_text
                self.generated_tokens[request_id] += generated_tokens
            if response.finish_reason != GenerationFinishReason.NONE:
                self.infer_count += 1
                continue

    @property
    def is_complete(self) -> bool:
        return self.infer_count == self.max_count
