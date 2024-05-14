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
            generate_token = response.generated_tokens
            if len(generate_token) == 0:
                # print(f"finish,request_id:{request_id},generated_tokens,{self.generated_tokens[request_id]}", flush=True)
                self.infer_count += 1
                continue
            if request_id not in self.generated_text.keys():
                self.generated_text[request_id] = response.generated_text
                self.generated_tokens[request_id] = list(generate_token)
            else:
                self.generated_text[request_id] += response.generated_text
                self.generated_tokens[request_id] += generate_token

    @property
    def is_complete(self) -> bool:
        return self.infer_count == self.max_count
