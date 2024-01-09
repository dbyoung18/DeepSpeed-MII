import mii
pipe = mii.pipeline("/mllnvme0/llama2-7b", tensor_parallel=2)

for i in range(3):
    response = pipe(["DeepSpeed is", "Seattle is"], max_new_tokens=128)
    print(response)
    print('-' * 30)
