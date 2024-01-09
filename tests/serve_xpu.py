import mii
client = mii.serve("facebook/opt-6.7b", tensor_parallel=2)
response = client.generate(["Deepspeed is", "Seattle is"], max_new_tokens=128)
print(response)
