import mii
client = mii.client("mistralai/Mistral-7B-v0.1")
response = client.generate("Deepspeed is", max_new_tokens=128)
print(response)
