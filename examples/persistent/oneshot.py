import argparse
import mii

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="/datadisk/share/llama2-7b", help="model name or path.")
parser.add_argument("--tp", type=int, default=2, help="Tensor-Parallel Size.")
parser.add_argument("--dp", type=int, default=1, help="Data-Parallel Size.")
parser.add_argument("--name", type=str, default="mii-deployment", help="mii deployment name")
parser.add_argument(
    "--prompts", type=str, nargs="+", default=[
        "Once upon a time, there existed a little girl, who liked to have adventures. She wanted to go to places and meet new people, and have fun.",
        "DeepSpeed is",
        "Seattle is",
        '<s>[INST] <<SYS>>\nYou are an AI assistant that helps people find information. User will you give you a question. Your task is to answer as faithfully as you can. While answering think step-bystep and justify your answer.\n<</SYS>>\n\nGiven the sentence "A woman with a fairy tattoo on her back is carrying a purse with a red floral print." can we conclude that "The woman\'s purse has red flowers on it."?\nOptions:\n- yes\n- it is not possible to tell\n- no Now, let\'s be accurate as possible. Some thinking first: [/INST]',
    ]
)
parser.add_argument("--max-new-tokens", type=int, default=128)
parser.add_argument("--skip_decode", action="store_true", help="response tokens w/o tokenzier.decode")
args = parser.parse_args()

client = mii.serve(
    args.model,
    tensor_parallel=args.tp,
    replica_num=args.dp,
    deployment_name=args.name,
    enable_restful_api=True,
    restful_api_port=28080,
    skip_decode=args.skip_decode,
)

import time
time.sleep(60)
print(f"Serving model {args.model} with server {args.name} on {args.tp * args.dp} GPU(s).")

try:
    responses = client.generate(
        args.prompts,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,  # Greedy
        # return_full_text=True,
    )
except Exception as ex:
    print(f"Encounter exception: {ex}")
    import traceback

    trace = traceback.format_exc()
    print(trace)
else:
    for i, r in enumerate(responses):
        print(f"response {i}\ngenerated_text:{r.generated_text}\ngenerated_tokens:{r.generated_tokens}\n", "-" * 80, "\n")
finally:
    client.terminate_server()
    print(f"Terminated server {args.name}.")
