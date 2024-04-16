import argparse
import mii

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="/datadisk/share/llama2-7b", help="model name or path.")
parser.add_argument(
    "--prompts", type=str, nargs="+", default=[
        "Once upon a time, there existed a little girl, who liked to have adventures. She wanted to go to places and meet new people, and have fun.",
        "DeepSpeed is",
        "Seattle is",
    ]
)
parser.add_argument("--max-new-tokens", type=int, default=128)
args = parser.parse_args()

client = mii.client(args.model)
responses = client.generate(
    args.prompts,
    max_new_tokens=args.max_new_tokens,
    do_sample=False,  # Greedy
    # return_full_text=True,
)

for r in responses:
    print(r, "\n", "-" * 80, "\n")
