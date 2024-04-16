import argparse
import mii

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="/datadisk/share/llama2-7b", help="model name or path.")
parser.add_argument("--tp", type=int, default=2, help="Tensor-Parallel Size.")
parser.add_argument("--dp", type=int, default=1, help="Data-Parallel Size.")
args = parser.parse_args()

client = mii.serve(
    args.model,
    tensor_parallel=args.tp,
    replica_num=args.dp,
    deployment_name="mii-deployment",
    enable_restful_api=True,
    restful_api_port=28080,
)

print(f"Serving model {args.model} on {args.tp * args.dp} GPU(s).")
print(f"Run `python client.py --model {args.model}` to connect.")
print(f"Run `python terminate.py --model {args.model}` to terminate.")
