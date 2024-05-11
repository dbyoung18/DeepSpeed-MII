import argparse
import mii

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="/datadisk/share/llama2-7b", help="model name or path.")
parser.add_argument("--tp", type=int, default=2, help="Tensor-Parallel Size.")
parser.add_argument("--dp", type=int, default=1, help="Data-Parallel Size.")
parser.add_argument("--name", type=str, default="mii-deployment", help="mii deployment name")
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

print(f"Serving model {args.model} with server {args.name} on {args.tp * args.dp} GPU(s).")
print(f"Run `python client.py --name {args.name}` to connect.")
print(f"Run `python terminate.py --name {args.name}` to terminate.")
