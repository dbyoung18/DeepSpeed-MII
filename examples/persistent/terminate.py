import argparse
import mii

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="/datadisk/share/llama2-7b", help="model name or path.")
args = parser.parse_args()

client = mii.client(args.model)
client.terminate_server()

print(f"Terminated server for model {args.model}.")