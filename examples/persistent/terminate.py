import argparse
import mii

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="mii-deployment", help="mii deployment name")
args = parser.parse_args()

client = mii.client(args.name)
client.terminate_server()

print(f"Terminated server {args.name}.")