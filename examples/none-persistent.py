import argparse
import mii

from request import Request, RequestGroup
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="meta-llama/llama-2-7b", help="model name or path."
    )
    parser.add_argument(
        "--num_samples", type=int, default=-1, help="Number of infer samples."
    )
    parser.add_argument("--tp", type=int, default=1, help="Tensor-Parallel Size.")
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=[
            "Once upon a time, there existed a little girl, who liked to have adventures. She wanted to go to places and meet new people, and have fun.",
            "DeepSpeed is",
            "Seattle is",
            '<s>[INST] <<SYS>>\nYou are an AI assistant that helps people find information. User will you give you a question. Your task is to answer as faithfully as you can. While answering think step-bystep and justify your answer.\n<</SYS>>\n\nGiven the sentence "A woman with a fairy tattoo on her back is carrying a purse with a red floral print." can we conclude that "The woman\'s purse has red flowers on it."?\nOptions:\n- yes\n- it is not possible to tell\n- no Now, let\'s be accurate as possible. Some thinking first: [/INST]',
        ],
    )
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--min_new_tokens", type=int, default=1)
    parser.add_argument(
        "--torch_dist_port",
        type=int,
        default=29500,
        help="Torch distributed port to be used.",
    )
    parser.add_argument(
        "--zmq_port",
        type=int,
        default=25555,
        help="Port number to use for the ZMQ communication.",
    )
    parser.add_argument(
        "--skip_encode",
        action="store_true",
        default=True,
        help="tokenizer.encode prompts ahead before mii.generate",
    )
    parser.add_argument(
        "--skip_decode",
        action="store_true",
        default=True,
        help="response tokens w/o tokenzier.decode",
    )
    args = parser.parse_args()
    return args


def main(args):
    # 1. generate requests
    if args.num_samples == -1:
        prompts = args.prompts
        args.num_samples = len(prompts)
    else:
        prompts = [args.prompts[i % len(args.prompts)] for i in range(args.num_samples)]
    if args.skip_encode:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            model_max_length=args.max_new_tokens,
            padding_side="left",
            use_fast=False,
        )
        request = RequestGroup(
            request_ids=list(range(1, args.num_samples + 1)),
            prompts=prompts,
            input_tokens=[tokenizer.encode(prompt) for prompt in prompts],
        )
        print(f"ahead encode prompts to {request.input_tokens}")
    else:
        request = RequestGroup(
            request_ids=list(range(1, args.num_samples + 1)),
            prompts=prompts,
            input_tokens=None,
        )
    # 2. init mii::pipeline
    if args.tp == 1:
        pipe = mii.pipeline(
            args.model,
            torch_dist_port=args.torch_dist_port,
            zmq_port_number=args.zmq_port,
            skip_decode=args.skip_decode,
        )
    else:
        pipe = mii.pipeline(
            args.model,
            tensor_parallel=args.tp,
            torch_dist_port=args.torch_dist_port,
            zmq_port_number=args.zmq_port,
            skip_decode=args.skip_decode,
        )
    # 3. infer requests
    responses = pipe(
        request.input_tokens if args.skip_encode else request.prompts,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        do_sample=False,  # Greedy
    )
    # 4. print responses
    if pipe.is_rank_0:
        for i in range(args.num_samples):
            generated_tokens = responses[i].generated_tokens.tolist()
            if args.skip_decode:
                generated_text = tokenizer.decode(generated_tokens)
            else:
                generated_test = responses[i].generated_text
            print(
                f"response {i}\ngenerated_text:{generated_text}\ngenerated_tokens:{generated_tokens}\n",
                "-" * 80,
                "\n",
            )


if __name__ == "__main__":
    args = parse_args()
    main(args)
