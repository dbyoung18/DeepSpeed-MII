import argparse
import mii
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    TopKLogitsWarper,
    TemperatureLogitsWarper,
    BeamSearchScorer,
)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="google-t5/t5-base", help="model name or path.")
    parser.add_argument("--tp", type=int, default=1, help="Tensor-Parallel Size.")
    parser.add_argument(
        "--prompts", type=str, nargs="+", default=[
            "Once upon a time, there existed a little girl, who liked to have adventures. She wanted to go to places and meet new people, and have fun.",
            # "DeepSpeed is",
            # "Seattle is",
            # '<s>[INST] <<SYS>>\nYou are an AI assistant that helps people find information. User will you give you a question. Your task is to answer as faithfully as you can. While answering think step-bystep and justify your answer.\n<</SYS>>\n\nGiven the sentence "A woman with a fairy tattoo on her back is carrying a purse with a red floral print." can we conclude that "The woman\'s purse has red flowers on it."?\nOptions:\n- yes\n- it is not possible to tell\n- no Now, let\'s be accurate as possible. Some thinking first: [/INST]',
            # "translate English to German: How old are you?"
        ]
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--skip_decode", action="store_true", help="response tokens w/o tokenzier.decode")
    parser.add_argument("--num_beams", type=int, default=4, help="Number of output sequences. Set > 1 for beam search.")
    args = parser.parse_args()
    return args


def test_transformers_beam_search(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    encoder_input_ids = tokenizer(args.prompts, return_tensors="pt").input_ids

    # define decoder start token ids
    input_ids = torch.ones((args.num_beams, 1), device=model.device, dtype=torch.long)
    input_ids = input_ids * model.config.decoder_start_token_id

    # add encoder_outputs to model keyword arguments
    model_kwargs = {
        "encoder_outputs": model.get_encoder()(
            encoder_input_ids.repeat_interleave(args.num_beams, dim=0), return_dict=True
        )
    }

    # instantiate beam scorer
    beam_scorer = BeamSearchScorer(
        batch_size=1,
        num_beams=args.num_beams,
        device=model.device,
    )

    # instantiate logits processors
    logits_processor = LogitsProcessorList(
        [MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id)]
    )

    outputs = model._beam_search(
        input_ids, beam_scorer, logits_processor=logits_processor, **model_kwargs
    )

    result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(f"\ntransformers:\ngenerated_text:{result}\ngenerated_tokens:{outputs}\n", "-" * 80, "\n")


def test_mii_beam_search(args):
    pipe = mii.pipeline(
        args.model,
        tensor_parallel=args.tp,
        skip_decode=args.skip_decode,
    )

    inputs = args.prompts
    # inputs = [pipe.tokenizer.encode(input) for input in inputs]
    # print(f"inputs::{inputs}", flush=True)

    responses = pipe(
        inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        num_beams=4,
    )

    if pipe.is_rank_0:
        for i, r in enumerate(responses):
            print(f"response {i}\ngenerated_text:{r.generated_text}\ngenerated_tokens:{r.generated_tokens}\n", "-" * 80, "\n")


if __name__ == "__main__":
    args = parse_arguments()
    args.model = "google-t5/t5-base"
    test_transformers_beam_search(args)
    args.model = "/datadisk/dbyoung/llama2-7b"
    # test_mii_beam_search(args)
