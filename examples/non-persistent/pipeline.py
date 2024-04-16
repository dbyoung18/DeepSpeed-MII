import argparse
import mii

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="/datadisk/share/llama2-7b", help="model name or path.")
parser.add_argument("--tp", type=int, default=2, help="Tensor-Parallel Size.")
parser.add_argument(
    "--prompts", type=str, nargs="+", default=[
        "Once upon a time, there existed a little girl, who liked to have adventures. She wanted to go to places and meet new people, and have fun.",
        # "DeepSpeed is",
        # "Seattle is",
        # "<s>[INST] <<SYS>>\nYou should describe the task and explain your answer. While answering a multiple choice question, first output the correct answer(s). Then explain why other answers are wrong. Think like you are answering to a five year old.\n<</SYS>>\n\nsure glad it isn't 97 until tomorrow again \n\nHow would the sentiment of this tweet be described?\nPossible answers: A. negative. B. positive. [/INST]", '<s>[INST] <<SYS>>\nYou should describe the task and explain your answer. While answering a multiple choice question, first output the correct answer(s). Then explain why other answers are wrong. Think like you are answering to a five year old.\n<</SYS>>\n\nWrite the next sentence in this paragraph:\n\nHow to treat an overactive bladder naturally \nControl your fluid intake throughout the day. \nTo avoid overwhelming your bladder, you should try to control how often you drink fluids throughout the day. Aim for reducing your fluid intake by 25 percent so your bladder is not too full.\n\nOPT:\n[I] One side effect of using an active bladder involves pressure in your bladder. This pressure can be carried out via straining, pushing out urine from your bladder.\n[II] Your doctor may also suggest a specific fluid reduction for you based on your condition. For example, you may try to drink a certain amount of water every hour or every other hour so you spread out your fluid intake.\n[III] Drink 800 to 227 milliliters (1 cup) of fluids daily. Drink this in a warm, soothing bath so that you do not overexert yourself.\n[IV] Limit your fluid intake to 500 mg a day (0.3 mg given orally). If you are at work or at school, try eliminating excessive or strenuous physical activity from your schedule. [/INST]',
    ]
)
parser.add_argument("--max-new-tokens", type=int, default=128)
args = parser.parse_args()

pipe = mii.pipeline(args.model, tensor_parallel=args.tp)

responses = pipe(
    args.prompts,
    max_new_tokens=args.max_new_tokens,
    do_sample=False,  # Greedy
    # return_full_text=True,
)

if pipe.is_rank_0:
    for r in responses:
        print(f"response:{r}\n", "-" * 80, "\n")
