import argparse
import re
import logging
import os
import torch
import json
from transformers import pipeline, set_seed
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

from model import LoRA_LLM

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s.%(msecs)03d] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def parse_args():
    parser = argparse.ArgumentParser()
    ## model arguments
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="The max length of the large language model.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Where do you want to store the pretrained models downloaded from huggingface.co",
    )

    ## LoRA method arguments
    parser.add_argument(
        "--lora_dim",
        type=int,
        default=8,
        help="If > 0, use LoRA for efficient training."
    )
    parser.add_argument(
        "--lora_module_name",
        type=str,
        default="layers.",
        help="The scope of LoRA."
    )
    parser.add_argument(
        '--only_optimize_lora',
        action='store_true',
        help='Only optimize the LoRA parameters.'
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum new tokens to generate per response",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="The device to load the model."
    )
    args = parser.parse_args()
    return args


def get_tokenizer_model(path, device):
    if os.path.exists(path):
        # Locally tokenizer loading has some issue, so we need to force download
        model_json = os.path.join(path, "config.json")
        if os.path.exists(model_json):
            model_json_file = json.load(open(model_json))
            model_name = model_json_file["_name_or_path"]
            tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                      fast_tokenizer=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(path, fast_tokenizer=True)

    tokenizer.pad_token = tokenizer.unk_token

    llm_config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
    )
    llm_config.pad_token_id = tokenizer.pad_token_id
    llm_config.use_cache = False

    # Model creation:
    model = LoRA_LLM(args, llm_config)
    model_state_dict = torch.load(f"{args.model_name_or_path}/pytorch_model.bin")
    model.load_state_dict(model_state_dict)
    return tokenizer, model.to(device)

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
BOS, EOS = "<s>", "</s>"

def get_user_input(conversation):
    if len(conversation) == 0:
        system_prompt = input("Enter system prompt (Default: A chat between...): ")
        if system_prompt == "":
            system_prompt = "A chat between a curious human and an artificial intelligence assistant. " \
                "The assistant gives helpful, detailed, and polite answers to the human's questions."
        conversation.append({
            "role": "system",
            "content": system_prompt
        })
    question = input("Enter input (type 'quit' to exit, 'clear' to clean memory): ")
    conversation.append({
        "role": "user",
        "content": question
    })
    
    return conversation, question == "quit", question == "clear"


def get_model_response(tokenizer, model, conversation, max_new_tokens, device):
    conversation = [
        {
            "role": conversation[1]["role"],
            "content": B_SYS + conversation[0]["content"] + E_SYS + conversation[1]["content"]
        }
    ] + conversation[2:]
    assert all([msg["role"] == "user" for msg in conversation[::2]]) and all(
        [msg["role"] == "assistant" for msg in conversation[1::2]]
    ), (
        "model only supports 'system', 'user' and 'assistant' roles, "
        "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
    )
    input_ids = []
    for question, answer in zip(conversation[::2], conversation[1::2]):
        tokens = [tokenizer.bos_token_id] + tokenizer.encode(
            f"{B_INST} {(question['content']).strip()} {E_INST} {(answer['content']).strip()} ", add_special_tokens=False, verbose=False
        ) + [tokenizer.eos_token_id]
        input_ids.extend(tokens)
    tokens = [tokenizer.bos_token_id] + tokenizer.encode(
        f"{B_INST} {(conversation[-1]['content']).strip()} {E_INST}", add_special_tokens=False, verbose=False
    )
    input_ids.extend(tokens)
    input_ids=torch.LongTensor(input_ids).to(device).unsqueeze(0)
    generate_ids = model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens)
    response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return response


def process_response(response):
    response = response.split("[/INST]")[-1].strip()
    return response


def main(args):
    logger.info("initializing tokenizer and model...")
    tokenizer, model = get_tokenizer_model(args.model_name_or_path, args.device)
    logger.info("done")
    set_seed(2024)

    conversation = []
    num_rounds = 0
    while True:
        num_rounds += 1
        conversation, quit, clear = get_user_input(conversation)

        if quit:
            break
        if clear:
            conversation, num_rounds = [], 0
            continue

        response = get_model_response(tokenizer, model, conversation,
                                      args.max_new_tokens, args.device)
        response = process_response(response)
        conversation.append({
            "role": "assistant",
            "content": response
        })

        print("-" * 30 + f" Round {num_rounds} " + "-" * 30)
        print(f"{response}")


if __name__ == "__main__":
    # Silence warnings about `max_new_tokens` and `max_length` being set
    logging.getLogger("transformers").setLevel(logging.ERROR)

    args = parse_args()
    main(args)

# Example:
"""
 Human: what is internet explorer?
 Assistant:
Internet Explorer is an internet browser developed by Microsoft. It is primarily used for browsing the web, but can also be used to run some applications. Internet Explorer is often considered the best and most popular internet browser currently available, though there are many other options available.

 Human: what is edge?
 Assistant:
 Edge is a newer version of the Microsoft internet browser, developed by Microsoft. It is focused on improving performance and security, and offers a more modern user interface. Edge is currently the most popular internet browser on the market, and is also used heavily by Microsoft employees.
"""