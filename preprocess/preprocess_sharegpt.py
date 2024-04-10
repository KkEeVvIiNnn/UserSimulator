import os
import json
import pickle
import logging
import argparse

import utils
import prompt

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s.%(msecs)03d] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare the finetune dataset for LLM-based user simulator."
    )
    ## data arguments
    parser.add_argument('--data_dir',
        help="Path to the dataset directory.",
        type=str,
        default="./data/raw"
    )
    parser.add_argument("--out_dir",
        help="Where to store the model.",
        type=str,
        default="./data",
    )
    parser.add_argument("--seed",
        help="A seed for reproducible training.",
        type=int,
        default=2024,
    )
    args = parser.parse_args()

    return args

def construct_data(data):
    system_prompt = prompt.system_prompt["sharegpt"]
    conversation = []
    for message in data["conversations"]:
        if message["from"] == "human":
            conversation.append({
                "role": "user",
                "content": message["value"]
            })
        elif message["from"] == "gpt":
            conversation.append({
                "role": "assistant",
                "content": message["value"]
            })
    compact_conversation = []
    for message in conversation:
        if len(compact_conversation) == 0 or message["role"] != compact_conversation[-1]["role"]:
            compact_conversation.append(message)
        else:
            compact_conversation[-1]["content"] += " " + message["content"]
    if len(compact_conversation) and compact_conversation[0]["role"] == "assistant":
        compact_conversation = compact_conversation[1:]
    return({
        "system_prompt": system_prompt,
        "conversation": compact_conversation
    })


def main():
    args = parse_args()
    utils.set_random_seed(args.seed)

    logger.info("constructing conversation data")
    raw_data_file_path = f"{args.data_dir}/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json"
    raw_data = json.load(open(raw_data_file_path))

    train_data_file_path = f"{args.out_dir}/sharegpt/conversation_train.jsonl"
    os.makedirs(os.path.dirname(train_data_file_path), exist_ok=True)
    train_fd = open(train_data_file_path, "w")
    
    for data in raw_data[:int(len(raw_data) * 0.9)]:
        data = construct_data(data)
        if len(data["conversation"]) > 1:
            train_fd.write(json.dumps(data, ensure_ascii=False)+'\n')

    valid_data_file_path = f"{args.out_dir}/sharegpt/conversation_valid.jsonl"
    os.makedirs(os.path.dirname(valid_data_file_path), exist_ok=True)
    valid_fd = open(valid_data_file_path, "w")
    
    for data in raw_data[int(len(raw_data) * 0.9):]:
        data = construct_data(data)
        if len(data["conversation"]) > 1:
            valid_fd.write(json.dumps(data, ensure_ascii=False)+'\n')

if __name__ == "__main__":
    main()