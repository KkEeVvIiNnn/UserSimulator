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
    id2title = data["movieMentions"]
    if not isinstance(id2title, dict):
        id2title = {}
    id2title = sorted(id2title.items(), key=lambda x: -len(x[0]))
    seeker_id = data["initiatorWorkerId"]
    recommender_id = data["respondentWorkerId"]
    system_prompt = prompt.system_prompt["redail"]
    def fill_text(text):
        for id, title in id2title:
            if not isinstance(title, str):
                title = "No Title"
            text = text.replace(f"@{id}", title)
        return text
    conversation = []
    for message in data["messages"]:
        if message["senderWorkerId"] == seeker_id:
            conversation.append({
                "role": "assistant",
                "content": fill_text(message["text"])
            })
        elif message["senderWorkerId"] == recommender_id:
            conversation.append({
                "role": "user",
                "content": fill_text(message["text"])
            })
    if conversation[0]["role"] == "assistant":
        conversation = [{"role": "user", "content": "Hi!"}] + conversation
    compact_conversation = []
    for message in conversation:
        if len(compact_conversation) == 0 or message["role"] != compact_conversation[-1]["role"]:
            compact_conversation.append(message)
        else:
            compact_conversation[-1]["content"] += " " + message["content"]
    return {
        "system_prompt": system_prompt,
        "conversation": compact_conversation
    }
def main():
    args = parse_args()
    utils.set_random_seed(args.seed)

    logger.info("constructing conversation data")
    raw_train_data_file_path = f"{args.data_dir}/redail/train_data.jsonl"
    train_data_file_path = f"{args.out_dir}/redail/conversation_train.jsonl"
    os.makedirs(os.path.dirname(train_data_file_path), exist_ok=True)
    train_fd = open(train_data_file_path, "w")
    for line in open(raw_train_data_file_path):
        data = json.loads(line)
        if len(data["messages"]) >= 3:
            data = construct_data(data)
            train_fd.write(json.dumps(data, ensure_ascii=False)+'\n')

    raw_valid_data_file_path = f"{args.data_dir}/redail/test_data.jsonl"
    valid_data_file_path = f"{args.out_dir}/redail/conversation_valid.jsonl"
    valid_fd = open(valid_data_file_path, "w")
    for line in open(raw_valid_data_file_path):
        data = json.loads(line)
        if len(data["messages"]) >= 3:
            data = construct_data(data)
            valid_fd.write(json.dumps(data, ensure_ascii=False)+'\n')

if __name__ == "__main__":
    main()