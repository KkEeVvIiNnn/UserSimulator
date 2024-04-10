import os
import json
import pickle
import logging
import argparse

import utils
import prompt

USER_SEQ_THRESHOLD = 10
USER_USED = 1000
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

def prepare_genre_preference(user2seq, item2meta, train_file_path, valid_file_path):
    def construct_user_profile(seq_for_profile):
        ret = ""
        for item in seq_for_profile[-10:]:
            if item2meta[item]["tags"]:
                tags = ', '.join(item2meta[item]["tags"])
                ret += f"{item2meta[item]['title']}({tags})\n"
            else:
                ret += f"{item2meta[item]['title']}\n"
        return ret
    def construct_tag_response(tag_list):
        if len(tag_list) == 1:
            return f"I like {tag_list[0]} games."
        if len(tag_list) == 2:
            return f"I like {tag_list[0]} and {tag_list[1]} games."
        else:
            tag_string = ", ".join(tag_list[:-1]) + ", and " + tag_list[-1]
            return f"I like {tag_string} games."

    os.makedirs(os.path.dirname(train_file_path), exist_ok=True)
    train_fd = open(train_file_path, "w")
    valid_fd = open(valid_file_path, "w")
    user_cnt = 0
    for user in user2seq:
        if len(user2seq[user]) < USER_SEQ_THRESHOLD:
            continue
        seq_for_profile = user2seq[user][:-4]
        user_profile = construct_user_profile(seq_for_profile)
        system_prompt = prompt.system_prompt["steam"].format(
            user_profile = user_profile
        )
        for item in user2seq[user][-4:-1]:
            if len(item2meta[item]["tags"]) > 0:
                response = construct_tag_response(item2meta[item]["tags"])
                conversation = [
                    {
                        "role": "user",
                        "content": "What kind of game do you like?"
                    },
                    {
                        "role": "assistant",
                        "content": response
                    }
                ]
                data = {
                    "system_prompt": system_prompt,
                    "conversation": conversation
                }
                train_fd.write(json.dumps(data, ensure_ascii=False)+'\n')
        valid_item = user2seq[user][-1]
        if len(item2meta[valid_item]["tags"]) > 0:
            response = construct_tag_response(item2meta[valid_item]["tags"])
            conversation = [
                {
                    "role": "user",
                    "content": "What kind of game do you like?"
                },
                {
                    "role": "assistant",
                    "content": response
                }
            ]
            data = {
                "system_prompt": system_prompt,
                "conversation": conversation
            }
            valid_fd.write(json.dumps(data, ensure_ascii=False)+'\n')

        user_cnt += 1
        if user_cnt == USER_USED:
            break

def prepare_item_preference(user2seq, user2negative_samples, item2meta, train_file_path, valid_file_path):
    def construct_user_profile(seq_for_profile):
        ret = ""
        for item in seq_for_profile[-10:]:
            if item2meta[item]["tags"]:
                tags = ', '.join(item2meta[item]["tags"])
                ret += f"{item2meta[item]['title']}({tags})\n"
            else:
                ret += f"{item2meta[item]['title']}\n"
        return ret

    os.makedirs(os.path.dirname(train_file_path), exist_ok=True)
    train_fd = open(train_file_path, "w")
    valid_fd = open(valid_file_path, "w")
    user_cnt = 0
    for user in user2seq:
        if len(user2seq[user]) < USER_SEQ_THRESHOLD:
            continue
        seq_for_profile = user2seq[user][:-4]
        user_profile = construct_user_profile(seq_for_profile)
        system_prompt = prompt.system_prompt["steam"].format(
            user_profile = user_profile
        )
        for item in user2seq[user][-4:-1]:
            question = f"Would you like to play this game: {item2meta[item]['title']}?"
            conversation = [
                {
                    "role": "user",
                    "content": question
                },
                {
                    "role": "assistant",
                    "content": f"Yes, I would like to play {item2meta[item]['title']}."
                }
            ]
            data = {
                "system_prompt": system_prompt,
                "conversation": conversation
            }
            train_fd.write(json.dumps(data, ensure_ascii=False)+'\n')
        for item in user2negative_samples[user][:3]:
            question = f"Would you like to play this game: {item2meta[item]['title']}?"
            conversation = [
                {
                    "role": "user",
                    "content": question
                },
                {
                    "role": "assistant",
                    "content": f"No, I don't like to play {item2meta[item]['title']}."
                }
            ]
            data = {
                "system_prompt": system_prompt,
                "conversation": conversation
            }
            train_fd.write(json.dumps(data, ensure_ascii=False)+'\n')

        valid_item = user2seq[user][-1]
        question = f"Would you like to play this game: {item2meta[valid_item]['title']}?"
        conversation = [
            {
                "role": "user",
                "content": question
            },
            {
                "role": "assistant",
                "content": f"Yes, I would like to play {item2meta[valid_item]['title']}."
            }
        ]
        data = {
            "system_prompt": system_prompt,
            "conversation": conversation
        }
        valid_fd.write(json.dumps(data, ensure_ascii=False)+'\n')
        valid_item = user2negative_samples[user][-1]
        question = f"Would you like to play this game: {item2meta[valid_item]['title']}?"
        conversation = [
            {
                "role": "user",
                "content": question
            },
            {
                "role": "assistant",
                "content": f"No, I don't like to play {item2meta[valid_item]['title']}."
            }
        ]
        data = {
            "system_prompt": system_prompt,
            "conversation": conversation
        }
        valid_fd.write(json.dumps(data, ensure_ascii=False)+'\n')

        user_cnt += 1
        if user_cnt == USER_USED:
            break

def prepare_review_preference(user2seq, review_dict, item2meta, train_file_path, valid_file_path):
    def construct_user_profile(seq_for_profile):
        ret = ""
        for item in seq_for_profile[-10:]:
            if item2meta[item]["tags"]:
                tags = ', '.join(item2meta[item]["tags"])
                ret += f"{item2meta[item]['title']}({tags})\n"
            else:
                ret += f"{item2meta[item]['title']}\n"
        return ret

    os.makedirs(os.path.dirname(train_file_path), exist_ok=True)
    train_fd = open(train_file_path, "w")
    valid_fd = open(valid_file_path, "w")
    user_cnt = 0
    for user in user2seq:
        if len(user2seq[user]) < USER_SEQ_THRESHOLD:
            continue
        seq_for_profile = user2seq[user][:-4]
        user_profile = construct_user_profile(seq_for_profile)
        system_prompt = prompt.system_prompt["steam"].format(
            user_profile = user_profile
        )
        for item in [seq_for_profile[0], seq_for_profile[-1], seq_for_profile[len(seq_for_profile)//2]]:
            if (user, item) in review_dict:
                question = f"What do you think of the game you have played: {item2meta[item]['title']}"
                conversation = [
                    {
                        "role": "user",
                        "content": question
                    },
                    {
                        "role": "assistant",
                        "content": review_dict[(user, item)]
                    }
                ]
                data = {
                    "system_prompt": system_prompt,
                    "conversation": conversation
                }
                train_fd.write(json.dumps(data, ensure_ascii=False)+'\n')
        for item in user2seq[user][-4:-1]:
            if (user, item) in review_dict:
                question = f"What do you think of this new game: {item2meta[item]['title']}"
                conversation = [
                    {
                        "role": "user",
                        "content": question
                    },
                    {
                        "role": "assistant",
                        "content": review_dict[(user, item)]
                    }
                ]
                data = {
                    "system_prompt": system_prompt,
                    "conversation": conversation
                }
                train_fd.write(json.dumps(data, ensure_ascii=False)+'\n')

        valid_item = user2seq[user][-1]
        if (user, valid_item) in review_dict:
            question = f"What do you think of this new game: {item2meta[valid_item]['title']}"
            conversation = [
                {
                    "role": "user",
                    "content": question
                },
                {
                    "role": "assistant",
                    "content": review_dict[(user, item)]
                }
            ]
            data = {
                "system_prompt": system_prompt,
                "conversation": conversation
            }
            valid_fd.write(json.dumps(data, ensure_ascii=False)+'\n')
        user_cnt += 1
        if user_cnt == USER_USED:
            break

def main():
    args = parse_args()
    utils.set_random_seed(args.seed)

    logger.info("reading meta data")
    meta_data_file_path = f"{args.data_dir}/steam/metadata.jsonl"
    item2meta = {}
    itemid = 0
    for line in open(meta_data_file_path):
        meta = json.loads(line)
        if "popular_tags" in meta and isinstance(meta["popular_tags"], str):
            tags = meta["popular_tags"].split(',')
        elif "tags" in meta:
            tags = meta["tags"]
        elif "genres" in meta:
            tags = meta["genres"]
        else:
            tags = []
        meta = {
            "title": meta["title"],
            "tags": tags,
        }
        itemid += 1
        item2meta[itemid] = meta
    tag2cnt = {}
    for item in item2meta:
        for tag in item2meta[item]['tags']:
            if tag not in tag2cnt:
                tag2cnt[tag] = 0
            tag2cnt[tag] += 1
    tag2cnt = {tag: cnt for tag, cnt in tag2cnt.items() if cnt >= 10}
    for item in item2meta:
        tags = [tag for tag in item2meta[item]["tags"] if tag in tag2cnt]
        tags = sorted(tags, key=lambda tag: -tag2cnt[tag])
        item2meta[item]["tags"] = tags[:5]
    logger.info(len(tag2cnt))

    logger.info("reading sequential data")
    sequentail_data_file_path = f"{args.data_dir}/steam/sequential_data.txt"
    user2seq = {}
    for line in open(sequentail_data_file_path):
        user, items = line.split('\t', 1)
        user = int(user)
        items = [int(item) for item in items.split(' ')]
        user2seq[user] = items

    logger.info("reading review data")
    review_data_file_path = f"{args.data_dir}/steam/review.pkl"
    review_dict = pickle.load(open(review_data_file_path, "rb"))

    logger.info("reading negative sample")
    negative_sample_file_path = f"{args.data_dir}/steam/negative_samples_pop.txt"
    user2negative_samples = {}
    for line in open(negative_sample_file_path):
        user, negative_samples = line.split('\t', 1)
        user = int(user)
        negative_samples = [int(item) for item in negative_samples.split('\t')]
        user2negative_samples[user] = negative_samples

    logger.info("constructing genre preference data")
    genre_preference_train_file_path = f"{args.out_dir}/steam/genre_preference_train.jsonl"
    genre_preference_valid_file_path = f"{args.out_dir}/steam/genre_preference_valid.jsonl"
    prepare_genre_preference(user2seq, item2meta, genre_preference_train_file_path, genre_preference_valid_file_path)
    
    logger.info("constructing item preference data")
    item_preference_train_file_path = f"{args.out_dir}/steam/item_preference_train.jsonl"
    item_preference_valid_file_path = f"{args.out_dir}/steam/item_preference_valid.jsonl"
    prepare_item_preference(user2seq, user2negative_samples, item2meta, item_preference_train_file_path, item_preference_valid_file_path)

    logger.info("constructing review preference data")
    review_preference_train_file_path = f"{args.out_dir}/steam/review_preference_train.jsonl"
    review_preference_valid_file_path = f"{args.out_dir}/steam/review_preference_valid.jsonl"
    prepare_review_preference(user2seq, review_dict, item2meta, review_preference_train_file_path, review_preference_valid_file_path)

if __name__ == "__main__":
    main()