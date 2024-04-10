import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

IGNORE_TOKEN_ID=-100
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

class TrainingDataset(Dataset):
    def __init__(self, args, tokenizer, split):
        self.tokenizer = tokenizer
        self.args = args
        self.dataset = []
        for dataset in args.dataset.split(','):
            temp_dataset = []
            for line in open(f"{args.data_dir}/{dataset}_{split}.jsonl"):
                data = json.loads(line)
                data["dataset"] = dataset
                temp_dataset.append(data)
            if split == "train":
                if len(temp_dataset) > args.train_sample_limit:
                    temp_dataset = temp_dataset[:args.train_sample_limit]
            elif split == "valid":
                if len(temp_dataset) > args.valid_sample_limit:
                    temp_dataset = temp_dataset[:args.valid_sample_limit]
            self.dataset.extend(temp_dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        raw_data = self.dataset[idx]
        tokenized_data = self.process_data_chat(raw_data)
        ret = {
            'input_ids': tokenized_data['input_ids'],
            'attention_mask': tokenized_data['attention_mask'],
            'labels': tokenized_data['labels'],
            'dataset': raw_data['dataset'],
        }
        return ret

    def process_data_chat(self, data):
        data["conversation"][0]["content"] = B_SYS + data["system_prompt"] + E_SYS + data["conversation"][0]["content"]
        assert all([msg["role"] == "user" for msg in data["conversation"][::2]]) and all(
            [msg["role"] == "assistant" for msg in data["conversation"][1::2]]
        ), (
            "model only supports 'system', 'user' and 'assistant' roles, "
            "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
        )
        input_ids = []
        labels = []
        for question, answer in zip(data["conversation"][::2], data["conversation"][1::2]):
            question_tokens = [self.tokenizer.bos_token_id] + self.tokenizer.encode(
                f"{B_INST} {(question['content']).strip()} {E_INST}", add_special_tokens=False, verbose=False
            )
            input_ids.extend(question_tokens)
            labels.extend([0] * len(question_tokens))
            answer_tokens = self.tokenizer.encode(
                f"{(answer['content']).strip()} ", add_special_tokens=False, verbose=False
            ) + [self.tokenizer.eos_token_id]
            input_ids.extend(answer_tokens)
            labels.extend(answer_tokens)
        input_ids = input_ids[:self.args.max_length]
        labels = labels[:self.args.max_length]
        return dict(
            input_ids=torch.LongTensor(input_ids),
            attention_mask=torch.LongTensor(input_ids).ne(self.tokenizer.pad_token_id),
            labels=torch.LongTensor(labels)
        )
    
    def pad_seq(self, seq, length, value):
        return [value] * (length-len(seq)) + seq

    def collator(self, features):
        max_length = max([f["input_ids"].size(0) for f in features])
        # max_length = self.tokenizer.model_max_length
        input_ids = torch.stack([F.pad(f["input_ids"], (0, max_length-f["input_ids"].size(0)), value=self.tokenizer.pad_token_id) for f in features])
        attention_mask = torch.stack([F.pad(f["attention_mask"], (0, max_length-f["attention_mask"].size(0))) for f in features])
        return_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        if "labels" in features[0]:
            labels = torch.stack([F.pad(f["labels"], (0, max_length-f["labels"].size(0)), value=IGNORE_TOKEN_ID) for f in features]) 
            return_dict["labels"] = labels
        return_dict['dataset'] = [f['dataset'] for f in features]
        return return_dict

