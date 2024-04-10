import argparse
import os
import math
import numpy as np
from tqdm import tqdm
import wandb

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    AutoConfig,
    AutoTokenizer,
    SchedulerType,
    get_scheduler,
    GenerationConfig
)

import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

from utils import set_random_seed, get_optimizer_grouped_parameters, save_hf_format, save_zero_three_model, print_rank_0, to_device, get_all_reduce_mean
from ds_utils import get_train_ds_config

from dataset.train_dataset import TrainingDataset, IGNORE_TOKEN_ID
from model import LoRA_LLM



def parse_args():
    ## data arguments
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2024,
        help="A seed for reproducible training."
    )
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
        default=0,
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
        "--lora_learning_rate",
        type=float,
        default=5e-4,
        help="Initial LoRA learning rate (after the potential warmup period) to use."
    )

    # data arguments
    parser.add_argument('--data_dir',
        type=str,
        help="Path to the dataset directory.",
        required=True,
    )
    parser.add_argument('--dataset',
        type=str,
        help="json file names containing the training data.",
        required=True,
    )
    parser.add_argument('--train_sample_limit',
        type=int,
        help="max number of samples per dataset for training.",
        required=True,
    )
    parser.add_argument('--valid_sample_limit',
        type=int,
        help="max number of samples per dataset for validation.",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to store the model."
    )

    # training arguments
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.,
        help="Weight decay to use."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument(
        '--gradient_checkpointing',
        action='store_true',
        help='Enable HF gradient checkpointing for model.'
    )
    parser.add_argument(
        '--disable_dropout',
        action='store_true',
        help='Disable the dropout of the model.'
    )
    
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus"
    )
    # deepspeed features
    parser.add_argument(
        '--offload',
        action='store_true',
        help='Enable ZeRO Offload techniques.'
    )
    parser.add_argument(
        '--dtype',
        type=str,
        default='fp16',
        choices=['fp16', 'bf16'],
    )
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).'
    )
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    if args.local_rank == -1:
        device = torch.device(get_accelerator().device_name())
    else:
        get_accelerator().set_device(args.local_rank)
        device = torch.device(get_accelerator().device_name(), args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    ds_config = get_train_ds_config(
        offload=args.offload,
        dtype=args.dtype,
        stage=args.zero_stage,
    )
    ds_config['train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config['train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size() * args.gradient_accumulation_steps

    # If passed along, set the training seed now.
    set_random_seed(args.seed)

    torch.distributed.barrier()

    # load_hf_tokenizer will get the correct tokenizer and set padding tokens based on the model family
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        model_max_length=args.max_length,
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.unk_token
    llm_config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
    )
    llm_config.pad_token_id = tokenizer.pad_token_id
    llm_config.use_cache = False

    # Model creation:
    model = LoRA_LLM(args, llm_config)

    # DataLoaders creation:
    train_dataset = TrainingDataset(args, tokenizer, split="train")
    valid_dataset = TrainingDataset(args, tokenizer, split="valid")
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        valid_sampler = SequentialSampler(valid_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        valid_sampler = DistributedSampler(valid_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=train_dataset.collator,
                                  sampler=train_sampler,
                                  batch_size=args.per_device_train_batch_size)
    valid_dataloader = DataLoader(valid_dataset,
                                 collate_fn=valid_dataset.collator,
                                 sampler=valid_sampler,
                                 batch_size=args.per_device_eval_batch_size)
    
    # Optimizer creation:
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
    model, args.weight_decay, args.lora_learning_rate)

    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_trainable_params}/{total_params}")

    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    optimizer = AdamOptimizer(optimizer_grouped_parameters,
                            lr=args.learning_rate,
                            betas=(0.9, 0.95))
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if args.global_rank <= 0:
        wandb_name = f"{args.output_dir.split('/')[-1]}_{args.model_name_or_path.split('/')[-1]}"
        wandb.init(project="UserSimulator", config=args, name=wandb_name)
    # Train!
    def evaluation(model, eval_dataloader):
        model.eval()
        losses = 0
        valid_result = []
        if args.global_rank <= 0:
            pbar = tqdm(total=len(eval_dataloader), desc="validating", ncols=80)
        for step, batch in enumerate(eval_dataloader):
            batch = to_device(batch, device)
            with torch.no_grad():
                outputs = model(**batch)
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = batch["labels"][..., 1:].contiguous()
                padding_mask = (shift_labels != IGNORE_TOKEN_ID)

                # total_length
                losses = torch.nn.functional.cross_entropy(shift_logits.view(-1, outputs.logits.size(-1)), shift_labels.view(-1), reduction='none')  
                # batch_size * seq_length
                losses = losses.view(outputs.logits.size(0), -1)  
                # batch_size
                individual_losses = (losses * padding_mask).sum(dim=1) / padding_mask.sum(dim=1)
                for idx in range(outputs.logits.size(0)):
                    valid_result.append({"dataset": batch["dataset"][idx], "loss": individual_losses[idx].item()})
            if args.global_rank <= 0:
                pbar.update(1)
        if args.global_rank <= 0:
            pbar.close()
        result_list = [None for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather_object(result_list, valid_result)
        valid_result = sum([list(x) for x in list(zip(*result_list))], [])
        dataset2loss = {}
        for result in valid_result:
            if result["dataset"] not in dataset2loss:
                dataset2loss[result["dataset"]] = []
            dataset2loss[result["dataset"]].append(result["loss"])
        
        ret = {}
        for dataset in dataset2loss:
            losses = [loss for loss in dataset2loss[dataset] if not np.isnan(loss)]
            ret[dataset] = sum(losses) / max(1, len(losses))
        return ret
    
    def save_model(sub_dir):
        if args.global_rank <= 0:
            save_hf_format(model, tokenizer, args, sub_folder=sub_dir)
        if args.zero_stage == 3:
            # For zero stage 3, each gpu only has a part of the model, so we need a special save function
            save_zero_three_model(
                model,
                args.global_rank,
                os.path.join(args.output_dir, sub_dir),
                zero_stage=args.zero_stage
            )
            
    print_rank_0("***** Running training *****", args.global_rank)
    print_rank_0(f"***** Evaluating perplexity, Epoch {0}/{args.num_train_epochs} *****", args.global_rank)
    dataset2loss = evaluation(model, valid_dataloader)
    print_rank_0(f"loss: {dataset2loss}", args.global_rank)
    if args.global_rank <= 0:
        for dataset in dataset2loss:
            wandb.log({f"valid_{dataset}_loss": dataset2loss[dataset], "training/step": 0})

    total_step = 0
    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank
        )
        model.train()
        if args.global_rank <= 0:
            pbar = tqdm(total=len(train_dataloader), desc="training", ncols=80)
        for step, batch in enumerate(train_dataloader):
            total_step += 1
            batch = to_device(batch, device)
            outputs = model(**batch)
            loss = outputs.loss
            print_rank_0(f"Epoch: {epoch}, Step: {step}, training/step: {total_step}, loss = {loss}", args.global_rank)
            if args.global_rank <= 0:
                wandb.log({"epoch": epoch, "step": step, "training/step": total_step, "loss": loss})
            model.backward(loss)
            model.step()
            with torch.no_grad():
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = batch["labels"][..., 1:].contiguous()
                padding_mask = (shift_labels != IGNORE_TOKEN_ID)

                # total_length
                losses = torch.nn.functional.cross_entropy(shift_logits.view(-1, outputs.logits.size(-1)), shift_labels.view(-1), reduction='none')  
                # batch_size * seq_length
                losses = losses.view(outputs.logits.size(0), -1)  
                # batch_size
                individual_losses = (losses * padding_mask).sum(dim=1) / padding_mask.sum(dim=1)
                step_result = []
                for idx in range(outputs.logits.size(0)):
                    step_result.append({"dataset": batch["dataset"][idx], "loss": individual_losses[idx].item()})
                result_list = [None for _ in range(torch.distributed.get_world_size())]
                torch.distributed.all_gather_object(result_list, step_result)
                step_result = sum([list(x) for x in list(zip(*result_list))], [])
                dataset2loss = {}
                for result in step_result:
                    if result["dataset"] not in dataset2loss:
                        dataset2loss[result["dataset"]] = []
                    dataset2loss[result["dataset"]].append(result["loss"])
                    for dataset in dataset2loss:
                        losses = [loss for loss in dataset2loss[dataset] if not np.isnan(loss)]
                        avg_loss = sum(losses) / max(1, len(losses))
                        if args.global_rank <= 0 and len(losses):
                            wandb.log(
                                {
                                    f"train_{dataset}_loss": avg_loss,
                                    "training/step": total_step
                                }
                            )
            if args.global_rank <= 0:
                pbar.update(1)
        if args.global_rank <= 0:
            pbar.close()

        print_rank_0(f"***** Evaluating perplexity, Epoch {epoch+1}/{args.num_train_epochs} *****", args.global_rank)
        dataset2loss = evaluation(model, valid_dataloader)
        print_rank_0(f"loss: {dataset2loss}", args.global_rank)
        if args.global_rank <= 0:
            for dataset in dataset2loss:
                wandb.log({f"valid_{dataset}_loss": dataset2loss[dataset], "training/step": total_step})
        save_model(f"epoch_{epoch}")

if __name__ == "__main__":
    main()