set -x #echo on
dataset="steam/genre_preference,steam/item_preference,steam/review_preference,redail/conversation,sharegpt/conversation"
task_name="first_try"
deepspeed --include localhost:1,2 --master_port=29499 ./src/train.py \
    --model_name_or_path /data00/wei_xu/LLMs/llama-2-7b-chat-hf \
    --max_length 2048 \
    --cache_dir /data00/wei_xu/.cache \
    --lora_dim 8 \
    --only_optimize_lora \
    --data_dir ./data \
    --dataset ${dataset} \
    --train_sample_limit 10000 \
    --valid_sample_limit 1000 \
    --output_dir ./output/${task_name} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --learning_rate 1e-5 \
    --weight_decay 0 \
    --gradient_accumulation_steps 1 \
    --warmup_steps 50 \
    --gradient_checkpointing \
    --zero_stage 3 \
    --deepspeed > ${task_name}.log