deepspeed --num_gpus 8 \
    --num_nodes 2 \
    --master_addr 10.128.15.200 \
    --master_port 60000 \
    --hostfile hostfile \
    fastchat/train/train_mem.py \
    --model_name_or_path ~/ssd/llama-13b-hf-transformers-4.29 \
    --data_path ~/projects/datasets/validated.json \
    --output_dir ~/ssd/output_kd_model_13b \
    --num_train_epochs 3 \
    --per_device_train_batch_size 40 \
    --per_device_eval_batch_size 40 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_steps 100 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fp16 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --deepspeed deepspeed_config.json \
    # --report_to "wandb" \

