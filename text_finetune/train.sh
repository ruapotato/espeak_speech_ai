#!/bin/bash

# Check if data exists
if [ ! -f "./cwq_audio_training_data.json" ]; then
    echo "Error: Training data not found. Run the tokenizer script first."
    exit 1
fi

# Run the fine-tuning process with memory-optimized parameters
litgpt finetune ./checkpoints/Qwen/Qwen2.5-0.5B-Instruct \
  --data JSON \
  --data.json_path ./cwq_audio_training_data.json \
  --data.val_split_fraction 0.05 \
  --out_dir ./audio_lm_model \
  --precision "bf16-mixed" \
  --devices 1 \
  --lora_r 4 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lora_query true \
  --lora_key false \
  --lora_value true \
  --lora_projection false \
  --lora_mlp false \
  --lora_head false \
  --train.epochs 15 \
  --train.micro_batch_size 1 \
  --train.global_batch_size 8 \
  --train.max_seq_length 4000 \
  --train.save_interval 100 \
  --train.log_interval 10 \
  --train.lr_warmup_steps 100 \
  --eval.interval 500 \
  --optimizer "{\"class_path\": \"torch.optim.AdamW\", \"init_args\": {\"lr\": 0.0002, \"weight_decay\": 0.01}}"
