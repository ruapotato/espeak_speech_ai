#!/bin/bash
# Check if data exists
if [ ! -f "./cwq_audio_training_data.json" ]; then
    echo "Error: Training data not found. Run the tokenizer script first."
    exit 1
fi
# Run the fine-tuning process with memory optimizations for RTX 3090
litgpt finetune ./checkpoints/meta-llama/Llama-3.2-1B-Instruct \
  --data JSON \
  --data.json_path ./cwq_audio_training_data.json \
  --data.val_split_fraction 0.05 \
  --out_dir ./audio_lm_model \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --precision bf16-mixed \
  --train.epochs 5 \
  --train.micro_batch_size 1 \
  --train.max_seq_length 2048 \
  --train.save_interval 100 \
  --optimizer "{\"class_path\": \"torch.optim.AdamW\", \"init_args\": {\"lr\": 0.0002, \"weight_decay\": 0.01}}"
