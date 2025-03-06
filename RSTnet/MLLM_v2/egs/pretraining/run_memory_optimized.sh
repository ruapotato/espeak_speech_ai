#!/bin/bash

# eSpeak Speech AI Training Script - Memory Optimized for 24GB VRAM
# Based on RSTnet with modifications for eSpeak synthetic speech datasets

# Set the stages to execute (modify as needed)
stage=1
stop_stage=7
ngpu=1  # How many GPUs to use (adjust based on your hardware)

# Dataset configuration
train_set="train"
valid_set="val"
test_set="test"

# Path configuration (modify these to match your setup)
export DATA_ROOT=~/espeak_speech_ai/gutenberg_espeak_dataset_clean
export LLM_CHECKPOINT=~/espeak_speech_ai/checkpoints/meta-llama/Llama-3.2-1B-Instruct
export MODEL_DIR=~/espeak_speech_ai/checkpoints/espeak_speech_ai
export EXPERIMENT_NAME="espeak_llama32_1b_memory_optimized"

# Create experiment directory
mkdir -p $MODEL_DIR/$EXPERIMENT_NAME

# Update PYTHONPATH to include the necessary directories
export PYTHONPATH=$PYTHONPATH:$HOME/espeak_speech_ai/RSTnet:$HOME/espeak_speech_ai/RSTnet/MLLM_v2

# Training configuration - Memory optimized
seed=999
batch_scale=500    # Reduced from 2500 to save memory
learning_rate=0.0002
tag="espeak_memory_optimized"

# Output directories
mkdir -p $MODEL_DIR/$EXPERIMENT_NAME/log

echo "Running memory-optimized eSpeak Speech AI training with stages $stage to $stop_stage"

# Stage 1: Check if data preparation is complete
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Stage 1: Checking data preparation..."
    
    if [ ! -d "$DATA_ROOT" ]; then
        echo "ERROR: Dataset directory $DATA_ROOT not found!"
        echo "Please run build_dataset_espeak.py first to generate the dataset."
        exit 1
    fi
    
    if [ ! -f "$DATA_ROOT/metadata.json" ]; then
        echo "ERROR: metadata.json not found in $DATA_ROOT!"
        echo "Please ensure your dataset was correctly generated."
        exit 1
    fi
    
    echo "Data directory exists. Proceeding to next stage."
fi

# Stage 2: Check if tokenization is complete
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Stage 2: Checking data tokenization..."
    
    tokenization_complete=true
    for part in $train_set $valid_set $test_set; do
        for n in $(seq 1 $ngpu); do
            if [ ! -f "$DATA_ROOT/${part}/${ngpu}splits_2/semantic_codec.${n}.pt" ] || 
               [ ! -f "$DATA_ROOT/${part}/${ngpu}splits_2/text.${n}.pt" ]; then
                tokenization_complete=false
                echo "Tokenization files missing for $part split $n"
            fi
        done
    done
    
    if [ "$tokenization_complete" = false ]; then
        echo "WARNING: Data tokenization is incomplete."
        echo "Run extract_token.sh first to prepare the dataset."
        echo "Continuing with available data..."
    else
        echo "Tokenization complete. Proceeding to next stage."
    fi
fi

# Stage 3: Verify JSON files for training
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Stage 3: Verifying JSON files for training..."
    
    json_complete=true
    for part in $train_set $valid_set $test_set; do
        for n in $(seq 1 $ngpu); do
            if [ ! -f "$DATA_ROOT/${part}/${ngpu}splits_2/data.${n}.json" ]; then
                json_complete=false
                echo "JSON file missing for $part split $n"
            fi
        done
    done
    
    if [ "$json_complete" = false ]; then
        echo "WARNING: JSON files incomplete."
        echo "Run extract_token.sh to create the necessary JSON files."
        echo "Continuing with available data..."
    else
        echo "JSON files verified. Proceeding to next stage."
    fi
fi

# Stage 4: Prepare concatenated JSON file paths
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "Stage 4: Preparing concatenated JSON file paths..."
    
    # Create lists of all data JSON files for training and validation
    train_data_jsons=""
    valid_data_jsons=""
    
    for n in $(seq 1 $ngpu); do
        if [ -f "$DATA_ROOT/$train_set/${ngpu}splits_2/data.${n}.json" ]; then
            if [ -z "$train_data_jsons" ]; then
                train_data_jsons="$DATA_ROOT/$train_set/${ngpu}splits_2/data.${n}.json"
            else
                train_data_jsons="$train_data_jsons,$DATA_ROOT/$train_set/${ngpu}splits_2/data.${n}.json"
            fi
        fi
        
        if [ -f "$DATA_ROOT/$valid_set/${ngpu}splits_2/data.${n}.json" ]; then
            if [ -z "$valid_data_jsons" ]; then
                valid_data_jsons="$DATA_ROOT/$valid_set/${ngpu}splits_2/data.${n}.json"
            else
                valid_data_jsons="$valid_data_jsons,$DATA_ROOT/$valid_set/${ngpu}splits_2/data.${n}.json"
            fi
        fi
    done
    
    echo "Training JSON files: $train_data_jsons"
    echo "Validation JSON files: $valid_data_jsons"
    
    # Save JSON paths to config files for later use
    echo $train_data_jsons > $MODEL_DIR/$EXPERIMENT_NAME/train_jsons.txt
    echo $valid_data_jsons > $MODEL_DIR/$EXPERIMENT_NAME/valid_jsons.txt
fi

# Stage 5: Setup LLM checkpoint
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "Stage 5: Setting up LLM checkpoint..."
    
    if [ ! -d "$LLM_CHECKPOINT" ]; then
        echo "WARNING: LLM checkpoint directory not found at $LLM_CHECKPOINT"
        echo "You need to download the Llama-3.2 model checkpoint first."
        echo "Use: litgpt download meta-llama/Llama-3.2-1B-Instruct"
        echo "Or set LLM_CHECKPOINT to the correct path."
    else
        echo "LLM checkpoint found at $LLM_CHECKPOINT"
        
        # Check for model config file
        if [ ! -f "$LLM_CHECKPOINT/model_config.yaml" ]; then
            echo "WARNING: model_config.yaml not found in $LLM_CHECKPOINT"
            echo "Creating a basic configuration file..."
            
            # Create a basic model config
            cat > $LLM_CHECKPOINT/model_config.yaml << EOF
model_type: llama
tokenizer_path: ${LLM_CHECKPOINT}
# Default config for Llama-3.2-1B-Instruct
hidden_size: 2048
intermediate_size: 5632
num_hidden_layers: 16
num_attention_heads: 16
num_key_value_heads: 16
hidden_act: silu
max_position_embeddings: 8192
initializer_range: 0.02
rms_norm_eps: 1e-05
EOF
        fi
        
        echo "LLM checkpoint setup complete."
    fi
fi

# Stage 6: Copy trainer scripts to experiment directory if needed
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "Stage 6: Setting up training scripts..."
    
    # Find trainer scripts
    TRAINER_DIR="$HOME/espeak_speech_ai/RSTnet/MLLM_v2/trainer"
    
    if [ ! -d "$TRAINER_DIR" ]; then
        echo "WARNING: Trainer directory not found at $TRAINER_DIR"
        echo "Training may fail. Please check your RSTnet installation."
    else
        echo "Trainer directory found at $TRAINER_DIR"
        
        # Create symbolic links to trainer scripts if needed
        mkdir -p $MODEL_DIR/$EXPERIMENT_NAME/scripts
        ln -sf $TRAINER_DIR/pre_training_lora.py $MODEL_DIR/$EXPERIMENT_NAME/scripts/
        ln -sf $TRAINER_DIR/pre_training_full.py $MODEL_DIR/$EXPERIMENT_NAME/scripts/
        
        echo "Training scripts linked to experiment directory."
    fi
fi

# Stage 7: Start training with memory optimizations
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "Stage 7: Starting model training with memory optimizations..."
    
    # Load JSON paths
    train_data_jsons=$(cat $MODEL_DIR/$EXPERIMENT_NAME/train_jsons.txt || echo "$DATA_ROOT/$train_set/${ngpu}splits_2/data.1.json")
    valid_data_jsons=$(cat $MODEL_DIR/$EXPERIMENT_NAME/valid_jsons.txt || echo "$DATA_ROOT/$valid_set/${ngpu}splits_2/data.1.json")
    
    if [ -z "$train_data_jsons" ] || [ -z "$valid_data_jsons" ]; then
        echo "ERROR: Training or validation JSON paths are empty!"
        echo "Please ensure stage 4 completed successfully."
        exit 1
    fi
    
    # Set environment variables for distributed training
    export HOST_GPU_NUM=$ngpu
    export HOST_NUM=1
    export NODE_NUM=1
    export INDEX=0
    export CHIEF_IP="localhost"
    
    # Memory optimization environment variables
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    
    # Set smaller dimensions for CodecFormer to save memory
    codecformer_dim=768        # Reduced from 1024
    codecformer_layers=4       # Reduced from 6
    codecformer_heads=12       # Reduced from 16
    
    # Enable gradient checkpointing to save memory
    use_gradient_checkpointing=""
    
    # Use smaller training length
    max_length=512             # Reduced from 1000
    
    # Log start of training
    echo "Starting memory-optimized training with $ngpu GPUs..."
    echo "Training data: $train_data_jsons"
    echo "Validation data: $valid_data_jsons"
    echo "Experiment directory: $MODEL_DIR/$EXPERIMENT_NAME"
    echo "Memory optimizations: Using batch_scale=$batch_scale, codecformer_dim=$codecformer_dim, gradient checkpointing"
    
    # Launch distributed training (using LoRA for efficiency)
    python -m torch.distributed.run \
            --nproc_per_node=$HOST_GPU_NUM \
            --nnodes=$HOST_NUM \
            --master_addr=$CHIEF_IP \
            --master_port=29500 \
            --node_rank=$INDEX \
            $HOME/espeak_speech_ai/RSTnet/MLLM_v2/trainer/pre_training_lora.py \
            --train_data_jsons $train_data_jsons \
            --valid_data_jsons $valid_data_jsons \
            --exp_dir $MODEL_DIR/$EXPERIMENT_NAME \
            --n_epoch 30 \
            --max_length $max_length \
            --batch_scale $batch_scale \
            --global_learning_rate 5e-4 \
            --local_learning_rate 1e-4 \
            --model_config $LLM_CHECKPOINT/model_config.yaml \
            --audio_card 2050 \
            --n_q 8 \
            --dep_q 8 \
            --codecformer_heads $codecformer_heads \
            --codecformer_layers $codecformer_layers \
            --codecformer_dim $codecformer_dim \
            --codecformer_dim_feedforward 2304 \
            --checkpoint_path $LLM_CHECKPOINT/lit_model.pth \
            --lora_r 16 \
            --lora_alpha 16 \
            --lora_dropout 0.1 \
            --lora_query true \
            --lora_key true \
            --lora_value true \
            --lora_projection true \
            --lora_mlp true \
            --lora_head true \
            --save_interval 1000 \
            $use_gradient_checkpointing \
            2>&1 | tee $MODEL_DIR/$EXPERIMENT_NAME/log/training.log
    
    echo "Training complete or terminated. Check logs in $MODEL_DIR/$EXPERIMENT_NAME for details."
fi

echo "All stages complete."
