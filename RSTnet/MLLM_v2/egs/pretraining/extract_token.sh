#!/bin/bash

# RSTnet Data Processing Script
# This script sets up the data processing pipeline for the RSTnet speech-text foundation model

# Set key variables
export DATA_ROOT=~/espeak_speech_ai/gutenberg_espeak_dataset_clean
export LLM_CHECKPOINT=~/espeak_speech_ai/checkpoints/meta-llama/Llama-3.2-1B-Instruct
export EXPERIMENT_NAME="gutenberg_llama32_1b"
export NGPU=1  # Adjust based on available GPUs

# Update PYTHONPATH to include the necessary directories
export PYTHONPATH=$PYTHONPATH:$HOME/espeak_speech_ai/RSTnet:$HOME/espeak_speech_ai/RSTnet/MLLM_v2:$HOME/espeak_speech_ai/RSTnet/MLLM_v2/tools:$HOME/espeak_speech_ai/RSTnet/MLLM_v2/egs/pretraining
echo "PYTHONPATH set to: $PYTHONPATH"

# Define correct paths to the required scripts
OFFLINE_CODEC_PATH="$HOME/espeak_speech_ai/RSTnet/MLLM_v2/egs/pretraining/local/offline_codec_tokenization_fixed.py"
TEXT_TOKENIZATION_PATH="$HOME/espeak_speech_ai/RSTnet/MLLM_v2/egs/pretraining/data_scripts/text_tokenization_scp.py"
CREATE_DATA_JSON_PATH="$HOME/espeak_speech_ai/RSTnet/MLLM_v2/egs/pretraining/data_scripts/create_data_json.py"

# Check if the required scripts exist
if [ ! -f "$OFFLINE_CODEC_PATH" ]; then
    echo "ERROR: offline_codec_tokenization_fixed.py not found at $OFFLINE_CODEC_PATH"
    exit 1
fi

if [ ! -f "$TEXT_TOKENIZATION_PATH" ]; then
    echo "ERROR: text_tokenization_scp.py not found at $TEXT_TOKENIZATION_PATH"
    exit 1
fi

if [ ! -f "$CREATE_DATA_JSON_PATH" ]; then
    echo "ERROR: create_data_json.py not found at $CREATE_DATA_JSON_PATH"
    exit 1
fi

# Create a symbolic link for tools and utils if they don't exist in the expected location
ln -sf $HOME/espeak_speech_ai/RSTnet/MLLM_v2/tools $HOME/espeak_speech_ai/RSTnet/MLLM_v2/egs/pretraining/tools
ln -sf $HOME/espeak_speech_ai/RSTnet/MLLM_v2/utils $HOME/espeak_speech_ai/RSTnet/MLLM_v2/egs/pretraining/utils
echo "Created symbolic links for tools and utils"

# Step 1: Create necessary directories
mkdir -p $DATA_ROOT/train/${NGPU}splits_2
mkdir -p $DATA_ROOT/val/${NGPU}splits_2
mkdir -p $DATA_ROOT/test/${NGPU}splits_2

# Step 2: Create tar.scp files based on metadata.json and audio files
echo "Creating tar.scp files from dataset..."
python3 - << 'EOF'
import json
import os
import glob

# Load metadata
with open(os.path.expanduser('~/espeak_speech_ai/gutenberg_espeak_dataset_clean/metadata.json'), 'r') as f:
    metadata = json.load(f)

# Create tar.scp files for train, val, test
data_root = os.path.expanduser('~/espeak_speech_ai/gutenberg_espeak_dataset_clean')
for split in ['train', 'val', 'test']:
    # Get all files in this split
    split_data = metadata.get(split, [])
    
    # Create tar.scp file
    with open(f"{data_root}/{split}/tar.scp", 'w') as f:
        for item in split_data:
            audio_id = item['id']
            audio_path = os.path.join(data_root, 'audio', f"{audio_id}.wav")
            if os.path.exists(audio_path):
                f.write(f"{audio_id} {audio_path}\n")
            else:
                print(f"Warning: Audio file not found: {audio_path}")
    
    print(f"Created tar.scp for {split} with {len(split_data)} entries")
EOF

# Step 3: Split the data for multiple GPUs
for part in train val test; do
    if [ -f "$DATA_ROOT/${part}/tar.scp" ]; then
        echo "Splitting $part data for $NGPU GPUs"
        cat $DATA_ROOT/${part}/tar.scp | shuf > $DATA_ROOT/${part}/tar.scp.shuf
        
        split_scp=""
        for n in $(seq 1 $NGPU); do
            split_scp="$split_scp $DATA_ROOT/${part}/${NGPU}splits_2/tar.${n}.scp"
        done
        
        # Check if utils/split_scp.pl exists in RSTnet
        SPLIT_SCP_PATH=$(find $HOME/espeak_speech_ai/RSTnet/ -name "split_scp.pl" | head -1)
        if [ ! -z "$SPLIT_SCP_PATH" ]; then
            perl $SPLIT_SCP_PATH $DATA_ROOT/${part}/tar.scp.shuf $split_scp
        else
            echo "WARNING: split_scp.pl not found. Implementing simple split in Python."
            # Simple Python implementation of splitting
            python3 - << EOF
import math

part = "$part"
ngpu = $NGPU
data_root = "$DATA_ROOT"

with open(f"{data_root}/{part}/tar.scp.shuf", 'r') as f:
    lines = f.readlines()

total_lines = len(lines)
chunk_size = math.ceil(total_lines / ngpu)

for n in range(1, ngpu + 1):
    start_idx = (n - 1) * chunk_size
    end_idx = min(n * chunk_size, total_lines)
    
    with open(f"{data_root}/{part}/{ngpu}splits_2/tar.{n}.scp", 'w') as f_out:
        f_out.writelines(lines[start_idx:end_idx])
    
    print(f"Split {part} data: {end_idx - start_idx} items in GPU {n}")
EOF
        fi
    else
        echo "WARNING: $DATA_ROOT/${part}/tar.scp does not exist. Skipping $part split."
    fi
done

# Step 4: Create tar_info.scp files (text information)
for part in train val test; do
    mkdir -p $DATA_ROOT/${part}/${NGPU}splits_2/log
    
    for n in $(seq 1 $NGPU); do
        # Create tar_info.scp file from tar.scp
        if [ -f "$DATA_ROOT/${part}/${NGPU}splits_2/tar.${n}.scp" ]; then
            python3 - << EOF
import os

part = "$part"
n = $n
ngpu = $NGPU
data_root = "$DATA_ROOT"
metadata_path = f"{data_root}/metadata.json"

import json
with open(metadata_path, 'r') as f:
    metadata = json.load(f)

# Create a dictionary mapping ID to text
id_to_text = {}
for split_data in metadata.values():
    if isinstance(split_data, list):
        for item in split_data:
            if 'id' in item and 'text' in item:
                id_to_text[item['id']] = item['text']

# Read tar.scp and create tar_info.scp with text info
with open(f"{data_root}/{part}/{ngpu}splits_2/tar.{n}.scp", 'r') as f:
    lines = f.readlines()

with open(f"{data_root}/{part}/{ngpu}splits_2/tar_info.{n}.scp", 'w') as f_out:
    for line in lines:
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            audio_id = parts[0]
            if audio_id in id_to_text:
                f_out.write(f"{audio_id} {id_to_text[audio_id]}\n")
            else:
                print(f"Warning: No text found for audio ID: {audio_id}")

# Also create a text.scp file for tokenization
with open(f"{data_root}/{part}/{ngpu}splits_2/text.{n}.scp", 'w') as f_out:
    for line in lines:
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            audio_id = parts[0]
            if audio_id in id_to_text:
                f_out.write(f"{audio_id} {id_to_text[audio_id]}\n")
EOF
            echo "Created tar_info.${n}.scp for ${part}"
            echo "Created text and tar_info files for ${part} split ${n}"
        else
            echo "ERROR: $DATA_ROOT/${part}/${NGPU}splits_2/tar.${n}.scp not found"
        fi
    done
done

# Step 5: Extract audio codec tokens - MODIFIED TO USE MIMI TOKENIZER
echo "Extracting audio codec tokens..."
for part in train val test; do
    for n in $(seq 1 $NGPU); do
        if [ -f "$DATA_ROOT/${part}/${NGPU}splits_2/tar.${n}.scp" ] && [ -f "$DATA_ROOT/${part}/${NGPU}splits_2/tar_info.${n}.scp" ]; then
            echo "Processing audio codec tokenization for $part split $n using MimiCodec..."
            
            # Change to the correct directory
            cd $HOME/espeak_speech_ai/RSTnet/MLLM_v2/egs/pretraining
            
            # Use MimiCodec tokenizer instead of SSL tokenizer
            python3 $OFFLINE_CODEC_PATH \
                --tar-file $DATA_ROOT/${part}/${NGPU}splits_2/tar.${n}.scp \
                --tar-info $DATA_ROOT/${part}/${NGPU}splits_2/tar_info.${n}.scp \
                --output-text $DATA_ROOT/${part}/${NGPU}splits_2/text.${n}.scp \
                --output-utt2spk $DATA_ROOT/${part}/${NGPU}splits_2/utt2spk.${n}.scp \
                --output-file $DATA_ROOT/${part}/${NGPU}splits_2/semantic_codec.${n}.pt \
                --tokenizer mimi --rank $n
        else
            echo "WARNING: Required files for $part split $n not found. Skipping audio codec tokenization."
        fi
    done
done

# Step 6: Text tokenization with LLM checkpoint
echo "Processing text tokenization with LLM..."
for part in train val test; do
    for n in $(seq 1 $NGPU); do
        if [ -f "$DATA_ROOT/${part}/${NGPU}splits_2/text.${n}.scp" ]; then
            echo "Processing text tokenization for $part split $n..."
            
            # Change to the correct directory
            cd $HOME/espeak_speech_ai/RSTnet/MLLM_v2/egs/pretraining
            
            python3 $TEXT_TOKENIZATION_PATH \
                --rank $n \
                --input-file $DATA_ROOT/${part}/${NGPU}splits_2/text.${n}.scp \
                --checkpoint_dir $LLM_CHECKPOINT \
                --output-file $DATA_ROOT/${part}/${NGPU}splits_2/text.${n}.pt
        else
            echo "WARNING: $DATA_ROOT/${part}/${NGPU}splits_2/text.${n}.scp not found. Skipping text tokenization."
        fi
    done
done

# Step 7: Create data JSON files for training
echo "Creating data JSON files..."
for part in train val test; do
    for n in $(seq 0 $((NGPU-1))); do
        if [ -f "$DATA_ROOT/${part}/${NGPU}splits_2/text.$((n+1)).pt" ] && [ -f "$DATA_ROOT/${part}/${NGPU}splits_2/semantic_codec.$((n+1)).pt" ]; then
            echo "Creating data JSON for $part split $((n+1))..."
            
            # Change to the correct directory
            cd $HOME/espeak_speech_ai/RSTnet/MLLM_v2/egs/pretraining
            
            python3 $CREATE_DATA_JSON_PATH \
              --task setence_level_text_audio_interleaved \
              --out-json $DATA_ROOT/${part}/${NGPU}splits_2/data.$((n+1)).json \
              --text_seq $DATA_ROOT/${part}/${NGPU}splits_2/text.$((n+1)).pt \
              --audio_seq $DATA_ROOT/${part}/${NGPU}splits_2/semantic_codec.$((n+1)).pt
        else
            echo "WARNING: Required PT files for $part split $((n+1)) not found. Skipping JSON creation."
        fi
    done
done

echo "Data processing complete. You can now proceed with model training using the processed data."
echo "Check the logs for any warnings or errors that might need attention."
