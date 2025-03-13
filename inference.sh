#!/bin/bash
# inference.sh - Generate speech from text using the fine-tuned model

set -e  # Exit on error

# Check if input parameters are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 \"Your text prompt\" output.wav [temperature]"
    echo "Example: $0 \"Say 'Hello, world!'\" hello.wav 0.8"
    exit 1
fi

# Configuration variables
MODEL_DIR="./audio_lm_model/final"
TEXT_PROMPT="$1"
OUTPUT_WAV="$2"
TEMPERATURE="${3:-0.8}"  # Default temperature is 0.8 if not provided
TEMP_TOKENS="temp_tokens.txt"
MAX_NEW_TOKENS=2000

echo "Generating speech tokens for: \"$TEXT_PROMPT\""

# Build the full prompt with instruction format
FULL_PROMPT="Generate speech audio for the following text.

Input: $TEXT_PROMPT

Output:"

# Run the LitGPT generate command to get tokens
litgpt generate "$MODEL_DIR" \
    --prompt "$FULL_PROMPT" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" > "$TEMP_TOKENS"

# Extract just the numbers from the output (after "Output:")
grep -A 1 "Output:" "$TEMP_TOKENS" | tail -n 1 | sed 's/[^0-9 ]//g' > "zeroth_codebook.txt"

echo "Generated tokens. Converting to audio..."

# Run the inference pipeline to convert tokens to audio
python inference-pipeline.py \
    --model_path "./big_mapper_model/checkpoint_epoch_19.pt" \
    --input_file "zeroth_codebook.txt" \
    --output_file "$OUTPUT_WAV" \
    --use_gpu \
    --temperature 0.01

echo "Audio generation complete! Saved to $OUTPUT_WAV"

# Clean up temporary files
rm -f "$TEMP_TOKENS" "zeroth_codebook.txt"
