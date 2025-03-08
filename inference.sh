#!/bin/bash
set -e  # Exit on error

# Check if text input was provided
if [ -z "$1" ]; then
    echo "Usage: $0 'LM input text' [lm_output_file.wav]"
    exit 1
fi

TEXT="$1"
OUTPUT_FILE="${2:-output.wav}"
MAX_TOKENS="600"  # Matching the max_tokens used in training
MODEL_OUTPUT="model_output.txt"
TEMP_DIR="temp_tokens"

echo "Generating audio for text: $TEXT"

# Create temporary directory for token files
mkdir -p "$TEMP_DIR"

# Step 1: Generate tokens using the fine-tuned LLM
echo "Generating audio tokens (max $MAX_TOKENS)..."
litgpt generate ./audio_lm_model/final \
    --prompt "Generate speech audio for the following text.\n\nInput: $TEXT\n\nOutput:" \
    --max_new_tokens $MAX_TOKENS > $MODEL_OUTPUT

# Count the generated tokens (approximate)
TOKEN_COUNT=$(grep -o "[0-9]\+:[0-9]\+" $MODEL_OUTPUT | wc -l)
echo "Generated approximately $TOKEN_COUNT tokens"

# Step 2: Parse tokens into separate codebook files
echo "Parsing tokens into codebook files..."
python token_parser.py $MODEL_OUTPUT || {
    echo "Error: Failed to parse tokens"
    echo "Model output for debugging:"
    cat $MODEL_OUTPUT
    exit 1
}

# Step 3: Convert tokens to audio using the decoder
echo "Converting tokens to audio..."
python decoder.py \
    --model_output codebook_0_tokens.txt \
    --output_file "$OUTPUT_FILE" \
    --apply_normalization \
    --verbosity 2

if [ -f "$OUTPUT_FILE" ]; then
    echo "Success! Audio saved to $OUTPUT_FILE"
    
    # Get audio duration if ffprobe is available
    if command -v ffprobe &> /dev/null; then
        DURATION=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$OUTPUT_FILE")
        echo "Audio duration: ${DURATION}s"
    fi
    
    echo "Play with: play $OUTPUT_FILE (if you have SoX installed) or use any audio player"
    
    # Clean up temporary files
    rm -f codebook_*_tokens.txt
else
    echo "Error: Failed to create audio output file."
    echo "Model output for debugging:"
    cat $MODEL_OUTPUT
fi

# Keep model output for reference
echo "Model output saved to $MODEL_OUTPUT for reference"
