#!/bin/bash

# Check if text input was provided
if [ -z "$1" ]; then
    echo "Usage: $0 'LM input text' [lm_output_file.wav]"
    exit 1
fi

TEXT="$1"
OUTPUT_FILE="${2:-output.wav}"
MAX_TOKENS="600"  # Matching the max_tokens used in training
MODEL_OUTPUT="model_output.txt"

echo "Generating audio for text: $TEXT"

# Step 1: Generate tokens using the fine-tuned LLM
echo "Generating audio tokens (max $MAX_TOKENS)..."
litgpt generate ./audio_lm_model/final \
    --prompt "Generate speech audio for the following text.\n\nInput: $TEXT\n\nOutput:" \
    --max_new_tokens $MAX_TOKENS > $MODEL_OUTPUT

# Count the generated tokens (approximate)
TOKEN_COUNT=$(grep -o "[0-9]\+" $MODEL_OUTPUT | wc -l)
echo "Generated approximately $TOKEN_COUNT tokens"

# Step 2: Convert tokens to audio using the decoder
echo "Converting tokens to audio..."
python decoder.py --model_output $MODEL_OUTPUT \
    --output_file $OUTPUT_FILE \
    --apply_normalization \
    --verbosity 2

if [ -f "$OUTPUT_FILE" ]; then
    echo "Success! Audio saved to $OUTPUT_FILE"
    
    # Get audio duration
    if command -v ffprobe &> /dev/null; then
        DURATION=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$OUTPUT_FILE")
        echo "Audio duration: ${DURATION}s"
    fi
    
    echo "Play with: play $OUTPUT_FILE (if you have SoX installed) or use any audio player"
else
    echo "Error: Failed to create audio output file."
    
    # Show model output for debugging
    echo "Model output for debugging:"
    cat $MODEL_OUTPUT
fi

# Keep temporary files for debugging
echo "Model output saved to $MODEL_OUTPUT for reference"
