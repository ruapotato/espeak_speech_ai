for file in /home/david/mist_speech/gutenberg_espeak_dataset_clean/audio/gutenberg_train_*.wav; do
  echo "Processing $file"
  # Create a temporary file
  temp_file="${file}.temp.wav"
  
  # Re-encode with strict parameters for MimiCodec
  ffmpeg -y -v error -i "$file" -ar 24000 -ac 1 -c:a pcm_s16le -f wav -rf64 never -bitexact "$temp_file"
  
  # Check if ffmpeg succeeded
  if [ $? -eq 0 ]; then
    # Replace the original file
    mv "$temp_file" "$file"
    echo "Fixed $file"
  else
    echo "Failed to fix $file"
    # Clean up the temp file if it exists
    rm -f "$temp_file"
  fi
done
