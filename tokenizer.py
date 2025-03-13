#!/usr/bin/env python3
import torch
import numpy as np
import wave
import os
import json
import argparse
from tqdm import tqdm
from transformers import MimiModel, AutoFeatureExtractor
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Create audio token dataset using Mimi model for proper tokenization")
    parser.add_argument("--metadata_path", type=str, default="./complex_web_questions_dataset/metadata.json",
                       help="Path to metadata.json file")
    parser.add_argument("--audio_dir", type=str, default="./complex_web_questions_dataset/audio",
                       help="Directory containing audio files")
    parser.add_argument("--output_file", type=str, default="./cwq_audio_training_data.json",
                       help="Output JSON file for training data")
    parser.add_argument("--val_output_file", type=str, default="./cwq_audio_validation_data.json",
                       help="Output JSON file for validation data")
    parser.add_argument("--max_examples", type=int, default=None,
                       help="Maximum number of examples to process")
    parser.add_argument("--val_examples", type=int, default=100,
                       help="Number of validation examples to process")
    parser.add_argument("--max_tokens", type=int, default=5000,
                       help="Maximum token sequence length")
    parser.add_argument("--test_file", type=str, help="Process a single test file and exit")
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze audio lengths instead of generating tokens")
    
    args = parser.parse_args()
    
    # Load the Mimi model only once
    print("Loading Mimi model...")
    model = MimiModel.from_pretrained("kyutai/mimi")
    feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
    
    # Test with a single file if requested
    if args.test_file:
        print(f"Testing with audio file: {args.test_file}")
        process_single_test_file(args.test_file, model, feature_extractor)
        return
    
    print(f"Loading metadata from {args.metadata_path}")
    with open(args.metadata_path, "r") as f:
        metadata = json.load(f)
    
    # Just analyze audio lengths if requested
    if args.analyze:
        analyze_audio_lengths(metadata, args.audio_dir, model, feature_extractor)
        return
    
    # Process training data
    training_data = []
    print("Processing training examples...")
    
    # Use subset if max_examples is specified
    train_items = metadata["train"]
    if args.max_examples:
        train_items = train_items[:args.max_examples]
    
    for item in tqdm(train_items):
        # Get audio file path
        audio_path = os.path.join(args.audio_dir, f"{item['id']}.wav")
        
        if not os.path.exists(audio_path):
            print(f"Warning: Audio file not found: {audio_path}")
            continue
        
        try:
            # Process the audio file - only using codebook 0
            tokens = process_audio_file(audio_path, model, feature_extractor, args.max_tokens)
            
            # Join tokens with spaces - simple number format
            all_tokens_str = " ".join([str(token) for token in tokens])
            
            # Format for instruction fine-tuning - use the question as the input
            instruction_example = {
                "instruction": "Generate speech audio for the following text.",
                "input": item['text'],  # This is the question from Complex Web Questions
                "output": all_tokens_str
            }
            
            training_data.append(instruction_example)
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
    
    print(f"Successfully processed {len(training_data)} training examples")
    
    # Process validation data
    validation_data = []
    print("Processing validation examples...")
    
    # Use subset of validation data
    val_items = metadata["val"][:args.val_examples]
    
    for item in tqdm(val_items):
        # Get audio file path
        audio_path = os.path.join(args.audio_dir, f"{item['id']}.wav")
        
        if not os.path.exists(audio_path):
            print(f"Warning: Audio file not found: {audio_path}")
            continue
        
        try:
            # Process the audio file - only using codebook 0
            tokens = process_audio_file(audio_path, model, feature_extractor, args.max_tokens)
            
            # Join tokens with spaces - simple number format
            all_tokens_str = " ".join([str(token) for token in tokens])
            
            # Format for instruction fine-tuning - use the question as the input
            instruction_example = {
                "instruction": "Generate speech audio for the following text.",
                "input": item['text'],  # This is the question from Complex Web Questions
                "output": all_tokens_str
            }
            
            validation_data.append(instruction_example)
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
    
    print(f"Successfully processed {len(validation_data)} validation examples")
    
    # Calculate statistics about token lengths
    if training_data:
        token_lengths = [len(example["output"].split()) for example in training_data]
        avg_length = sum(token_lengths) / len(token_lengths)
        max_length = max(token_lengths)
        min_length = min(token_lengths)
        
        print(f"\nToken length statistics:")
        print(f"Average tokens per example: {avg_length:.2f}")
        print(f"Maximum tokens: {max_length}")
        print(f"Minimum tokens: {min_length}")
        
        # Show histogram of lengths
        length_bins = [0, 50, 100, 200, 300, 500, 1000]
        hist = [0] * (len(length_bins) + 1)
        for length in token_lengths:
            for i, bin_max in enumerate(length_bins):
                if length <= bin_max:
                    hist[i] += 1
                    break
            else:
                hist[-1] += 1
        
        print("\nLength distribution:")
        for i in range(len(length_bins)):
            if i == 0:
                print(f"0-{length_bins[i]}: {hist[i]} examples")
            else:
                print(f"{length_bins[i-1]}-{length_bins[i]}: {hist[i]} examples")
        print(f">{length_bins[-1]}: {hist[-1]} examples")
    
    # Save data
    print(f"Saving training data to {args.output_file}")
    with open(args.output_file, "w") as f:
        json.dump(training_data, f)
    
    print(f"Saving validation data to {args.val_output_file}")
    with open(args.val_output_file, "w") as f:
        json.dump(validation_data, f)
    
    # Sample output
    if training_data:
        print("\nSample training example:")
        sample = training_data[0]
        print(f"Text input: {sample['input']}")
        print(f"Number of tokens: {len(sample['output'].split(' '))}")
        print(f"First few tokens: {' '.join(sample['output'].split(' ')[:10])}...")
    
    print("Done!")

def load_audio_with_wave(audio_path):
    """Load audio using wave module as in test.py"""
    try:
        with wave.open(audio_path, 'rb') as wf:
            # Get audio info
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            
            # Read frames
            frames = wf.readframes(n_frames)
            
            # Convert to numpy
            if sample_width == 2:  # 16-bit
                audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            elif sample_width == 1:  # 8-bit
                audio_data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128) / 128.0
            elif sample_width == 4:  # 32-bit
                audio_data = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                raise ValueError(f"Unsupported sample width: {sample_width}")
            
            # Convert stereo to mono if needed
            if channels > 1:
                audio_data = audio_data.reshape(-1, channels)
                audio_data = np.mean(audio_data, axis=1)
            
            return audio_data, sample_rate
    
    except Exception as e:
        raise Exception(f"Error loading audio with wave module: {e}")

def process_audio_file(audio_path, model, feature_extractor, max_tokens=None):
    """Process a single audio file using the Mimi model and return tokens from codebook 0 only"""
    # Load audio
    audio_data, sample_rate = load_audio_with_wave(audio_path)
    
    # Ensure audio is at the right sample rate for the model
    target_sr = feature_extractor.sampling_rate
    if sample_rate != target_sr:
        # Simple resampling using linear interpolation
        old_time = np.linspace(0, len(audio_data)/sample_rate, len(audio_data))
        new_time = np.linspace(0, len(audio_data)/sample_rate, int(len(audio_data)*target_sr/sample_rate))
        audio_data = np.interp(new_time, old_time, audio_data)
    
    # Preprocess the audio
    inputs = feature_extractor(raw_audio=audio_data, sampling_rate=target_sr, return_tensors="pt")
    
    # Encode the audio to get tokens
    with torch.no_grad():
        encoded = model.encode(inputs["input_values"])
    
    # Get the tokens
    if hasattr(encoded, 'audio_codes'):
        audio_codes = encoded.audio_codes
    else:
        audio_codes = encoded
    
    # Extract tokens from just codebook 0
    tokens = audio_codes[0][0].cpu().numpy().tolist()
    
    # Limit tokens if needed
    if max_tokens and len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    
    return tokens

def analyze_audio_lengths(metadata, audio_dir, model, feature_extractor):
    """Analyze audio lengths when tokenized with Mimi model (using only codebook 0)"""
    token_counts = []  # For codebook 0
    durations = []
    
    print("Analyzing audio token lengths...")
    for item in tqdm(metadata["train"][:100]):  # Limit to 100 for speed
        audio_path = os.path.join(audio_dir, f"{item['id']}.wav")
        if os.path.exists(audio_path):
            try:
                # Get audio file info
                audio_data, sample_rate = load_audio_with_wave(audio_path)
                duration = len(audio_data) / sample_rate  # in seconds
                durations.append(duration)
                
                # Encode with Mimi
                inputs = feature_extractor(raw_audio=audio_data, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt")
                with torch.no_grad():
                    encoded = model.encode(inputs["input_values"])
                
                if hasattr(encoded, 'audio_codes'):
                    audio_codes = encoded.audio_codes
                else:
                    audio_codes = encoded
                
                # Only use codebook 0
                token_counts.append(audio_codes[0][0].shape[0])
                
            except Exception as e:
                print(f"Error analyzing {audio_path}: {e}")
    
    # Calculate statistics
    if token_counts:
        # For codebook 0
        avg_tokens = sum(token_counts) / len(token_counts)
        max_tokens = max(token_counts)
        min_tokens = min(token_counts)
        
        print(f"\nToken length statistics (codebook 0):")
        print(f"  Average tokens per example: {avg_tokens:.2f}")
        print(f"  Maximum tokens: {max_tokens}")
        print(f"  Minimum tokens: {min_tokens}")
        
        # Calculate tokens per second
        tokens_per_second = [t/d for t, d in zip(token_counts, durations)]
        avg_tps = sum(tokens_per_second) / len(tokens_per_second)
        
        print(f"\nToken rates:")
        print(f"  Average tokens per second: {avg_tps:.2f}")
        
        # Recommend max_tokens value
        p95_tokens = int(np.percentile(token_counts, 95))
        suggested_max_tokens = min(p95_tokens + 50, 1000)
        
        print(f"\nRecommended --max_tokens value: {suggested_max_tokens}")
        print(f"  This will fully represent approximately 95% of your audio files.")

def process_single_test_file(audio_path, model, feature_extractor):
    """Process a single test file and show detailed output"""
    try:
        # Load and process the audio
        audio_data, sample_rate = load_audio_with_wave(audio_path)
        print(f"Loaded audio: {len(audio_data)} samples, {sample_rate}Hz, {len(audio_data)/sample_rate:.2f} seconds")
        
        # Ensure audio is at the right sample rate for the model
        target_sr = feature_extractor.sampling_rate
        if sample_rate != target_sr:
            print(f"Resampling audio from {sample_rate}Hz to {target_sr}Hz")
            # Simple resampling using linear interpolation
            old_time = np.linspace(0, len(audio_data)/sample_rate, len(audio_data))
            new_time = np.linspace(0, len(audio_data)/sample_rate, int(len(audio_data)*target_sr/sample_rate))
            audio_data = np.interp(new_time, old_time, audio_data)
        
        # Preprocess the audio
        inputs = feature_extractor(raw_audio=audio_data, sampling_rate=target_sr, return_tensors="pt")
        
        # Encode the audio to get tokens
        print("Encoding audio with Mimi")
        with torch.no_grad():
            encoded = model.encode(inputs["input_values"])
        
        # Get the tokens
        if hasattr(encoded, 'audio_codes'):
            audio_codes = encoded.audio_codes
        else:
            audio_codes = encoded
        
        # Print token info
        print(f"Audio codes shape: {audio_codes.shape}")
        
        # Get tokens from codebook 0
        tokens = audio_codes[0][0].cpu().numpy().tolist()
        
        print(f"\nCodebook 0 approach:")
        print(f"Generated {len(tokens)} tokens")
        print(f"First 20 tokens: {' '.join(map(str, tokens[:20]))}")
        print(f"Last 20 tokens: {' '.join(map(str, tokens[-20:]))}")
        print(f"Token min: {min(tokens)}, max: {max(tokens)}")
        
        # Save tokens to a file
        output_file = "codebook_0_tokens.txt"
        with open(output_file, 'w') as f:
            f.write(' '.join(map(str, tokens)))
        print(f"Saved tokens to {output_file}")
        
        # Decode the tokens back to audio using only codebook 0
        print("Decoding tokens back to audio using codebook 0")
        with torch.no_grad():
            # Create new audio codes with just codebook 0
            new_audio_codes = torch.zeros_like(audio_codes)
            new_audio_codes[0, 0] = audio_codes[0, 0]
            
            # Decode back to audio
            decoded_audio = model.decode(new_audio_codes)[0].numpy()
        
        # Save the single-codebook decoded audio
        output_wav = "codebook_0_test.wav"
        with wave.open(output_wav, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 2 bytes for int16
            wf.setframerate(target_sr)
            decoded_int16 = (np.clip(decoded_audio, -1.0, 1.0) * 32767).astype(np.int16)
            wf.writeframes(decoded_int16.tobytes())
        
        print(f"Saved decoded audio to {output_wav}")
        
        # Calculate and display token rate
        token_rate = len(tokens) / (len(audio_data) / target_sr)
        print(f"Token rate: {token_rate:.2f} tokens per second")
        print("Test successful!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
