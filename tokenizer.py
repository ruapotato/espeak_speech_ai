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
                       help="Maximum token sequence length per codebook")
    parser.add_argument("--test_file", type=str, help="Process a single test file and exit")
    parser.add_argument("--use_all_codebooks", action="store_true", default=True,
                       help="Use all 32 codebooks together (default)")
    parser.add_argument("--codebook", type=int, default=0,
                       help="Which specific codebook to use (0-31) if not using all")
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze audio lengths instead of generating tokens")
    parser.add_argument("--token_format", choices=["prefix", "separate"], default="prefix",
                        help="How to format tokens from multiple codebooks: 'prefix' adds codebook ID (e.g., '0:123'), 'separate' keeps codebooks in separate files")
    
    args = parser.parse_args()
    
    # Load the Mimi model only once
    print("Loading Mimi model...")
    model = MimiModel.from_pretrained("kyutai/mimi")
    feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
    
    # Test with a single file if requested
    if args.test_file:
        print(f"Testing with audio file: {args.test_file}")
        process_single_test_file(args.test_file, model, feature_extractor, args.codebook)
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
            # Process the audio file
            tokens = process_audio_file(audio_path, model, feature_extractor, 
                                       args.use_all_codebooks, args.codebook, args.max_tokens)
            
            # Join tokens with spaces
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
            # Process the audio file
            tokens = process_audio_file(audio_path, model, feature_extractor,
                                      args.use_all_codebooks, args.codebook, args.max_tokens)
            
            # Join tokens with spaces
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

def process_audio_file(audio_path, model, feature_extractor, use_all_codebooks=True, codebook=0, max_tokens=None):
    """Process a single audio file using the Mimi model and return tokens from all codebooks or a specific one"""
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
    
    if use_all_codebooks:
        # Get tokens from all codebooks, combining them with prefixes
        # Shape of audio_codes[0] is [32, sequence_length]
        all_codebooks = audio_codes[0].cpu().numpy()
        
        # Tokenize the sequence - we'll prefix each token with its codebook
        # For example: "0:123 1:456 2:789" meaning codebook0-token123, codebook1-token456, etc.
        tokens = []
        sequence_length = all_codebooks.shape[1]
        
        # Limit sequence length if needed
        if max_tokens and sequence_length > max_tokens:
            sequence_length = max_tokens
        
        # Create the token sequence with all codebooks
        for pos in range(sequence_length):
            for cb in range(all_codebooks.shape[0]):
                tokens.append(f"{cb}:{all_codebooks[cb, pos]}")
        
        return tokens
    else:
        # Extract tokens from just the specified codebook
        tokens = audio_codes[0][codebook].cpu().numpy().tolist()
        
        # Limit tokens if needed
        if max_tokens and len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
        
        return tokens

def analyze_audio_lengths(metadata, audio_dir, model, feature_extractor):
    """Analyze audio lengths when tokenized with Mimi model"""
    token_counts_single = []  # For single codebook
    token_counts_all = []     # For all codebooks
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
                
                all_tokens = audio_codes[0].cpu().numpy()
                
                # Count tokens for a single codebook
                token_counts_single.append(all_tokens.shape[1])
                
                # Count tokens for all codebooks combined
                token_counts_all.append(all_tokens.shape[0] * all_tokens.shape[1])
                
            except Exception as e:
                print(f"Error analyzing {audio_path}: {e}")
    
    # Calculate statistics
    if token_counts_single:
        # For single codebook
        avg_tokens_single = sum(token_counts_single) / len(token_counts_single)
        max_tokens_single = max(token_counts_single)
        min_tokens_single = min(token_counts_single)
        
        # For all codebooks
        avg_tokens_all = sum(token_counts_all) / len(token_counts_all)
        max_tokens_all = max(token_counts_all)
        min_tokens_all = min(token_counts_all)
        
        print(f"\nToken length statistics:")
        print(f"Single codebook:")
        print(f"  Average tokens per example: {avg_tokens_single:.2f}")
        print(f"  Maximum tokens: {max_tokens_single}")
        print(f"  Minimum tokens: {min_tokens_single}")
        
        print(f"\nAll codebooks combined:")
        print(f"  Average tokens per example: {avg_tokens_all:.2f}")
        print(f"  Maximum tokens: {max_tokens_all}")
        print(f"  Minimum tokens: {min_tokens_all}")
        
        # Calculate tokens per second
        tokens_per_second_single = [t/d for t, d in zip(token_counts_single, durations)]
        avg_tps_single = sum(tokens_per_second_single) / len(tokens_per_second_single)
        
        tokens_per_second_all = [t/d for t, d in zip(token_counts_all, durations)]
        avg_tps_all = sum(tokens_per_second_all) / len(tokens_per_second_all)
        
        print(f"\nToken rates:")
        print(f"  Average tokens per second (single codebook): {avg_tps_single:.2f}")
        print(f"  Average tokens per second (all codebooks): {avg_tps_all:.2f}")
        
        # Recommend max_tokens values
        p95_tokens_single = int(np.percentile(token_counts_single, 95))
        suggested_max_tokens_single = min(p95_tokens_single + 50, 1000)
        
        p95_tokens_all = int(np.percentile(token_counts_all, 95))
        suggested_max_tokens_all = min(p95_tokens_all + 50, 5000)
        
        print(f"\nRecommended --max_tokens values:")
        print(f"  For single codebook: {suggested_max_tokens_single}")
        print(f"  For all codebooks: {suggested_max_tokens_all}")
        print(f"  These will fully represent approximately 95% of your audio files.")

def process_single_test_file(audio_path, model, feature_extractor, codebook=0):
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
        
        # All token data
        all_tokens = audio_codes[0].cpu().numpy()  # Shape: [32, sequence_length]
        
        # Print token statistics for each codebook
        print("Token statistics per codebook:")
        for i in range(all_tokens.shape[0]):
            tokens = all_tokens[i]
            print(f"  Codebook {i}: {len(np.unique(tokens))} unique tokens")
        
        # Show results for all-codebooks approach
        all_codebook_tokens = []
        for pos in range(min(20, all_tokens.shape[1])):  # Just show first 20 positions
            for cb in range(all_tokens.shape[0]):
                all_codebook_tokens.append(f"{cb}:{all_tokens[cb, pos]}")
        
        print("\nAll codebooks approach (first 20 frames):")
        print(" ".join(all_codebook_tokens[:20*32]))  # Show only first 20 frames x 32 codebooks
        
        # Also show single codebook approach
        tokens = all_tokens[codebook].tolist()
        print(f"\nSingle codebook approach (codebook {codebook}):")
        print(f"Generated {len(tokens)} tokens")
        print(f"First 20 tokens: {tokens[:20]}")
        print(f"Last 20 tokens: {tokens[-20:]}")
        print(f"Token min: {min(tokens)}, max: {max(tokens)}")
        
        # Save tokens from all codebooks
        print("Saving tokens from all codebooks to separate files")
        for i in range(all_tokens.shape[0]):
            output_file = f"codebook_{i}_tokens.txt"
            with open(output_file, 'w') as f:
                f.write(' '.join(map(str, all_tokens[i].tolist())))
            print(f"Saved tokens for codebook {i} to {output_file}")
        
        # Save the complete token data
        print("Saving all tokens to all_tokens.npy")
        np.save('all_tokens.npy', all_tokens)
        
        # Now decode the tokens back to audio using all codebooks
        print("Decoding tokens back to audio using all codebooks")
        with torch.no_grad():
            # Use audio_codes directly since it already has all codebooks
            decoded_audio = model.decode(audio_codes)[0].numpy()
        
        # Save the decoded audio with all codebooks
        output_wav = "all_codebooks_test.wav"
        with wave.open(output_wav, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 2 bytes for int16
            wf.setframerate(target_sr)
            decoded_int16 = (np.clip(decoded_audio, -1.0, 1.0) * 32767).astype(np.int16)
            wf.writeframes(decoded_int16.tobytes())
        
        print(f"Saved fully decoded audio to {output_wav}")
        
        # Also test decoding with only one codebook
        print(f"Decoding tokens using only codebook {codebook}")
        with torch.no_grad():
            # Create new audio codes with just one codebook
            new_audio_codes = torch.zeros_like(audio_codes)
            new_audio_codes[0, codebook] = audio_codes[0, codebook]
            
            # Decode back to audio
            single_decoded_audio = model.decode(new_audio_codes)[0].numpy()
        
        # Save the single-codebook decoded audio
        single_output_wav = f"codebook_{codebook}_only_test.wav"
        with wave.open(single_output_wav, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 2 bytes for int16
            wf.setframerate(target_sr)
            single_decoded_int16 = (np.clip(single_decoded_audio, -1.0, 1.0) * 32767).astype(np.int16)
            wf.writeframes(single_decoded_int16.tobytes())
        
        print(f"Saved single-codebook decoded audio to {single_output_wav}")
        
        # Calculate and display token rates
        token_rate_single = len(tokens) / (len(audio_data) / target_sr)
        token_rate_all = len(tokens) * 32 / (len(audio_data) / target_sr)
        print(f"Token rate (single codebook): {token_rate_single:.2f} tokens per second")
        print(f"Token rate (all codebooks): {token_rate_all:.2f} tokens per second")
        print("Test successful!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
