#!/usr/bin/env python3
"""
Test script for Mimi using all 32 codebooks.
Loads a specific audio file, converts to tokens from all codebooks, then back to audio.
"""

import torch
import numpy as np
import wave
from transformers import MimiModel, AutoFeatureExtractor

def main():
    # Fixed audio path
    audio_path = "./complex_web_questions_dataset/audio/nq_train_000111.wav"
    print(f"Testing with audio file: {audio_path}")
    
    # Load the audio
    try:
        # Use wave module for reliable loading
        with wave.open(audio_path, 'rb') as wf:
            # Get audio info
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            
            print(f"Audio info: {channels} channels, {sample_width} bytes/sample, {sample_rate} Hz, {n_frames} frames")
            
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
                print(f"Unsupported sample width: {sample_width}")
                return
            
            # Convert stereo to mono if needed
            if channels > 1:
                audio_data = audio_data.reshape(-1, channels)
                audio_data = np.mean(audio_data, axis=1)
    
    except Exception as e:
        print(f"Error loading audio: {e}")
        return
    
    # Load Mimi model
    print("Loading Mimi model")
    model = MimiModel.from_pretrained("kyutai/mimi")
    feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
    
    # Ensure audio is at the right sample rate
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
    
    # Save the complete token data
    print("Saving all tokens to all_tokens.npy")
    np.save('all_tokens.npy', all_tokens)
    
    # Save tokens from all codebooks as space-separated strings
    print("Saving tokens to separate text files")
    for i in range(all_tokens.shape[0]):
        tokens = all_tokens[i]
        with open(f'codebook_{i}_tokens.txt', 'w') as f:
            f.write(' '.join(map(str, tokens)))
    
    # Now decode the tokens back to audio using all codebooks
    print("Decoding tokens back to audio (using all 32 codebooks)")
    
    # We can use the audio_codes directly since it already has all codebooks
    with torch.no_grad():
        decoded_audio = model.decode(audio_codes)[0].numpy()
    
    # Save the decoded audio
    output_file = "./test.wav"
    
    # Save with wave module for reliability
    try:
        # Convert to int16
        decoded_int16 = (np.clip(decoded_audio, -1.0, 1.0) * 32767).astype(np.int16)
        
        with wave.open(output_file, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 2 bytes for int16
            wf.setframerate(target_sr)
            wf.writeframes(decoded_int16.tobytes())
        
        print(f"Saved fully decoded audio to {output_file}")
    except Exception as e:
        print(f"Error saving audio: {e}")
    
    print("Done!")

if __name__ == "__main__":
    main()
