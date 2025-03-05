#!/usr/bin/env python3
# Save as test_mimi_codec.py

import torch
import torchaudio
from tools.tokenizer.MimiCodec.mimi_tokenizer import MimiTokenizer

# Initialize the tokenizer on the first GPU
device = torch.device('cuda:0')
tokenizer = MimiTokenizer(device=device)

# Test with a sample file
test_file = '/home/david/mist_speech/gutenberg_espeak_dataset_clean/audio/gutenberg_train_000268.wav'

try:
    # Try loading with torchaudio (same as MimiTokenizer uses)
    print(f"Loading {test_file} with torchaudio...")
    wav, sr = torchaudio.load(test_file)
    print(f"  Shape: {wav.shape}, Sample rate: {sr}")
    
    # Test tokenizing
    print("Tokenizing...")
    codes = tokenizer.tokenize(wav, sr)
    print(f"  Codes shape: {codes.shape}")
    print("Successfully tokenized!")
    
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    
    # Try fixing the file
    print("\nTrying to fix the file...")
    try:
        # Load the WAV file, resample if needed, and save in a format torchaudio likes
        import soundfile as sf
        import numpy as np
        
        # Load with soundfile
        audio_data, sample_rate = sf.read(test_file)
        print(f"  Loaded with soundfile: shape {audio_data.shape}, sr {sample_rate}")
        
        # Convert to tensor
        audio_tensor = torch.tensor(audio_data).float()
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)  # Add channel dimension
        
        # Resample if needed
        if sample_rate != 24000:
            audio_tensor = torchaudio.transforms.Resample(sample_rate, 24000)(audio_tensor)
        
        # Save fixed version
        fixed_file = test_file + ".fixed.wav"
        torchaudio.save(fixed_file, audio_tensor, 24000, bits_per_sample=16)
        print(f"  Saved fixed file to {fixed_file}")
        
        # Try loading with torchaudio
        wav, sr = torchaudio.load(fixed_file)
        print(f"  Reloaded with torchaudio: shape {wav.shape}, sr {sr}")
        
        # Try tokenizing again
        codes = tokenizer.tokenize(wav, sr)
        print(f"  Codes shape: {codes.shape}")
        print("Successfully tokenized fixed file!")
        
    except Exception as e2:
        print(f"Error fixing: {type(e2).__name__}: {e2}")
