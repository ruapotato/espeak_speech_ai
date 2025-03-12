#!/usr/bin/env python3
"""
Codebook Dataset Generator

This script extracts codebooks from audio files and creates a dataset
for training a neural network that can predict the other codebooks from the 0th codebook.

Usage:
python codebook-dataset-generator.py --input_dir ./gutenberg_espeak_dataset_clean/audio --output_dir ./codebook_dataset
"""

import torch
import numpy as np
import wave
import os
import json
import argparse
from tqdm import tqdm
from transformers import MimiModel, AutoFeatureExtractor
from pathlib import Path
import logging

def setup_logging(verbosity=2):
    """Set up logging with specified verbosity level"""
    log_levels = {
        0: logging.ERROR,
        1: logging.WARNING,
        2: logging.INFO,
        3: logging.DEBUG
    }
    level = log_levels.get(verbosity, logging.INFO)
    
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=level
    )
    return logging.getLogger("codebook-dataset-generator")

def load_audio(audio_path):
    """Load audio using wave module"""
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
        raise Exception(f"Error loading audio: {e}")

def extract_codebooks(audio_data, sample_rate, model, feature_extractor):
    """Extract codebooks from audio data using Mimi model"""
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
    
    # Get codebooks
    all_codebooks = audio_codes[0].cpu().numpy()
    
    # Find number of codebooks actually available
    num_codebooks = all_codebooks.shape[0]
    
    return all_codebooks, num_codebooks

def process_audio_directory(input_dir, output_dir, max_files=None, max_seq_len=2000):
    """Process all audio files in a directory and create dataset"""
    logger = setup_logging()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Mimi model
    logger.info("Loading Mimi model")
    model = MimiModel.from_pretrained("kyutai/mimi")
    feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
    
    # Get all WAV files
    wav_files = list(Path(input_dir).glob("**/*.wav"))
    logger.info(f"Found {len(wav_files)} WAV files")
    
    # Limit number of files if specified
    if max_files and len(wav_files) > max_files:
        wav_files = wav_files[:max_files]
        logger.info(f"Using {len(wav_files)} files")
    
    # Initialize dataset
    dataset = []
    
    # Process each audio file
    for i, wav_file in enumerate(tqdm(wav_files, desc="Processing audio files")):
        try:
            # Load audio
            audio_data, sample_rate = load_audio(str(wav_file))
            
            # Extract codebooks
            all_codebooks, num_codebooks = extract_codebooks(
                audio_data, sample_rate, model, feature_extractor
            )
            
            # Limit sequence length if needed
            seq_len = min(all_codebooks.shape[1], max_seq_len)
            all_codebooks = all_codebooks[:, :seq_len]
            
            # Get zeroth codebook
            zeroth_codebook = all_codebooks[0].tolist()
            
            # Get other codebooks
            other_codebooks = []
            for j in range(1, num_codebooks):
                other_codebooks.append(all_codebooks[j].tolist())
            
            # Add to dataset
            dataset.append({
                "id": os.path.basename(wav_file).split(".")[0],
                "file_path": str(wav_file),
                "zeroth_codebook": zeroth_codebook,
                "other_codebooks": other_codebooks,
                "num_codebooks": num_codebooks,
                "sequence_length": seq_len
            })
            
            # Save individual example
            example_path = os.path.join(output_dir, f"example_{i:06d}.json")
            with open(example_path, 'w') as f:
                json.dump({
                    "id": os.path.basename(wav_file).split(".")[0],
                    "zeroth_codebook": zeroth_codebook,
                    "other_codebooks": other_codebooks
                }, f)
            
            # Print info occasionally
            if i % 10 == 0:
                logger.info(f"Processed {i+1}/{len(wav_files)} files, found {num_codebooks} codebooks")
        
        except Exception as e:
            logger.error(f"Error processing {wav_file}: {e}")
    
    # Save metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    logger.info(f"Saving metadata to {metadata_path}")
    with open(metadata_path, 'w') as f:
        json.dump({
            "total_examples": len(dataset),
            "codebook_stats": {
                "min_codebooks": min(item["num_codebooks"] for item in dataset),
                "max_codebooks": max(item["num_codebooks"] for item in dataset),
                "min_sequence_length": min(item["sequence_length"] for item in dataset),
                "max_sequence_length": max(item["sequence_length"] for item in dataset)
            }
        }, f, indent=2)
    
    # Create train/validation split indices
    num_examples = len(dataset)
    indices = list(range(num_examples))
    
    # Use 90% for training, 10% for validation
    train_size = int(0.9 * num_examples)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Save splits
    splits_path = os.path.join(output_dir, "splits.json")
    logger.info(f"Saving train/val splits to {splits_path}")
    with open(splits_path, 'w') as f:
        json.dump({
            "train": train_indices,
            "val": val_indices
        }, f)
    
    logger.info(f"Successfully processed {len(dataset)} files")
    logger.info(f"Dataset saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Generate dataset for training codebook decoder")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory containing audio files")
    parser.add_argument("--output_dir", type=str, default="./codebook_dataset",
                       help="Directory to save dataset")
    parser.add_argument("--max_files", type=int, default=None,
                       help="Maximum number of files to process")
    parser.add_argument("--max_seq_len", type=int, default=2000,
                       help="Maximum sequence length to keep")
    
    args = parser.parse_args()
    process_audio_directory(args.input_dir, args.output_dir, args.max_files, args.max_seq_len)

if __name__ == "__main__":
    main()
