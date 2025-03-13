#!/usr/bin/env python3
"""
Inference script for codebook decoder model

This script loads a trained decoder model and uses it to generate audio from
zeroth codebook tokens. It can use GPU acceleration for inference.

Usage:
python inference-pipeline.py --model_path ./big_mapper_model/final_model.pt --input_file ./output/zeroth_codebook.txt --output_file output.wav --use_gpu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
import argparse
import logging
import wave
import re
from tqdm import tqdm
from transformers import MimiModel, AutoFeatureExtractor

# Define CodebookDecoderModel here so we don't need to import from another file
class CodebookDecoderModel(nn.Module):
    def __init__(self, vocab_size=1024, embed_dim=256, hidden_dim=512, num_layers=2, num_codebooks=32):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Simplify architecture for initial testing
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,  # Reduced number of layers
            bidirectional=False,    # Remove bidirectional for simplicity
            batch_first=True,
            dropout=0.1
        )
        
        self.output_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, vocab_size)
            ) for _ in range(num_codebooks - 1)
        ])
    
    def forward(self, input_ids, lengths=None):
        batch_size = input_ids.size(0)
        
        # Input validation - ensure token indices are within vocab size
        input_ids = torch.clamp(input_ids, 0, self.vocab_size - 1)
        
        # Embedding
        embedded = self.embedding(input_ids)
        
        # Process through LSTM - no packing needed due to bucketing
        lstm_output, _ = self.lstm(embedded)
        
        # Generate predictions
        predictions = []
        for layer in self.output_layers:
            pred = layer(lstm_output)
            predictions.append(pred)
        
        return torch.stack(predictions, dim=1)
    
    def predict(self, input_ids, temperature=1.0):
        """Generate predictions for inference"""
        with torch.no_grad():
            # Input validation
            input_ids = torch.clamp(input_ids, 0, self.vocab_size - 1)
            
            # Get logits for each codebook
            outputs = self(input_ids)  # [batch, num_codebooks-1, seq_len, vocab_size]
            
            # Generate samples for each codebook
            predictions = []
            for i in range(outputs.shape[1]):
                # Apply temperature
                if temperature != 1.0:
                    logits = outputs[:, i] / temperature
                else:
                    logits = outputs[:, i]
                
                # Convert to probabilities
                probs = F.softmax(logits, dim=-1)
                
                # Sample from distribution
                sampled = torch.multinomial(
                    probs.reshape(-1, self.vocab_size),
                    num_samples=1
                ).reshape(input_ids.shape)
                
                predictions.append(sampled)
            
            # Stack all codebooks (including zeroth)
            all_codebooks = torch.stack([input_ids] + predictions, dim=1)
            
            return all_codebooks

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
    return logging.getLogger("inference")

def get_model_dimensions(checkpoint):
    """Extract model dimensions from the checkpoint"""
    # Get embedding weight to determine embed_dim
    embed_dim = checkpoint["model_state_dict"]["embedding.weight"].shape[1]
    
    # Get hidden dimension from the LSTM
    hidden_dim = checkpoint["model_state_dict"]["lstm.weight_ih_l0"].shape[0] // 4
    
    # Get number of codebooks from the output layers
    num_output_layers = len([k for k in checkpoint["model_state_dict"].keys() if k.startswith("output_layers.")])
    num_codebooks = num_output_layers // 4 + 1  # Each output layer has 4 parameter tensors + 1 for zeroth codebook
    
    return embed_dim, hidden_dim, num_codebooks

def load_model(model_path, device='cpu'):
    """Load the trained decoder model"""
    logger = logging.getLogger("inference")
    logger.info(f"Loading model from {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get vocabulary size from checkpoint or default to 2048
    vocab_size = checkpoint.get("vocab_size", 2048)
    
    # Extract dimensions from checkpoint
    embed_dim, hidden_dim, num_codebooks = get_model_dimensions(checkpoint)
    logger.info(f"Model parameters: vocab_size={vocab_size}, num_codebooks={num_codebooks}")
    logger.info(f"Model dimensions: embed_dim={embed_dim}, hidden_dim={hidden_dim}")
    
    # Create model with correct dimensions
    model = CodebookDecoderModel(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_codebooks=num_codebooks
    )
    
    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Move to device
    model = model.to(device)
    
    # Set to evaluation mode
    model.eval()
    
    return model, num_codebooks

def load_zeroth_codebook(input_file):
    """Load zeroth codebook tokens from file"""
    logger = logging.getLogger("inference")
    logger.info(f"Loading zeroth codebook from {input_file}")
    
    with open(input_file, 'r') as f:
        content = f.read().strip()
    
    # Check if we have the indexed format (e.g., "0:1673 1:563 2:396...")
    if re.search(r'\d+:\d+', content):
        logger.info("Detected indexed token format")
        # Parse the indexed format, extracting just the token values
        tokens = []
        for item in content.split():
            parts = item.split(':')
            if len(parts) == 2:
                try:
                    tokens.append(int(parts[1]))
                except ValueError:
                    logger.warning(f"Could not parse token: {item}")
            else:
                logger.warning(f"Unexpected token format: {item}")
    else:
        logger.info("Detected plain token format")
        # Simple space-separated integers
        tokens = []
        for item in content.split():
            try:
                tokens.append(int(item))
            except ValueError:
                logger.warning(f"Could not parse token: {item}")
    
    logger.info(f"Loaded {len(tokens)} tokens")
    return tokens

def create_audio(all_codebooks, mimi_model, feature_extractor, device='cpu'):
    """Create audio from all codebooks using Mimi model"""
    logger = logging.getLogger("inference")
    logger.info(f"Creating audio from {all_codebooks.shape[1]} codebooks")
    
    # Ensure all_codebooks is in the right shape for Mimi: [batch, seq_len, codebooks]
    if all_codebooks.dim() == 3 and all_codebooks.shape[1] == all_codebooks.shape[2]:
        # Shape is [batch, codebooks, seq_len], need to transpose
        all_codebooks = all_codebooks.permute(0, 2, 1)
    
    # Generate audio
    logger.info(f"Generating audio with shape {all_codebooks.shape}")
    with torch.no_grad():
        try:
            # Move to CPU first to avoid CUDA errors during decoding
            all_codebooks_cpu = all_codebooks.cpu()
            decoded_audio = mimi_model.decode(all_codebooks_cpu)[0].numpy()
            logger.info(f"Generated {len(decoded_audio)} audio samples")
            return decoded_audio
        except Exception as e:
            logger.error(f"Error generating audio: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Try with a subset of codebooks as fallback
            logger.info("Trying fallback with fewer codebooks...")
            try:
                # Try with just the zeroth codebook
                minimal_codebooks = torch.zeros((1, all_codebooks_cpu.shape[1], 1), dtype=torch.long)
                minimal_codebooks[0, :, 0] = all_codebooks_cpu[0, :, 0]  # Just use zeroth codebook
                
                decoded_audio = mimi_model.decode(minimal_codebooks)[0].numpy()
                logger.info(f"Generated {len(decoded_audio)} audio samples with fallback")
                return decoded_audio
            except Exception as e2:
                logger.error(f"Fallback also failed: {e2}")
                return np.zeros(1000)  # Return empty audio as last resort

def save_audio(audio_data, sample_rate, output_file):
    """Save audio data to WAV file"""
    logger = logging.getLogger("inference")
    logger.info(f"Saving audio to {output_file}")
    
    # Convert to int16
    audio_int16 = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16)
    
    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 2 bytes for int16
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())

def main():
    parser = argparse.ArgumentParser(description="Run inference with codebook decoder model")
    
    # Input/output arguments
    parser.add_argument("--model_path", type=str, default="./big_mapper_model/final_model.pt",
                      help="Path to trained model checkpoint")
    parser.add_argument("--input_file", type=str, required=True,
                      help="File containing zeroth codebook tokens")
    parser.add_argument("--output_file", type=str, required=True,
                      help="Output WAV file path")
    
    # Optional arguments
    parser.add_argument("--use_gpu", action="store_true",
                      help="Use GPU for inference")
    parser.add_argument("--temperature", type=float, default=0.01,
                      help="Sampling temperature (higher = more random)")
    parser.add_argument("--batch_size", type=int, default=1,
                      help="Batch size for inference")
    parser.add_argument("--max_seq_len", type=int, default=2000,
                      help="Maximum sequence length to process at once")
    parser.add_argument("--verbosity", type=int, default=2,
                      help="Verbosity level (0-3)")
    
    args = parser.parse_args()
    logger = setup_logging(args.verbosity)
    
    # Set device
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using GPU for inference")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU for inference")
    
    # Load mapper model
    decoder_model, num_codebooks = load_model(args.model_path, device=device)
    
    # Load zeroth codebook
    zeroth_tokens = load_zeroth_codebook(args.input_file)
    
    # Load Mimi model for audio generation
    logger.info("Loading Mimi model")
    mimi_model = MimiModel.from_pretrained("kyutai/mimi")
    feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
    
    # Convert tokens to tensor
    zeroth_tensor = torch.tensor([zeroth_tokens], dtype=torch.long, device=device)
    
    # Process in chunks if necessary
    if len(zeroth_tokens) > args.max_seq_len:
        logger.info(f"Processing in chunks (sequence length: {len(zeroth_tokens)}, max: {args.max_seq_len})")
        
        # Process in chunks and concatenate results
        all_audio = []
        for i in range(0, len(zeroth_tokens), args.max_seq_len):
            end = min(i + args.max_seq_len, len(zeroth_tokens))
            chunk = zeroth_tensor[:, i:end]
            
            logger.info(f"Processing chunk {i}-{end} ({end-i} tokens)")
            
            # Generate all codebooks
            with torch.no_grad():
                all_codebooks = decoder_model.predict(chunk, temperature=args.temperature)
            
            # Generate audio for this chunk
            chunk_audio = create_audio(all_codebooks, mimi_model, feature_extractor, device)
            all_audio.append(chunk_audio)
        
        # Concatenate all audio chunks
        audio_data = np.concatenate(all_audio)
    else:
        # Process all at once
        # Generate all codebooks
        with torch.no_grad():
            all_codebooks = decoder_model.predict(zeroth_tensor, temperature=args.temperature)
        
        # Generate audio
        audio_data = create_audio(all_codebooks, mimi_model, feature_extractor, device)
    
    # Save audio
    save_audio(audio_data, feature_extractor.sampling_rate, args.output_file)
    
    logger.info(f"Done! Audio saved to {args.output_file}")

if __name__ == "__main__":
    main()
