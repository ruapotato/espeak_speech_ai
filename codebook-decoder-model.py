#!/usr/bin/env python3
"""
Codebook Decoder Model Training Script with bucketing, enhanced debugging,
and fixed token validation to handle out-of-range indices
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import json
import logging
import argparse
from tqdm import tqdm
from pathlib import Path
import numpy as np
from collections import defaultdict

def setup_logging(output_dir):
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def analyze_dataset(data_dir, split='train'):
    """Analyze dataset to determine vocabulary size"""
    logger = logging.getLogger(__name__)
    logger.info("Analyzing dataset to determine vocabulary size...")
    
    data_dir = Path(data_dir)
    # Load splits
    with open(data_dir / 'splits.json', 'r') as f:
        splits = json.load(f)
    
    # Get indices for this split
    indices = splits[split]
    
    # Initialize counters
    max_token_value = 0
    
    # Process each example
    for idx in tqdm(indices, desc="Analyzing dataset"):
        try:
            example_path = data_dir / f"example_{idx:06d}.json"
            if not example_path.exists():
                continue
                
            with open(example_path, 'r') as f:
                example = json.load(f)
            
            # Check zeroth codebook
            zeroth_codebook = example["zeroth_codebook"]
            if zeroth_codebook:
                max_token_value = max(max_token_value, max(zeroth_codebook))
            
            # Check other codebooks
            other_codebooks = example["other_codebooks"]
            for codebook in other_codebooks:
                if codebook:
                    max_token_value = max(max_token_value, max(codebook))
        except Exception as e:
            logger.warning(f"Error processing example {idx}: {e}")
    
    # Add a small buffer to max_token_value
    vocab_size = max_token_value + 1
    
    logger.info(f"Found max token value: {max_token_value}")
    logger.info(f"Setting vocabulary size to: {vocab_size}")
    
    return vocab_size

class BucketedCodebookDataset(Dataset):
    def __init__(self, data_dir, split='train', max_len=1000, bucket_size=32, vocab_size=1024):
        self.data_dir = Path(data_dir)
        self.max_len = max_len
        self.bucket_size = bucket_size
        self.vocab_size = vocab_size
        self.logger = logging.getLogger(__name__)
        
        # Load splits
        with open(self.data_dir / 'splits.json', 'r') as f:
            splits = json.load(f)
        
        # Process and bucket all examples
        self.examples = []
        self.buckets = defaultdict(list)
        
        for idx in splits[split]:
            try:
                file_path = self.data_dir / f'example_{idx:06d}.json'
                if not file_path.exists():
                    continue
                
                with open(file_path, 'r') as f:
                    example = json.load(f)
                
                # Clip zeroth codebook tokens to valid range and truncate to max_len
                zeroth_codebook = [min(token, self.vocab_size - 1) for token in example['zeroth_codebook'][:max_len]]
                if not zeroth_codebook:
                    continue
                    
                other_codebooks = example['other_codebooks']
                if not other_codebooks or len(other_codebooks) == 0:
                    continue
                
                # Truncate other codebooks to match zeroth_codebook length and clip token values
                other_codebooks = [
                    [min(token, self.vocab_size - 1) for token in cb[:len(zeroth_codebook)]]
                    for cb in other_codebooks
                ]
                
                # Get bucket index (round up to nearest bucket_size)
                bucket_idx = (len(zeroth_codebook) + bucket_size - 1) // bucket_size * bucket_size
                
                self.buckets[bucket_idx].append({
                    'zeroth_codebook': zeroth_codebook,
                    'other_codebooks': other_codebooks,
                    'length': len(zeroth_codebook)
                })
                
            except Exception as e:
                self.logger.warning(f"Error loading {file_path}: {str(e)}")
                continue
        
        # Flatten buckets into examples list, keeping similar lengths together
        for bucket_size in sorted(self.buckets.keys()):
            self.examples.extend(self.buckets[bucket_size])
        
        self.logger.info(f"Created {len(self.buckets)} buckets with {len(self.examples)} total examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        input_tensor = torch.tensor(example['zeroth_codebook'], dtype=torch.long)
        label_tensor = torch.tensor(example['other_codebooks'], dtype=torch.long)
        
        return {
            'input_ids': input_tensor,
            'labels': label_tensor,
            'length': example['length']
        }

def safe_collate_fn(batch):
    """Collate function that ensures all sequences in batch have same length"""
    # All sequences in a batch should have same length due to bucketing
    lengths = torch.tensor([x['length'] for x in batch])
    max_len = lengths.max().item()
    
    # Pre-allocate tensors
    batch_size = len(batch)
    num_codebooks = batch[0]['labels'].size(0)
    
    input_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    labels = torch.full((batch_size, num_codebooks, max_len), -100, dtype=torch.long)
    
    # Fill tensors
    for i, item in enumerate(batch):
        seq_len = item['length']
        input_ids[i, :seq_len] = item['input_ids']
        labels[i, :, :seq_len] = item['labels']
    
    return {
        'input_ids': input_ids,
        'labels': labels,
        'lengths': lengths
    }

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
        
        # Input validation
        if input_ids.dim() != 2:
            raise ValueError(f"Expected input_ids to have 2 dimensions, got {input_ids.dim()}")
        
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

def train_epoch(model, dataloader, optimizer, device, logger):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
        try:
            # Debug info
            logger.debug(f"Processing batch {batch_idx}")
            logger.debug(f"Input shape: {batch['input_ids'].shape}")
            logger.debug(f"Label shape: {batch['labels'].shape}")
            
            # Move data to device
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Ensure input tokens are within vocab range (sanity check)
            input_ids = torch.clamp(input_ids, 0, model.vocab_size - 1)
            
            # Forward pass
            outputs = model(input_ids)
            
            # Compute loss
            loss = 0
            for i in range(outputs.shape[1]):
                loss += nn.functional.cross_entropy(
                    outputs[:, i].transpose(1, 2),
                    labels[:, i],
                    ignore_index=-100
                )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Log every 100 batches
            if batch_idx % 100 == 0:
                logger.info(f"Batch {batch_idx}: Loss = {loss.item():.4f}")
            
        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {str(e)}")
            continue
    
    return total_loss / (num_batches if num_batches > 0 else 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_len", type=int, default=1000)
    parser.add_argument("--bucket_size", type=int, default=32)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--cpu_only", action="store_true", help="Force CPU training")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.output_dir)
    
    # Enable debug mode
    if args.debug:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        torch.backends.cudnn.enabled = False
    
    # Set device
    if args.cpu_only:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    try:
        # Analyze dataset to find vocabulary size
        vocab_size = analyze_dataset(args.data_dir)
        logger.info(f"Using vocabulary size: {vocab_size}")
        
        # Load datasets with correct vocabulary size
        train_dataset = BucketedCodebookDataset(
            args.data_dir, split='train',
            max_len=args.max_len,
            bucket_size=args.bucket_size,
            vocab_size=vocab_size
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,  # No shuffle needed due to bucketing
            num_workers=0,  # Single worker for debugging
            pin_memory=True,
            collate_fn=safe_collate_fn,
            drop_last=True
        )
        
        # Initialize model with correct vocabulary size
        model = CodebookDecoderModel(
            vocab_size=vocab_size,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim
        ).to(device)
        
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
        
        # Training loop
        for epoch in range(args.epochs):
            logger.info(f"Starting epoch {epoch + 1}/{args.epochs}")
            
            train_loss = train_epoch(model, train_loader, optimizer, device, logger)
            logger.info(f"Epoch {epoch + 1} - Training Loss: {train_loss:.4f}")
            
            # Save checkpoint
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                'vocab_size': vocab_size,
            }, checkpoint_path)
            
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
