#!/usr/bin/env python3
"""
Fixed CPU-based Model with Dynamic Vocabulary Size

This version scans the dataset first to determine the correct vocabulary size
and clips any out-of-range tokens to prevent embedding errors.

Usage:
python fixed-vocab-model.py --data_dir ./codebook_dataset --output_dir ./mapper_model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
import argparse
from tqdm import tqdm
import logging
import random

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
    return logging.getLogger("codebook-mapper")

def analyze_dataset(data_dir, split="train"):
    """Analyze dataset to find max token values and number of codebooks"""
    logger = logging.getLogger("codebook-mapper")
    logger.info("Analyzing dataset to determine vocabulary size...")
    
    # Load splits
    splits_path = os.path.join(data_dir, "splits.json")
    with open(splits_path, 'r') as f:
        splits = json.load(f)
    
    # Get indices for this split
    indices = splits[split]
    
    # Initialize counters
    max_token_value = 0
    max_codebooks = 0
    
    # Process each example
    for idx in tqdm(indices, desc="Analyzing dataset"):
        try:
            example_path = os.path.join(data_dir, f"example_{idx:06d}.json")
            with open(example_path, 'r') as f:
                example = json.load(f)
            
            # Check zeroth codebook
            zeroth_codebook = example["zeroth_codebook"]
            if zeroth_codebook:
                max_token_value = max(max_token_value, max(zeroth_codebook))
            
            # Check other codebooks
            other_codebooks = example["other_codebooks"]
            max_codebooks = max(max_codebooks, len(other_codebooks) + 1)  # +1 for zeroth
            
            for codebook in other_codebooks:
                if codebook:
                    max_token_value = max(max_token_value, max(codebook))
        except Exception as e:
            logger.warning(f"Error processing example {idx}: {e}")
    
    # Add a small buffer to max_token_value
    vocab_size = max_token_value + 1
    
    logger.info(f"Found max token value: {max_token_value}")
    logger.info(f"Setting vocabulary size to: {vocab_size}")
    logger.info(f"Number of codebooks: {max_codebooks}")
    
    return vocab_size, max_codebooks

class CodebookMapperDataset(Dataset):
    """Dataset for codebook mapping with token validation"""
    def __init__(self, data_dir, split="train", max_samples=None, vocab_size=1024):
        """
        Args:
            data_dir: Directory containing dataset
            split: "train" or "val"
            max_samples: Maximum number of samples to load
            vocab_size: Size of vocabulary for token clipping
        """
        self.logger = logging.getLogger("codebook-mapper.dataset")
        self.data_dir = data_dir
        self.vocab_size = vocab_size
        
        # Load splits
        splits_path = os.path.join(data_dir, "splits.json")
        with open(splits_path, 'r') as f:
            splits = json.load(f)
        
        # Get indices for this split
        self.indices = splits[split]
        if max_samples is not None:
            self.indices = self.indices[:max_samples]
        
        self.logger.info(f"Loaded {len(self.indices)} {split} examples")
        
        # Pre-load data for faster training
        self.examples = []
        for idx in tqdm(self.indices, desc=f"Loading {split} data"):
            try:
                example_path = os.path.join(data_dir, f"example_{idx:06d}.json")
                with open(example_path, 'r') as f:
                    example = json.load(f)
                
                # Get zeroth codebook and clip tokens to vocab_size
                zeroth_codebook = [min(token, vocab_size - 1) for token in example["zeroth_codebook"]]
                
                # Get other codebooks and clip tokens
                other_codebooks = []
                for codebook in example["other_codebooks"]:
                    other_codebooks.append([min(token, vocab_size - 1) for token in codebook])
                
                # Store the example
                self.examples.append({
                    "zeroth_codebook": zeroth_codebook,
                    "other_codebooks": other_codebooks
                })
            except Exception as e:
                self.logger.error(f"Error loading example {idx}: {e}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        zeroth_codebook = example["zeroth_codebook"]
        other_codebooks = example["other_codebooks"]
        
        return {
            "zeroth_tokens": zeroth_codebook,
            "other_tokens": other_codebooks
        }

class SimpleMapper(nn.Module):
    """Simple codebook mapper - maps a token in 0th codebook to tokens in other codebooks"""
    def __init__(self, vocab_size=1024, num_codebooks=32, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_codebooks = num_codebooks
        
        # Embedding for zeroth codebook tokens
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Simple MLP for mapping
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Output heads for each codebook (1 through 31)
        self.output_heads = nn.ModuleList([
            nn.Linear(hidden_dim, vocab_size) for _ in range(num_codebooks - 1)
        ])
    
    def forward(self, zeroth_tokens):
        """
        Forward pass through the model
        
        Args:
            zeroth_tokens: Tensor of tokens from zeroth codebook [batch_size, seq_len]
            
        Returns:
            List of tensors for each other codebook [batch_size, seq_len, vocab_size]
        """
        # Clip input tokens to valid range
        zeroth_tokens = torch.clamp(zeroth_tokens, 0, self.vocab_size - 1)
        
        # Embed zeroth tokens
        embedded = self.embedding(zeroth_tokens)  # [batch_size, seq_len, embed_dim]
        
        # Process through MLP
        hidden = self.mlp(embedded)  # [batch_size, seq_len, hidden_dim]
        
        # Generate outputs for each codebook
        outputs = []
        for head in self.output_heads:
            outputs.append(head(hidden))  # [batch_size, seq_len, vocab_size]
        
        return outputs
    
    def predict(self, zeroth_tokens, temperature=1.0):
        """Generate predictions for inference"""
        with torch.no_grad():
            # Clip input tokens
            zeroth_tokens = torch.clamp(zeroth_tokens, 0, self.vocab_size - 1)
            
            # Forward pass
            logits = self(zeroth_tokens)
            
            # Generate predictions
            predictions = []
            for logits_i in logits:
                # Apply temperature
                if temperature != 1.0:
                    logits_i = logits_i / temperature
                
                # Sample from distribution
                probs = F.softmax(logits_i, dim=-1)
                tokens = torch.multinomial(
                    probs.reshape(-1, self.vocab_size),
                    num_samples=1
                ).reshape(zeroth_tokens.shape)
                
                predictions.append(tokens)
            
            # Stack all codebooks (including zeroth)
            all_codebooks = torch.stack([zeroth_tokens] + predictions, dim=1)
            
            return all_codebooks

def collate_fn(batch):
    """Custom collate function for variable-length sequences"""
    # Find max sequence length
    max_len = max([len(item["zeroth_tokens"]) for item in batch])
    
    # Initialize tensors
    batch_size = len(batch)
    zeroth_batch = torch.zeros(batch_size, max_len, dtype=torch.long)
    
    # Lists to hold other codebooks
    other_batches = []
    num_codebooks = len(batch[0]["other_tokens"])
    for i in range(num_codebooks):
        other_batches.append(torch.zeros(batch_size, max_len, dtype=torch.long))
    
    # Mask for valid tokens
    mask_batch = torch.zeros(batch_size, max_len, dtype=torch.bool)
    
    # Fill tensors
    for i, item in enumerate(batch):
        seq_len = len(item["zeroth_tokens"])
        
        # Fill zeroth codebook
        zeroth_batch[i, :seq_len] = torch.tensor(item["zeroth_tokens"][:seq_len], dtype=torch.long)
        
        # Fill other codebooks
        for j, codebook in enumerate(item["other_tokens"]):
            if j < len(other_batches):
                cb_len = min(seq_len, len(codebook))
                other_batches[j][i, :cb_len] = torch.tensor(codebook[:cb_len], dtype=torch.long)
        
        # Set mask
        mask_batch[i, :seq_len] = True
    
    return {
        "zeroth_tokens": zeroth_batch,
        "other_tokens": other_batches,
        "mask": mask_batch,
        "num_codebooks": num_codebooks + 1  # +1 for zeroth
    }

def train_model(args):
    """Train the mapper model"""
    logger = setup_logging(args.verbosity)
    logger.info("Starting training")
    
    # Analyze dataset to get vocabulary size
    vocab_size, num_codebooks = analyze_dataset(args.data_dir)
    
    # Force CPU for training
    train_device = torch.device("cpu")
    logger.info(f"Training on {train_device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create datasets with correct vocab size
    train_dataset = CodebookMapperDataset(
        args.data_dir,
        split="train",
        max_samples=args.max_samples,
        vocab_size=vocab_size
    )
    
    val_dataset = CodebookMapperDataset(
        args.data_dir,
        split="val",
        max_samples=args.max_val_samples,
        vocab_size=vocab_size
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Create model with correct vocab size
    model = SimpleMapper(
        vocab_size=vocab_size,
        num_codebooks=num_codebooks,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim
    )
    model.to(train_device)
    
    # Log model size
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model has {num_params:,} parameters")
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Create loss function
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0
        train_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            zeroth_tokens = batch["zeroth_tokens"].to(train_device)
            other_tokens = [tokens.to(train_device) for tokens in batch["other_tokens"]]
            mask = batch["mask"].to(train_device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(zeroth_tokens)
            
            # Calculate loss (masked by valid tokens)
            batch_loss = 0
            for i, logits in enumerate(outputs):
                if i < len(other_tokens):
                    # Get target
                    target = other_tokens[i]
                    
                    # Calculate per-token loss
                    B, S, V = logits.shape  # batch, seq, vocab
                    token_loss = criterion(
                        logits.reshape(-1, V),
                        target.reshape(-1)
                    ).reshape(B, S)
                    
                    # Apply mask
                    masked_loss = token_loss * mask.float()
                    
                    # Average over valid tokens
                    if mask.sum() > 0:
                        avg_loss = masked_loss.sum() / mask.sum()
                        batch_loss += avg_loss
            
            # Average over codebooks
            if len(outputs) > 0:
                batch_loss = batch_loss / len(outputs)
            
            # Backward and optimize
            batch_loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            optimizer.step()
            
            # Update metrics
            train_loss += batch_loss.item()
            train_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({"loss": batch_loss.item()})
        
        # Calculate average loss
        train_loss = train_loss / max(1, train_batches)
        
        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
            for batch_idx, batch in enumerate(progress_bar):
                # Move to device
                zeroth_tokens = batch["zeroth_tokens"].to(train_device)
                other_tokens = [tokens.to(train_device) for tokens in batch["other_tokens"]]
                mask = batch["mask"].to(train_device)
                
                # Forward pass
                outputs = model(zeroth_tokens)
                
                # Calculate loss (masked by valid tokens)
                batch_loss = 0
                for i, logits in enumerate(outputs):
                    if i < len(other_tokens):
                        # Get target
                        target = other_tokens[i]
                        
                        # Calculate per-token loss
                        B, S, V = logits.shape  # batch, seq, vocab
                        token_loss = criterion(
                            logits.reshape(-1, V),
                            target.reshape(-1)
                        ).reshape(B, S)
                        
                        # Apply mask
                        masked_loss = token_loss * mask.float()
                        
                        # Average over valid tokens
                        if mask.sum() > 0:
                            avg_loss = masked_loss.sum() / mask.sum()
                            batch_loss += avg_loss
                
                # Average over codebooks
                if len(outputs) > 0:
                    batch_loss = batch_loss / len(outputs)
                
                # Update metrics
                val_loss += batch_loss.item()
                val_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({"loss": batch_loss.item()})
        
        # Calculate average validation loss
        val_loss = val_loss / max(1, val_batches)
        
        # Log metrics
        logger.info(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "num_codebooks": num_codebooks,
            "vocab_size": vocab_size
        }
        
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save(checkpoint, best_model_path)
            logger.info(f"New best model saved: {best_model_path}")
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, "final_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "num_codebooks": num_codebooks,
        "vocab_size": vocab_size
    }, final_model_path)
    logger.info(f"Final model saved: {final_model_path}")
    
    return final_model_path

def main():
    parser = argparse.ArgumentParser(description="Train a simple codebook mapper model")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True,
                      help="Directory containing dataset")
    parser.add_argument("--output_dir", type=str, default="./mapper_model",
                      help="Directory to save trained model")
    parser.add_argument("--max_samples", type=int, default=None,
                      help="Maximum number of training samples to use")
    parser.add_argument("--max_val_samples", type=int, default=None,
                      help="Maximum number of validation samples to use")
    
    # Model arguments
    parser.add_argument("--embed_dim", type=int, default=128,
                      help="Embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=256,
                      help="Hidden dimension")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=16,
                      help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=20,
                      help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                      help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0001,
                      help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                      help="Maximum gradient norm for clipping")
    parser.add_argument("--verbosity", type=int, default=2,
                      help="Verbosity level (0-3)")
    
    args = parser.parse_args()
    model_path = train_model(args)
    
    # Test the model
    print(f"\nTraining complete! Model saved to {model_path}")
    print("To use this model for inference with GPU acceleration, run:")
    print(f"python inference.py --model_path {model_path} --input_file your_zeroth_codebook.txt --output_file output.wav --use_gpu")

if __name__ == "__main__":
    main()
