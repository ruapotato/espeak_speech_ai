#!/usr/bin/env python3
"""
Codebook Decoder Model Training Script

This script trains a model to predict other codebooks from the zeroth codebook.
Includes GPU acceleration and modern PyTorch training practices.
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

class CodebookDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        self.data_dir = Path(data_dir)
        
        # Load splits
        with open(self.data_dir / 'splits.json', 'r') as f:
            splits = json.load(f)
        
        # Get relevant example indices
        self.example_indices = splits[split]
        
    def __len__(self):
        return len(self.example_indices)
    
    def __getitem__(self, idx):
        example_idx = self.example_indices[idx]
        example_path = self.data_dir / f'example_{example_idx:06d}.json'
        
        with open(example_path, 'r') as f:
            example = json.load(f)
        
        # Convert to tensors
        zeroth_codebook = torch.tensor(example['zeroth_codebook'], dtype=torch.long)
        other_codebooks = torch.tensor(example['other_codebooks'], dtype=torch.long)
        
        return {
            'input_ids': zeroth_codebook,
            'labels': other_codebooks
        }

class CodebookDecoderModel(nn.Module):
    def __init__(self, vocab_size=1024, embed_dim=256, hidden_dim=512, num_layers=4, num_codebooks=4):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Bidirectional LSTM for sequence processing
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )
        
        # Output layers for each codebook
        self.output_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, vocab_size)
            ) for _ in range(num_codebooks - 1)  # -1 because we don't predict the zeroth codebook
        ])
        
    def forward(self, input_ids):
        # Embed input tokens
        embedded = self.embedding(input_ids)  # [batch, seq_len, embed_dim]
        
        # Process through LSTM
        lstm_output, _ = self.lstm(embedded)  # [batch, seq_len, hidden_dim*2]
        
        # Generate predictions for each codebook
        predictions = []
        for layer in self.output_layers:
            pred = layer(lstm_output)  # [batch, seq_len, vocab_size]
            predictions.append(pred)
        
        return torch.stack(predictions, dim=1)  # [batch, num_codebooks-1, seq_len, vocab_size]

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        outputs = model(input_ids)
        
        # Compute loss
        loss = 0
        for i in range(outputs.shape[1]):
            loss += nn.functional.cross_entropy(
                outputs[:, i].view(-1, outputs.shape[-1]),
                labels[:, i].view(-1),
                ignore_index=-100
            )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids)
            
            loss = 0
            for i in range(outputs.shape[1]):
                loss += nn.functional.cross_entropy(
                    outputs[:, i].view(-1, outputs.shape[-1]),
                    labels[:, i].view(-1),
                    ignore_index=-100
                )
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Load datasets
    train_dataset = CodebookDataset(args.data_dir, split='train')
    val_dataset = CodebookDataset(args.data_dir, split='val')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    model = CodebookDecoderModel(
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim
    ).to(device)
    
    # Setup optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        logging.info(f"Epoch {epoch + 1} - Training Loss: {train_loss:.4f}")
        
        # Validate
        val_loss = validate(model, val_loader, device)
        logging.info(f"Epoch {epoch + 1} - Validation Loss: {val_loss:.4f}")
        
        # Save checkpoint if best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(args.output_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
            logging.info(f"Saved best model checkpoint to {checkpoint_path}")
        
        # Save latest model
        checkpoint_path = os.path.join(args.output_dir, 'latest_model.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }, checkpoint_path)

if __name__ == "__main__":
    main()
