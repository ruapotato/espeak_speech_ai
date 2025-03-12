#!/usr/bin/env python3
"""
SimpleMapper model definition module

This file defines the SimpleMapper class for codebook mapping.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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
