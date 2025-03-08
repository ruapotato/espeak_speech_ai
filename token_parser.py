#!/usr/bin/env python3
"""
Parse LLM output and convert to separate codebook files
"""
import re
import sys
import numpy as np

def parse_llm_output(input_file):
    """Parse LLM output and separate into codebook files"""
    # Read the input file
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Extract the response part
    response_match = re.search(r'### Response:\s*(.*?)(?=\n\n|$)', content, re.DOTALL)
    if not response_match:
        print("Error: Could not find Response section")
        return False
        
    tokens_text = response_match.group(1).strip()
    
    # Initialize arrays for each codebook
    codebooks = [[] for _ in range(32)]
    
    # Parse token patterns like "X:Y"
    token_pattern = re.compile(r'(\d+):(\d+)')
    matches = token_pattern.findall(tokens_text)
    
    if not matches:
        print("Error: No valid tokens found")
        return False
    
    # Group tokens by codebook
    for cb, token in matches:
        cb = int(cb)
        token = int(token)
        if cb < 32:  # Only accept valid codebook IDs
            codebooks[cb].append(token)
    
    # Find longest sequence
    max_len = max(len(tokens) for tokens in codebooks)
    
    # Pad all sequences to same length
    for i in range(32):
        while len(codebooks[i]) < max_len:
            codebooks[i].append(0)  # Pad with zeros
    
    # Save each codebook to a separate file
    for i in range(32):
        output_file = f'codebook_{i}_tokens.txt'
        with open(output_file, 'w') as f:
            f.write(' '.join(map(str, codebooks[i])))
    
    print(f"Successfully parsed {len(matches)} tokens into {max_len} frames")
    return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <model_output.txt>")
        sys.exit(1)
    
    success = parse_llm_output(sys.argv[1])
    sys.exit(0 if success else 1)
