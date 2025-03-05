#!/usr/bin/env python3
# Author: # UniAudio Teams

import sys
import torch
import argparse
import logging
import os
import torchaudio
from tools.tokenizer.MimiCodec.mimi_tokenizer import MimiTokenizer
from tools.tokenizer.GLM4V.semantic import SSLTokenizer

def get_parser():
    parser = argparse.ArgumentParser(
        description="convert a data list, do tokenization and save as a torch .pt file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-file", type=str, default=None, help="text file in the format <exampe_id> <content>")
    parser.add_argument("--tar-file", type=str, default=None, help="we use tar chunk to save audio information")
    parser.add_argument("--tar-info", type=str, default=None, help="the file to save tar information")
    parser.add_argument("--wav-scp", type=str, default=None, help="kaldi wav.scp file")
    parser.add_argument("--segments", type=str, default=None, help="kaldi segment file")
    parser.add_argument("--output-file", type=str, help="dict")
    parser.add_argument("--output-text", type=str, default=None, help="dict")
    parser.add_argument("--output-utt2spk", type=str, default=None, help="dict")
    parser.add_argument("--tokenizer", type=str, choices=['audio', 'g2p', 'stablecodec', 'semantic', 'mimi', 'ssl'], help="what tokenizer to use")
    parser.add_argument("--rank", type=int, help="local GPU rank, if applicable")
    parser.add_argument("--batch-size", type=int, default=1, help="for batch tokenization")
    return parser

def main(args):
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format=f"%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    )
    args = get_parser().parse_args(args)
    args.rank -= 1 # run.pl starts from 1 but the exact jobid / gpuid starts from 0   
    max_gpu = torch.cuda.device_count()
    logging.info(f"max gpu {max_gpu}")
    args.rank = (args.rank % max_gpu) #

    device = torch.device(f"cuda:{args.rank}")
    logging.info(f"Using device: {device}")

    # Initialize tokenizer
    tokenizer = MimiTokenizer(device=device)
    logging.info('tokenizer built')
    
    # Output files
    if args.output_text is not None:
        f_text = open(args.output_text, 'w')
        f_utt2spk = open(args.output_utt2spk, 'w')
    
    # Data dictionary to store results
    data_dict = {}
    
    # Create tar_info file
    if args.tar_info is not None:
        f_info = open(args.tar_info, 'w')
    
    # Process based on input type
    if args.tar_file is not None:
        # Read paths from tar file
        for i, line in enumerate(open(args.tar_file, 'r')):
            parts = line.strip().split()
            if len(parts) < 2:
                logging.error(f"Invalid line in tar file: {line}")
                continue
                
            key = parts[0]  # First part is the key
            path = parts[-1]  # Last part is the file path
            
            try:
                # Load with torchaudio (same approach as our working example)
                logging.info(f"Processing file {i+1}: {key} from {path}")
                wav, sample_rate = torchaudio.load(path)
                logging.debug(f"  Shape: {wav.shape}, Sample rate: {sample_rate}")
                
                # Tokenize
                codes = tokenizer.tokenize(wav, sample_rate)
                logging.debug(f"  Codes shape: {codes.shape}")
                
                # Save results
                data_dict[key] = codes
                
                # Write to output files
                if args.tar_info is not None:
                    f_info.write(f"{key}\n")
                if args.output_text is not None:
                    # Get text from file name as simple placeholder
                    text = os.path.basename(path).replace('.wav', '')
                    f_text.write(f"{key} {text}\n")
                    f_utt2spk.write(f"{key} unknown\n")
                
                # Log progress
                if (i+1) % 10 == 0:
                    logging.info(f"Successfully processed {i+1} files")
                    
            except Exception as e:
                logging.error(f"Error processing {path}: {e}")
    
    # Save the processed data
    torch.save(data_dict, args.output_file)
    logging.info(f"Completed processing. Saved {len(data_dict)} entries to {args.output_file}")
    
    # Close output files
    if args.tar_info is not None:
        f_info.close()
    if args.output_text is not None:
        f_text.close()
        f_utt2spk.close()

if __name__ == "__main__":
    main(sys.argv[1:])
