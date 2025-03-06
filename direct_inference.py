#!/usr/bin/env python3
'''
Final solution for eSpeak Speech AI
Using reverse delay approach with tuned temperature settings
'''
import os
import sys
import logging
import argparse
import yaml
import torch
import torchaudio
import numpy as np
from pathlib import Path

# Add RSTnet paths to Python path
sys.path.append(os.path.expanduser("~/espeak_speech_ai/RSTnet"))
sys.path.append(os.path.expanduser("~/espeak_speech_ai/RSTnet/MLLM_v2"))

# Now import the necessary modules
from utils.sampling import sample_token, sample_token_audio, sample_token_audio_2048
from models.llama_streaming import GPT, Config
from tools.tokenizer.MimiCodec.mimi_tokenizer import MimiTokenizer
from utils.train_utils import to_device
from utils.dataloader import get_data_iterator_tokenizer_vocabulary
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_parser():
    parser = argparse.ArgumentParser(description='Direct inference for eSpeak Speech AI')
    # Model related
    parser.add_argument('--resume', type=str, 
                        default='~/espeak_speech_ai/checkpoints/meta-llama/Llama-3.2-1B-Instruct/lit_model.pth',
                        help='Model checkpoint to use')
    parser.add_argument('--exp_dir', type=str, 
                        default='~/espeak_speech_ai/checkpoints/espeak_speech_ai/espeak_llama32_1b_memory_optimized',
                        help='Experiment directory')
    
    # Inference related
    parser.add_argument('--inference_mode', type=str, default='sampling', 
                         choices=['sampling', 'greedy', 'teacher-force'],
                         help='Inference mode')
    parser.add_argument('--temp', type=float, default=0.5, 
                        help='Softmax temperature for audio generation (lower = more deterministic)')
    parser.add_argument('--temp_text', type=float, default=0.7, 
                        help='Softmax temperature for text generation')
    parser.add_argument('--topk', type=int, default=30, 
                        help='Top-k candidates in sampling')
    parser.add_argument('--seed', type=int, default=888, 
                        help='Random seed')
    
    # Device related
    parser.add_argument('--rank', type=int, default=0 if torch.cuda.is_available() else -1, 
                        help='GPU rank. -1 means CPU')
    
    # Task related
    parser.add_argument('--task_name', type=str, default='TTS',
                        choices=['TTS', 'ASR', 'text_only', 'audio_only'],
                        help='Task name')
    
    # Input/output related
    parser.add_argument('--text', type=str, default=None,
                        help='Text to convert to speech (for TTS)')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode')
    parser.add_argument('--data_json', type=str, default=None, 
                        help="Data JSON for inference")
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help="Output directory for generated files")
    parser.add_argument('--generate_target', type=str, default="audio", 
                        choices=['audio', 'text'],
                        help="Type of output to generate")
    parser.add_argument('--debug_mode', action='store_true',
                        help="Enable debug logging and token visualization")
    
    return parser

def load_model(args):
    """Load the model and prepare it for inference"""
    # Try to load Llama tokenizer if available
    llm_tokenizer = None
    try:
        llm_tokenizer_path = os.path.expanduser('~/espeak_speech_ai/checkpoints/meta-llama/Llama-3.2-1B-Instruct')
        llm_tokenizer = AutoTokenizer.from_pretrained(llm_tokenizer_path)
        logger.info(f"Loaded LLM tokenizer from {llm_tokenizer_path}")
    except Exception as e:
        logger.warning(f"Could not load Llama tokenizer: {e}. Using character-based tokenization.")
    
    # Expand user paths
    checkpoint_path = os.path.expanduser(args.resume)
    exp_dir = os.path.expanduser(args.exp_dir)
    
    # Set device
    if args.rank >= 0 and torch.cuda.is_available():
        args.rank = args.rank % torch.cuda.device_count()
        device = torch.device(f'cuda:{args.rank}')
    else:
        device = torch.device('cpu')
    
    logger.info(f'Using device: {device}')
    
    # Load config
    config_path = os.path.join(exp_dir, 'config.yaml')
    with open(config_path, "r", encoding="utf-8") as f:
        train_args = yaml.safe_load(f)
        train_args = argparse.Namespace(**train_args)
    
    # Fix any missing parameters that might be needed
    if not hasattr(train_args, 'parallel_number'):
        train_args.parallel_number = 9  # Default for eSpeak model
    
    if not hasattr(train_args, 'model_config'):
        train_args.model_config = os.path.expanduser('~/espeak_speech_ai/checkpoints/meta-llama/Llama-3.2-1B-Instruct/model_config.yaml')
    
    # Create model
    logger.info("Initializing model...")
    config = Config.from_file(
        train_args.model_config, 
        lora_r=train_args.lora_r,
        lora_alpha=train_args.lora_alpha, 
        lora_dropout=train_args.lora_dropout, 
        lora_query=train_args.lora_query, 
        lora_key=train_args.lora_key,
        lora_value=train_args.lora_value, 
        lora_projection=train_args.lora_projection, 
        lora_mlp=train_args.lora_mlp,
        lora_head=train_args.lora_head, 
        audio_card=train_args.audio_card, 
        codecformer_dim=train_args.codecformer_dim,
        n_q=train_args.n_q, 
        dep_q=train_args.dep_q, 
        codecformer_heads=train_args.codecformer_heads, 
        codecformer_layers=train_args.codecformer_layers,
        codecformer_hidden_scale=train_args.codecformer_hidden_scale, 
        causal=train_args.causal,
        codecformer_multi_linear=train_args.codecformer_multi_linear, 
        codecformer_weights_per_step=train_args.codecformer_weights_per_step,
        codecformer_dim_feedforward=train_args.codecformer_dim_feedforward
    )
    
    # Create model with mixed precision for efficiency
    training_dtype = torch.bfloat16
    model = GPT(config).to(device=device, dtype=training_dtype)
    model.eval()
    
    # Load checkpoint - FIXED VERSION WITH FALLBACK
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    try:
        # First try loading without weights_only (safer approach may fail with some checkpoints)
        logger.info("Attempting to load checkpoint with default settings")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Determine state dict format
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
            
        # Load the state dict with strict=False to allow missing keys for LoRA
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        logger.info(f"Missing {len(missing_keys)} keys in state dict")
        logger.info(f"Unexpected {len(unexpected_keys)} keys in state dict")
        logger.info("Checkpoint loaded successfully")
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        sys.exit(1)
    
    # Initialize audio tokenizer
    logger.info("Initializing audio tokenizer...")
    audio_tokenizer = MimiTokenizer(device=device)
    logger.info("Audio tokenizer initialized with default settings")
    
    return model, audio_tokenizer, device, train_args, llm_tokenizer

def inspect_audio_tokens(tokens, label=""):
    """Debug utility to inspect audio token statistics"""
    if tokens is None:
        logger.debug(f"{label} tokens: None")
        return
        
    try:
        if isinstance(tokens, torch.Tensor):
            min_val = tokens.min().item()
            max_val = tokens.max().item()
            mean_val = tokens.float().mean().item()
            shape = tokens.shape
            logger.debug(f"{label} tokens: shape={shape}, min={min_val}, max={max_val}, mean={mean_val:.2f}")
        else:
            logger.debug(f"{label} tokens: {type(tokens)}")
    except Exception as e:
        logger.debug(f"Error inspecting {label} tokens: {e}")

def create_input_sequence(text, device, llm_tokenizer=None, text_empty_token=128002, semantic_pad_token=2049):
    """
    Create input sequence from text input with proper tokenization
    """
    # Use the LLM's tokenizer if provided, otherwise do basic tokenization
    if llm_tokenizer:
        # Use Llama tokenizer to get proper tokens
        encoded = llm_tokenizer.encode(text, add_special_tokens=True)
        text_tokens = torch.tensor(encoded, dtype=torch.long, device=device)
    else:
        # Fallback to char-level tokenization
        char_to_token = {}
        for i, c in enumerate("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?-'\""):
            char_to_token[c] = i + 1
        
        # Tokenize text
        text_tokens = []
        for c in text:
            if c in char_to_token:
                text_tokens.append(char_to_token[c])
            else:
                text_tokens.append(1)  # Unknown character token
        text_tokens = torch.tensor(text_tokens, dtype=torch.long, device=device)
    
    # Log tokenization details for debugging
    logger.info(f"Text to tokenize: '{text}'")
    logger.info(f"Tokenized length: {len(text_tokens)}")
    logger.info(f"First few tokens: {text_tokens[:10]}")
    
    # Create sequence tensor (9 channels: 1 text + 8 audio)
    seq_len = len(text_tokens)
    if seq_len == 0:
        raise ValueError("Text tokenized to an empty sequence. Please check the input text.")
        
    seq = torch.zeros((9, seq_len + 20), dtype=torch.long, device=device)
    mask = torch.ones((9, seq_len + 20), dtype=torch.bool, device=device)
    
    # Fill with text tokens and empty audio tokens
    seq[0, :seq_len] = text_tokens
    seq[0, seq_len:] = text_empty_token
    
    for i in range(1, 9):
        seq[i, :] = semantic_pad_token
    
    return seq, mask

def prepare_for_inference(seq, mask, task_name):
    """Prepare sequence and mask for inference based on task type"""
    text_pad_token = 128003
    acoustic_pad_token = 2049
    semantic_pad_token = 2049
    text_empty_token = 128002
    
    # Add debug logging
    logger.info(f"Initial sequence shape: {seq.shape}")
    logger.info(f"Initial mask shape: {mask.shape}")
    
    # Reshape for batch
    seq = seq.unsqueeze(0)  # Add batch dimension
    mask = mask.unsqueeze(0)
    
    logger.info(f"After unsqueeze - sequence shape: {seq.shape}, mask shape: {mask.shape}")
    
    # For TTS task, we need a different approach since we're generating audio from text
    if task_name == 'TTS':
        # Find the last non-empty-token position in the text channel
        text_indices = (seq[0, 0, :] != text_empty_token).nonzero(as_tuple=True)[0]
        
        if len(text_indices) == 0:
            # No valid text found
            logger.error("No valid text tokens found in the sequence.")
            return None, None, 0, 0, None
            
        # Get the last valid text position
        prefix_len = text_indices[-1].item() + 1
        logger.info(f"Using prefix length: {prefix_len} for TTS task")
        
        # Use text as prefix, but don't truncate yet
        prefix = seq.clone()
        return_mask = mask.clone()
        
        # Set max and min generation lengths
        maxlen = min(150, prefix_len * 3)  # Increased for longer audio
        minlen = min(50, prefix_len)       # Increased for more stable output
        
        # We don't have ground truth audio for generation
        gt_audio = None
        
        return prefix, return_mask, maxlen, minlen, gt_audio

    # Original code for other tasks
    # Remove padding based on task
    if task_name in ['text_only', 'ASR']:
        # Find end of text
        pad_len = seq[0, 0, :].eq(text_pad_token).int().sum().item()
        mask = mask[:, :, :seq.shape[2] - pad_len]
        seq = seq[:, :, :seq.shape[2] - pad_len]
    elif task_name == 'audio_only':
        # Find end of audio
        pad_len = seq[0, 1:2, :].eq(semantic_pad_token).int().sum().item()
        mask = mask[:, :, :seq.shape[2] - pad_len]
        seq = seq[:, :, :seq.shape[2] - pad_len]
    
    # Prepare for different tasks
    if task_name == 'text_only':
        # For text continue: use half of text as prompt
        prefix_len = seq.shape[2] // 2
        prefix = seq[:, :, :prefix_len]
        mask = mask[:, :, :prefix_len]
        maxlen = prefix_len
        minlen = prefix_len
        gt_audio = None
    elif task_name == 'audio_only':
        # For audio continue: use half of sequence as prompt
        prefix_len = seq.shape[2] // 2
        prefix = seq[:, :, :prefix_len]
        mask = mask[:, :, :prefix_len]
        maxlen = prefix_len
        minlen = prefix_len
        gt_audio = None
    elif task_name == 'ASR':
        # For ASR: use audio as prefix, generate text
        prefix_len = seq[0, 0, :].eq(text_empty_token).int().sum().item()
        prefix = seq[:, :, :prefix_len+1]
        mask = mask[:, :, :prefix_len+1]
        maxlen = seq.shape[2]-prefix_len + 13
        minlen = seq.shape[2]-prefix_len - 13
        gt_audio = None
    
    return prefix, mask, maxlen, minlen, gt_audio

def apply_reverse_delay(tokens):
    """Apply a reverse delay to token streams 1-7 (proven to work in experiments)"""
    # Make a deep copy first
    result = tokens.clone()
    
    # Make sure we're working with [seq_len, 8] format for easier processing
    need_transpose = False
    if result.dim() == 2 and result.shape[0] == 8 and result.shape[1] > 8:
        result = result.transpose(0, 1)
        need_transpose = True
    
    # Apply reverse delay
    seq_len = result.shape[0]
    if seq_len > 1:
        for i in range(1, 8):  # Channels 1-7
            # Shift each channel backward by 1 position
            temp = result[1:, i].clone()
            result[:-1, i] = temp
            # Pad the last position
            result[-1, i] = 2049  # Pad token
    
    # Transpose back if needed
    if need_transpose:
        result = result.transpose(0, 1)
    
    return result

def visualize_tokens_as_image(tokens, output_file="token_visualization.png"):
    """
    Visualizes the token sequence as a 2D image to help debug token patterns.
    
    Args:
        tokens: Token tensor of shape [seq_len, 8] or [8, seq_len]
        output_file: File to save the visualization to
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Make sure tokens are on CPU and convert to numpy
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.detach().cpu().numpy()
        
        # Make sure shape is [seq_len, 8]
        if tokens.shape[0] == 8 and tokens.shape[1] > 8:
            tokens = tokens.T
            
        # Create figure
        plt.figure(figsize=(10, 6))
        plt.imshow(tokens, aspect='auto', cmap='viridis')
        plt.colorbar(label='Token ID')
        plt.xlabel('Channel')
        plt.ylabel('Frame')
        plt.title('Audio Token Visualization')
        plt.tight_layout()
        plt.savefig(output_file)
        logger.info(f"Token visualization saved to {output_file}")
    except Exception as e:
        logger.warning(f"Could not visualize tokens: {e}")

def generate_speech(model, audio_tokenizer, input_seq, input_mask, task_name, args):
    """
    Generate speech from input sequence
    Using the reverse_delay approach that was proven to work in experiments
    """
    device = input_seq.device
    
    # Parameters
    use_sampling = True if args.inference_mode == "sampling" else False
    temp_text = args.temp_text
    top_k_text = args.topk
    temp = args.temp
    top_k = args.topk
    
    # Prepare for inference
    result = prepare_for_inference(input_seq, input_mask, task_name)
    
    # Check if prepare_for_inference returned None (error condition)
    if result is None or result[0] is None:
        logger.error("Failed to prepare sequence for inference.")
        return None
        
    prefix, mask, maxlen, minlen, gt_audio = result
    
    # Check for empty prefix
    if prefix.shape[2] == 0:
        logger.error("Empty prefix sequence. Cannot generate with no context.")
        return None
    
    # Prepare for generation
    final_results = []
    pre_gen_len = prefix.shape[2]
    
    logger.info(f"Starting generation with prefix shape: {prefix.shape}")
    logger.info(f"Generation parameters: maxlen={maxlen}, minlen={minlen}")
    
    try:
        # Set model to evaluation mode
        model.eval()
        
        # Generation loop
        for g_idx in range(maxlen):
            print(".", end="", flush=True)
            g_len = prefix.shape[2]
            
            # Global inference step
            with torch.no_grad():
                # Get initial token
                init_token = model._get_initial_token()
                init_token = init_token.expand(prefix.shape[0], -1, -1)
                global_prefix = torch.cat([init_token, prefix], dim=-1)
                
                # Forward pass through transformer
                transformer_out, text_logits = model.forward_global(global_prefix)
                
                # Add padding for next token
                local_pad = torch.ones_like(prefix[:, :, 0:1]) * model.initial_token_id
                prefix = torch.cat([prefix, local_pad], dim=-1)
                
                # Sample next text token
                valid_text_logits = text_logits[:, -1:, :]
                text_token = sample_token(
                    valid_text_logits.float(),
                    use_sampling,
                    temp_text,
                    top_k_text)
                prefix[:, 0, -1] = text_token.squeeze()
                
                # Generate audio tokens for each channel
                audio_seq = []
                flag = True
                
                for l_idx in range(8):
                    # Get text embeddings
                    text_indices = prefix[:, 0, :]
                    local_start_token = model.codecformer_text_emb(text_indices)
                    
                    # Forward pass through local model
                    sequence = prefix[:, 1:, :]
                    logits = model.forward_local(
                        local_start_token=local_start_token, 
                        sequence=sequence, 
                        transformer_out=transformer_out
                    )
                    
                    # Get logits for current channel
                    valid_logits = logits[:, -1:, l_idx:l_idx+1, :]
                    
                    # Sample audio token with adjusted temperature based on channel
                    # Use lower temperature for more stable generation
                    current_temp = temp if l_idx > 0 else temp * 0.9  # Slightly lower temp for first channel
                    
                    if g_len == pre_gen_len:
                        # First frame
                        next_token = sample_token_audio(
                            valid_logits.float(),
                            use_sampling,
                            current_temp,
                            top_k,
                        )
                    elif l_idx > 0 and g_len > minlen:
                        # Middle frames
                        next_token = sample_token_audio(
                            valid_logits.float(),
                            use_sampling,
                            current_temp,
                            top_k,
                        )
                    else:
                        # Special case
                        next_token = sample_token_audio_2048(
                            valid_logits.float(),
                            use_sampling,
                            current_temp,
                            top_k
                        )
                        
                    # End generation if we've reached a stopping condition
                    if (g_idx > minlen) and (l_idx > 2) and next_token[0, 0] >= 2048:
                        flag = False
                        break
                        
                    # Add token to audio sequence
                    audio_seq.append(next_token.squeeze().item())
                    prefix[:, l_idx+1, g_len] = next_token.squeeze()
                
                # End generation if we've reached a stopping condition
                if not flag:
                    break
                    
                # Add the generated audio frame - ensure it's a valid 8-channel frame
                if len(audio_seq) == 8:
                    audio_seq_tensor = torch.tensor(audio_seq, device=device)
                    final_results.append(audio_seq_tensor)
                else:
                    logger.warning(f"Generated incomplete audio frame with {len(audio_seq)} channels")
        
        print(" done!")
            
        # Stack all frames
        if final_results:
            # Stack as [frames, channels]
            final_results = torch.stack(final_results, dim=0).to(device)
            
            if args.debug_mode:
                inspect_audio_tokens(final_results, "Raw generated tokens")
                visualize_tokens_as_image(final_results, os.path.join(args.output_dir, "tokens_original.png"))
            
            # Apply REVERSE DELAY - key insight from experiments
            final_results = apply_reverse_delay(final_results)
            
            if args.debug_mode:
                inspect_audio_tokens(final_results, "After reverse delay")
                visualize_tokens_as_image(final_results, os.path.join(args.output_dir, "tokens_reverse_delay.png"))
            
            # Ensure tokens are in valid range before detokenization
            final_results = torch.clamp(final_results, 0, 2047)
            
            # Format for detokenizer: should be [8, seq_len]
            if final_results.dim() == 2 and final_results.shape[1] == 8:
                final_results = final_results.transpose(0, 1)
            
            if args.debug_mode:
                inspect_audio_tokens(final_results, "Final tokens for detokenization")
            
            return final_results
        else:
            logger.error("No audio frames were generated.")
            return None
            
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return None

def safe_detokenize(tokenizer, tokens, max_audio_length=10000):
    """
    Safely detokenize audio tokens with proper dimension handling to match training pipeline.
    
    Args:
        tokenizer: The MimiCodec tokenizer
        tokens: Audio tokens tensor
        max_audio_length: Maximum allowed audio length to prevent CUDA errors
        
    Returns:
        Detokenized audio waveform or None if detokenization fails
    """
    try:
        # Log original shape for debugging
        logger.info(f"Detokenizing tokens with shape: {tokens.shape}")
        
        # Validate token range (MimiCodec expects tokens in range [0, 2047])
        original_min = tokens.min().item()
        original_max = tokens.max().item()
        
        if original_min < 0 or original_max >= 2048:
            logger.warning(f"Tokens outside valid range: min={original_min}, max={original_max}")
            tokens = torch.clamp(tokens, 0, 2047)
            logger.info(f"Clipped to range: min={tokens.min().item()}, max={tokens.max().item()}")
        
        # Convert tokens to Long type (int64) for embedding lookup
        tokens = tokens.long()
        
        # Ensure tokens are in the format expected by MimiCodec: [8, sequence_length]
        if tokens.dim() == 2:
            if tokens.shape[0] == 8:
                # Already in correct format [8, seq_len]
                pass
            elif tokens.shape[1] == 8:
                # Need to transpose from [seq_len, 8] to [8, seq_len]
                tokens = tokens.transpose(0, 1)
                logger.info(f"Transposed tokens to shape: {tokens.shape}")
            else:
                logger.error(f"Unexpected token shape: {tokens.shape}. Expected 8 streams.")
                return None
        elif tokens.dim() == 3:
            # Handle 3D tensor (might be [batch=1, streams=8, seq_len] or [batch=1, seq_len, streams=8])
            if tokens.shape[0] == 1 and tokens.shape[1] == 8:
                tokens = tokens.squeeze(0)
            elif tokens.shape[0] == 1 and tokens.shape[2] == 8:
                tokens = tokens.squeeze(0).transpose(0, 1)
            else:
                logger.error(f"Unexpected 3D token shape: {tokens.shape}")
                return None
        
        # Limit sequence length to prevent OOM errors
        if tokens.shape[1] > max_audio_length:
            logger.warning(f"Truncating long audio sequence from {tokens.shape[1]} to {max_audio_length}")
            tokens = tokens[:, :max_audio_length]
        
        # Perform detokenization with error handling
        try:
            audio = tokenizer.detokenize(tokens)
            logger.info(f"Successfully detokenized to audio with shape: {audio.shape}")
            
            # Normalize audio (preventing too loud or too quiet output)
            audio = audio / (audio.abs().max() + 1e-8) * 0.9
            
        except RuntimeError as e:
            # Fall back to CPU if CUDA memory issues
            if "CUDA" in str(e):
                logger.warning("CUDA error in detokenization. Falling back to CPU...")
                cpu_tokens = tokens.detach().cpu()
                cpu_tokenizer = tokenizer.cpu()
                audio = cpu_tokenizer.detokenize(cpu_tokens)
                # Normalize audio
                audio = audio / (audio.abs().max() + 1e-8) * 0.9
                tokenizer.to(tokens.device)  # Move tokenizer back
            else:
                raise
        
        return audio
        
    except Exception as e:
        logger.error(f"Error during detokenization: {e}")
        import traceback
        traceback.print_exc()
        
        # Return a simple beep tone as fallback
        logger.warning("Generating fallback audio tone")
        sample_rate = 24000  # Use default sample rate
        duration = 1.0  # seconds
        t = torch.arange(0, duration, 1.0/sample_rate)
        # Generate a 440 Hz sine wave
        audio = 0.5 * torch.sin(2 * torch.pi * 440 * t).unsqueeze(0)
        return audio

def main():
    """Main function"""
    # Parse arguments
    parser = get_parser()
    args = parser.parse_args()
    
    # Set debug level if requested
    if args.debug_mode:
        logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)
        
    # Create output directory
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load model
    model, audio_tokenizer, device, train_args, llm_tokenizer = load_model(args)
    
    # Check if we need to process data from JSON
    if args.data_json:
        # Use data JSON for inference
        try:
            logger.info(f"Loading data from {args.data_json}")
            
            # Get data iterator
            _, valid_iter = get_data_iterator_tokenizer_vocabulary(
                args=train_args,
                train_jsons=[],
                valid_jsons=[args.data_json],
                delay_step=1,  # Match the delay step used in training
                batch_scale=train_args.batch_scale,
                minibatch_debug=train_args.minibatch_debug,
                max_length=train_args.max_length,
                min_length=train_args.min_length,
                parallel_number=train_args.parallel_number,
                text_empty_token=128002,
                semantic_empty_token=2048,
                acoustic_empty_token=2048,
                acoustic_pad_token=2049,
                semantic_pad_token=2049,
                text_pad_token=128003
            )
            
            # Process batches
            for b_idx, batch in enumerate(valid_iter):
                logger.info(f"Processing batch {b_idx+1}")
                seqs, masks, lengths, example_ids = to_device(batch, device=device, non_blocking=False)
                
                for i_idx in range(len(seqs)):
                    # Generate speech
                    searched_results = generate_speech(
                        model, audio_tokenizer, seqs[i_idx], masks[i_idx], args.task_name, args
                    )
                    
                    if searched_results is not None:
                        if args.generate_target == 'audio':
                            # Convert to audio and save
                            detokenized = safe_detokenize(audio_tokenizer, searched_results)
                            detokenized = detokenized.detach().cpu()
                            file_name = f"{example_ids[i_idx]}_sample.wav"
                            file_path = os.path.join(output_dir, file_name)
                            logger.info(f"Saving audio to {file_path}")
                            torchaudio.save(file_path, detokenized, 24000, bits_per_sample=16, encoding='PCM_S')
                
                if b_idx >= 5:
                    break
                    
        except Exception as e:
            logger.error(f"Error processing data JSON: {e}")
            import traceback
            traceback.print_exc()
    
    # Interactive mode
    elif args.interactive:
        logger.info("Running in interactive mode")
        print("\n" + "=" * 60)
        print("    eSpeak Speech AI - Interactive Mode")
        print("=" * 60)
        print("Type your message and press Enter. The AI will respond with speech.")
        print("Type 'exit', 'quit', or just 'q' to exit.")
        print("=" * 60 + "\n")
        
        count = 1
        
        while True:
            try:
                # Get user input
                text = input("\nYou: ").strip()
                
                # Check for exit
                if text.lower() in ['exit', 'quit', 'q', '']:
                    print("\nExiting. Goodbye!")
                    break
                
                # Create input sequence
                input_seq, input_mask = create_input_sequence(text, device, llm_tokenizer)
                
                # Generate speech
                print("Generating speech...", end="", flush=True)
                audio_tokens = generate_speech(model, audio_tokenizer, input_seq, input_mask, 'TTS', args)
                
                if audio_tokens is not None:
                    # Visualize tokens if in debug mode
                    if args.debug_mode:
                        visualize_tokens_as_image(
                            audio_tokens, 
                            output_file=os.path.join(output_dir, f"tokens_{count}.png")
                        )
                    
                    # Convert to audio and save
                    audio_output = safe_detokenize(audio_tokenizer, audio_tokens)
                    audio_output = audio_output.detach().cpu()
                    
                    # Create unique filename
                    output_file = os.path.join(output_dir, f"response_{count}.wav")
                    count += 1
                    
                    # Save audio
                    torchaudio.save(output_file, audio_output, 24000, bits_per_sample=16, encoding='PCM_S')
                    print(f"\nSpeech generated and saved to {output_file}")
                else:
                    print("\nFailed to generate speech.")
                    
            except KeyboardInterrupt:
                print("\nExiting. Goodbye!")
                break
                
            except Exception as e:
                print(f"\nError: {e}")
                print("Please try again.")
    # Single text mode
    elif args.text:
        logger.info(f"Converting text to speech: '{args.text}'")
        
        # Create input sequence
        input_seq, input_mask = create_input_sequence(args.text, device, llm_tokenizer)
        
        # Generate speech
        print("Generating speech...", end="", flush=True)
        audio_tokens = generate_speech(model, audio_tokenizer, input_seq, input_mask, 'TTS', args)
        
        if audio_tokens is not None:
            # Make sure audio tokens are valid before detokenization
            logger.info(f"Audio tokens shape: {audio_tokens.shape}")
            logger.info(f"Audio tokens min: {audio_tokens.min().item()}, max: {audio_tokens.max().item()}")
            
            # Visualize tokens if in debug mode
            if args.debug_mode:
                visualize_tokens_as_image(
                    audio_tokens, 
                    output_file=os.path.join(output_dir, "tokens.png")
                )
            
            # Convert to audio and save
            audio_output = safe_detokenize(audio_tokenizer, audio_tokens)
            audio_output = audio_output.detach().cpu()
            
            # Create output file
            output_file = os.path.join(output_dir, "output.wav")
            
            # Save audio
            torchaudio.save(output_file, audio_output, 24000, bits_per_sample=16, encoding='PCM_S')
            print(f"\nSpeech generated and saved to {output_file}")
        else:
            print("\nFailed to generate speech.")
    
    else:
        logger.error("No input provided. Use --text, --interactive, or --data_json")

if __name__ == "__main__":
    main()
