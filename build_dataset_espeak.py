#!/usr/bin/env python3
# Modified version of build_dataset_espeak.py to use Natural Questions dataset
# Usage: python build_dataset_espeak.py --output_dir ./natural_questions_dataset --target_hours 2.0 --voice "en-us+m3"

import os
import subprocess
import json
import random
import tqdm
import argparse
import time
import shutil
import wave
import requests
from datasets import load_dataset

def install_dependencies():
    """Install required dependencies"""
    try:
        subprocess.run(["espeak", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("espeak is already installed")
    except FileNotFoundError:
        print("Installing espeak...")
        subprocess.run(["apt-get", "update"], check=True)
        subprocess.run(["apt-get", "install", "-y", "espeak"], check=True)
        print("espeak installed successfully")
    
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("ffmpeg is already installed")
    except FileNotFoundError:
        print("Installing ffmpeg...")
        subprocess.run(["apt-get", "install", "-y", "ffmpeg"], check=True)
        print("ffmpeg installed successfully")

def get_response_from_ollama(prompt, model="llama3.2:1b", max_tokens=50):
    """Get a response from Ollama API"""
    try:
        response = requests.post('http://localhost:11434/api/generate', 
                               json={
                                   'model': model,
                                   'prompt': prompt,
                                   'max_tokens': max_tokens
                               })
        
        if response.status_code == 200:
            # Parse the streaming response
            full_text = ""
            for line in response.text.strip().split('\n'):
                if line:
                    try:
                        data = json.loads(line)
                        if 'response' in data:
                            full_text += data['response']
                    except json.JSONDecodeError:
                        pass
            
            return full_text.strip()
        else:
            print(f"Error from Ollama API: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        return None

def extract_first_n_words(text, n=100):
    """Extract the first n words from a text"""
    words = text.split()
    return " ".join(words[:n])

def generate_clean_audio(text, output_path, voice="en-us", speed=150):
    """Generate clean speech using espeak and ffmpeg"""
    temp_path = output_path + ".temp.wav"
    
    try:
        # Generate speech with espeak
        subprocess.run(
            ["espeak", "-v", voice, "-s", str(speed), "-w", temp_path, text],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        
        # Convert to exact format required
        subprocess.run([
            "ffmpeg", "-y", 
            "-i", temp_path,
            "-ar", "24000",     # 24kHz sample rate 
            "-ac", "1",         # mono
            "-c:a", "pcm_s16le", # 16-bit PCM
            "-f", "wav",        # Force WAV format
            "-bitexact",        # Use bitexact encoding
            output_path
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Validate the output with wave module
        try:
            with wave.open(output_path, 'rb') as wav_file:
                if wav_file.getframerate() != 24000 or wav_file.getnchannels() != 1 or wav_file.getsampwidth() != 2:
                    print(f"Generated file has incorrect format: {wav_file.getframerate()=}, {wav_file.getnchannels()=}, {wav_file.getsampwidth()=}")
                    return False
                
                # Verify file has actual content
                if wav_file.getnframes() == 0:
                    print(f"Generated file has no frames")
                    return False
        except Exception as e:
            print(f"Wave validation failed: {e}")
            return False
            
        os.remove(temp_path)
        return True
    
    except Exception as e:
        print(f"Error in audio generation: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False

def create_tar_scp_file(audio_files, output_dir):
    """Create tar.scp file which points to audio files"""
    tar_scp_path = os.path.join(output_dir, "tar.scp")
    with open(tar_scp_path, "w") as f:
        for audio_id, audio_path in audio_files.items():
            full_path = os.path.abspath(audio_path)
            f.write(f"{audio_id} {full_path}\n")
    return tar_scp_path

def create_tar_info_scp_file(metadata, output_dir):
    """Create tar_info.scp file with metadata about the samples"""
    tar_info_path = os.path.join(output_dir, "tar_info.scp")
    with open(tar_info_path, "w") as f:
        for item in metadata:
            # Format: utterance_id transcript
            f.write(f"{item['id']} {item['text']}\n")
    return tar_info_path

def load_complex_web_questions(num_samples=5000):
    """Load questions from the Complex Web Questions dataset"""
    print("Loading Complex Web Questions dataset...")
    
    # We need to specify the config name 'complex_web_questions'
    dataset = load_dataset("drt/complex_web_questions", "complex_web_questions", split="train")
    print(f"Loaded {len(dataset)} examples from Complex Web Questions dataset")
    
    # Extract questions from the dataset
    questions = []
    for item in tqdm.tqdm(dataset, desc="Extracting questions"):
        try:
            question_text = item['question']
            if isinstance(question_text, str) and len(question_text) > 5:
                questions.append(question_text)
        except (KeyError, TypeError):
            continue
    
    print(f"Extracted {len(questions)} questions from Complex Web Questions dataset")
    
    # Shuffle and limit to num_samples
    random.shuffle(questions)
    return questions[:num_samples]

def main(args):
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_dir = os.path.join(args.output_dir, "train")
    val_dir = os.path.join(args.output_dir, "val")
    test_dir = os.path.join(args.output_dir, "test")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    audio_dir = os.path.join(args.output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    
    # Make sure dependencies are installed
    install_dependencies()
    
    # Load questions from Complex Web Questions dataset
    questions = load_complex_web_questions(args.total_samples)
    
    print(f"Total questions collected: {len(questions)}")
    
    # Check if we have enough questions
    if len(questions) < args.total_samples:
        print(f"Warning: Only found {len(questions)} questions, fewer than requested {args.total_samples}")
        args.total_samples = len(questions)
    
    # Calculate train/val/test splits
    train_samples = int(args.total_samples * 0.8)
    val_samples = int(args.total_samples * 0.1)
    test_samples = args.total_samples - train_samples - val_samples
    
    print(f"Creating {train_samples} training samples...")
    print(f"Creating {val_samples} validation samples...")
    print(f"Creating {test_samples} test samples...")
    
    # Process and save audio files
    metadata = {
        "train": [],
        "val": [],
        "test": []
    }
    
    audio_files = {
        "train": {},
        "val": {},
        "test": {}
    }
    
    # Track audio duration and successful generations
    total_duration = 0.0
    successful_train = 0
    successful_val = 0
    successful_test = 0
    target_hours = args.target_hours
    target_seconds = target_hours * 3600
    
    questions_used = 0
    voice = args.voice if args.voice else "en-us"
    
    # Generate training samples
    print("Generating training samples...")
    for i, question in enumerate(tqdm.tqdm(questions[:train_samples])):
        # Get response from Ollama
        response = get_response_from_ollama(question)
        if not response:
            print(f"Failed to get response for: {question}")
            continue
        
        # Take first few words of response
        first_words = extract_first_n_words(response, args.words_per_response)
        if not first_words:
            print(f"No words found in response for: {question}")
            continue
        
        sample_id = f"nq_train_{i:06d}"
        output_path = os.path.join(audio_dir, f"{sample_id}.wav")
        
        if generate_clean_audio(first_words, output_path, voice=voice, speed=args.speed):
            # Get duration using ffprobe
            try:
                result = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", 
                     "default=noprint_wrappers=1:nokey=1", output_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                duration = float(result.stdout.strip())
                total_duration += duration
            except:
                # If ffprobe fails, estimate duration based on text length
                duration = len(first_words) * 0.06  # rough estimate: 60ms per character
                total_duration += duration
            
            metadata["train"].append({
                "id": sample_id,
                "text": question,
                "response_start": first_words  # Store the text used for audio generation
            })
            
            audio_files["train"][sample_id] = output_path
            questions_used += 1
            successful_train += 1
            
            # Check if we've reached our target duration
            if total_duration >= target_seconds:
                print(f"Reached target duration of {target_hours} hours ({total_duration/3600:.2f} hours)")
                break
        
        # Small delay to avoid hammering Ollama API
        time.sleep(0.1)
    
    # If we still need more audio, continue with validation samples
    if total_duration < target_seconds and questions_used < len(questions):
        val_questions = questions[train_samples:train_samples+val_samples]
        print(f"Generating {len(val_questions)} validation samples...")
        
        for i, question in enumerate(tqdm.tqdm(val_questions)):
            # Get response from Ollama
            response = get_response_from_ollama(question)
            if not response:
                print(f"Failed to get response for: {question}")
                continue
            
            # Take first few words of response
            first_words = extract_first_n_words(response, args.words_per_response)
            if not first_words:
                print(f"No words found in response for: {question}")
                continue
            
            sample_id = f"nq_val_{i:06d}"
            output_path = os.path.join(audio_dir, f"{sample_id}.wav")
            
            if generate_clean_audio(first_words, output_path, voice=voice, speed=args.speed):
                try:
                    result = subprocess.run(
                        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", 
                         "default=noprint_wrappers=1:nokey=1", output_path],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    duration = float(result.stdout.strip())
                    total_duration += duration
                except:
                    duration = len(first_words) * 0.06
                    total_duration += duration
                
                metadata["val"].append({
                    "id": sample_id,
                    "text": question,
                    "response_start": first_words
                })
                
                audio_files["val"][sample_id] = output_path
                questions_used += 1
                successful_val += 1
                
                # Check if we've reached our target duration
                if total_duration >= target_seconds:
                    print(f"Reached target duration of {target_hours} hours ({total_duration/3600:.2f} hours)")
                    break
            
            # Small delay to avoid hammering Ollama API
            time.sleep(0.1)
    
    # If we still need more audio, continue with test samples
    if total_duration < target_seconds and questions_used < len(questions):
        test_questions = questions[train_samples+val_samples:args.total_samples]
        print(f"Generating {len(test_questions)} test samples...")
        
        for i, question in enumerate(tqdm.tqdm(test_questions)):
            # Get response from Ollama
            response = get_response_from_ollama(question)
            if not response:
                print(f"Failed to get response for: {question}")
                continue
            
            # Take first few words of response
            first_words = extract_first_n_words(response, args.words_per_response)
            if not first_words:
                print(f"No words found in response for: {question}")
                continue
            
            sample_id = f"nq_test_{i:06d}"
            output_path = os.path.join(audio_dir, f"{sample_id}.wav")
            
            if generate_clean_audio(first_words, output_path, voice=voice, speed=args.speed):
                try:
                    result = subprocess.run(
                        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", 
                         "default=noprint_wrappers=1:nokey=1", output_path],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    duration = float(result.stdout.strip())
                    total_duration += duration
                except:
                    duration = len(first_words) * 0.06
                    total_duration += duration
                
                metadata["test"].append({
                    "id": sample_id,
                    "text": question,
                    "response_start": first_words
                })
                
                audio_files["test"][sample_id] = output_path
                questions_used += 1
                successful_test += 1
                
                # Check if we've reached our target duration
                if total_duration >= target_seconds:
                    print(f"Reached target duration of {target_hours} hours ({total_duration/3600:.2f} hours)")
                    break
            
            # Small delay to avoid hammering Ollama API
            time.sleep(0.1)
    
    # Create tar.scp and tar_info.scp files
    create_tar_scp_file(audio_files["train"], train_dir)
    create_tar_info_scp_file(metadata["train"], train_dir)
    
    create_tar_scp_file(audio_files["val"], val_dir)
    create_tar_info_scp_file(metadata["val"], val_dir)
    
    create_tar_scp_file(audio_files["test"], test_dir)
    create_tar_info_scp_file(metadata["test"], test_dir)
    
    # Save metadata as JSON for reference
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump({
            "train": metadata["train"],
            "val": metadata["val"],
            "test": metadata["test"]
        }, f, indent=2)
    
    # Create split directories for ngpu
    train_splits_dir = os.path.join(train_dir, f"{args.ngpu}splits_2")
    os.makedirs(train_splits_dir, exist_ok=True)
    os.makedirs(os.path.join(train_splits_dir, "log"), exist_ok=True)
    
    val_splits_dir = os.path.join(val_dir, f"{args.ngpu}splits_2")
    os.makedirs(val_splits_dir, exist_ok=True)
    os.makedirs(os.path.join(val_splits_dir, "log"), exist_ok=True)
    
    test_splits_dir = os.path.join(test_dir, f"{args.ngpu}splits_2") 
    os.makedirs(test_splits_dir, exist_ok=True)
    os.makedirs(os.path.join(test_splits_dir, "log"), exist_ok=True)
    
    # Copy scp files to splits directory for 1-GPU
    if args.ngpu == 1:
        shutil.copy(os.path.join(train_dir, "tar.scp"), os.path.join(train_splits_dir, "tar.1.scp"))
        shutil.copy(os.path.join(train_dir, "tar_info.scp"), os.path.join(train_splits_dir, "tar_info.1.scp"))
        
        shutil.copy(os.path.join(val_dir, "tar.scp"), os.path.join(val_splits_dir, "tar.1.scp"))
        shutil.copy(os.path.join(val_dir, "tar_info.scp"), os.path.join(val_splits_dir, "tar_info.1.scp"))
        
        shutil.copy(os.path.join(test_dir, "tar.scp"), os.path.join(test_splits_dir, "tar.1.scp"))
        shutil.copy(os.path.join(test_dir, "tar_info.scp"), os.path.join(test_splits_dir, "tar_info.1.scp"))
    else:
        # Here we would split the files for multiple GPUs if needed
        pass
    
    print(f"\nDataset created at {args.output_dir}")
    print(f"Training samples: {successful_train}")
    print(f"Validation samples: {successful_val}")
    print(f"Test samples: {successful_test}")
    print(f"Total audio duration: {total_duration/3600:.2f} hours")
    
    if total_duration/3600 < target_hours:
        print(f"WARNING: Collected {total_duration/3600:.2f} hours of audio, which is less than the target {target_hours} hours")
        print("Consider increasing the total_samples parameter")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate speech dataset using Complex Web Questions and Ollama responses")
    parser.add_argument("--output_dir", type=str, default="complex_web_questions_dataset", 
                        help="Directory to save the dataset")
    parser.add_argument("--total_samples", type=int, default=5000, 
                        help="Maximum number of samples to generate")
    parser.add_argument("--target_hours", type=float, default=2.0, 
                        help="Target hours of audio to generate")
    parser.add_argument("--ngpu", type=int, default=1, 
                        help="Number of GPUs to split data for")
    parser.add_argument("--voice", type=str, default="en-us+m3", 
                        help="espeak voice to use (en-us+m3 for male voice)")
    parser.add_argument("--speed", type=int, default=150, 
                        help="Speech rate (words per minute)")
    parser.add_argument("--words_per_response", type=int, default=100,
                        help="Number of words to take from each response")
    
    args = parser.parse_args()
    main(args)
