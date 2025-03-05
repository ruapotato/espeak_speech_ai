#!/usr/bin/env python3
#python ./build_dataset_espeak_clean.py --output_dir ./gutenberg_espeak_dataset_clean --target_hours 7.0 --voice "en-us+m3"
import os
import subprocess
import json
import random
import tqdm
import argparse
import re
import requests
import nltk
from nltk.tokenize import sent_tokenize
import time
import shutil
import wave

def install_nltk_packages():
    """Install required NLTK packages"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

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

def get_gutenberg_book(book_id):
    """Download a book from Project Gutenberg by ID"""
    url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        # Try alternative URL format
        url = f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt"
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        # Try another alternative
        url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        print(f"Failed to download book {book_id}, status code: {response.status_code}")
        return None
    except Exception as e:
        print(f"Error downloading book {book_id}: {e}")
        return None

def clean_gutenberg_text(text):
    """Clean the Gutenberg text, removing headers, footers, and chapter markers"""
    # Find the start of the book (after the header)
    start_markers = [
        "*** START OF THIS PROJECT GUTENBERG EBOOK",
        "*** START OF THE PROJECT GUTENBERG EBOOK",
        "***START OF THE PROJECT GUTENBERG EBOOK",
        "*END*THE SMALL PRINT"
    ]
    
    start_idx = 0
    for marker in start_markers:
        if marker in text:
            parts = text.split(marker, 1)
            if len(parts) > 1:
                text = parts[1]
                break
    
    # Find the end of the book (before the footer)
    end_markers = [
        "*** END OF THIS PROJECT GUTENBERG EBOOK",
        "*** END OF THE PROJECT GUTENBERG EBOOK",
        "***END OF THE PROJECT GUTENBERG EBOOK",
        "End of the Project Gutenberg EBook",
        "End of Project Gutenberg's"
    ]
    
    for marker in end_markers:
        if marker in text:
            parts = text.split(marker, 1)
            if len(parts) > 1:
                text = parts[0]
                break
    
    # Remove chapter headings and other formatting
    text = re.sub(r'\r', '', text)  # Remove carriage returns
    text = re.sub(r'\n\n+', '\n\n', text)  # Normalize multiple newlines
    
    return text

def extract_good_sentences(text, min_length=30, max_length=200):
    """Extract well-formed sentences from text"""
    sentences = sent_tokenize(text)
    good_sentences = []
    
    for sentence in sentences:
        # Clean up whitespace
        sentence = re.sub(r'\s+', ' ', sentence).strip()
        
        # Check length and structure
        if min_length <= len(sentence) <= max_length and sentence.endswith(('.', '?', '!')):
            # Check for balanced quotes and parentheses
            if sentence.count('"') % 2 == 0 and sentence.count('(') == sentence.count(')'):
                # Avoid sentences with unusual formatting or specialized content
                if not re.search(r'[^\w\s.,;:!?()"\'-]', sentence):
                    good_sentences.append(sentence)
    
    return good_sentences

def download_books(book_ids):
    """Download and process multiple books"""
    all_sentences = []
    
    for book_id in tqdm.tqdm(book_ids, desc="Downloading books"):
        print(f"Downloading book {book_id}...")
        text = get_gutenberg_book(book_id)
        if text:
            print(f"Cleaning text for book {book_id}...")
            clean_text = clean_gutenberg_text(text)
            print(f"Extracting sentences from book {book_id}...")
            sentences = extract_good_sentences(clean_text)
            print(f"Found {len(sentences)} good sentences in book {book_id}")
            all_sentences.extend(sentences)
            # Slight delay to avoid hammering the server
            time.sleep(1)
    
    return all_sentences


def generate_clean_audio(text, output_path, voice="en-us", speed=150):
    """Generate clean speech using espeak and ffmpeg to ensure compatibility with MimiCodec"""
    temp_path = output_path + ".temp.wav"
    
    try:
        # Generate speech with espeak
        subprocess.run(
            ["espeak", "-v", voice, "-s", str(speed), "-w", temp_path, text],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        
        # Convert to exact format required by MimiCodec
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
    install_nltk_packages()
    
    # Define book IDs to download (classics with good public domain texts)
    # 1342: Pride and Prejudice, 84: Frankenstein, 11: Alice in Wonderland,
    # 1661: The Adventures of Sherlock Holmes, 1952: The Yellow Wallpaper
    book_ids = args.book_ids if args.book_ids else [1342, 84, 11, 1661, 1952, 2701, 98, 1400, 345, 1080]
    print("Will download and process these books:", book_ids)
    
    # Download and process books
    all_sentences = download_books(book_ids)
    random.shuffle(all_sentences)
    
    print(f"Total sentences collected: {len(all_sentences)}")
    
    # Check if we have enough sentences
    if len(all_sentences) < args.total_samples:
        print(f"Warning: Only found {len(all_sentences)} sentences, fewer than requested {args.total_samples}")
        args.total_samples = len(all_sentences)
    
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
    
    sentences_used = 0
    voice = args.voice if args.voice else "en-us"
    
    # Generate training samples
    print("Generating training samples...")
    for i, text in enumerate(tqdm.tqdm(all_sentences[:train_samples])):
        sample_id = f"gutenberg_train_{i:06d}"
        output_path = os.path.join(audio_dir, f"{sample_id}.wav")
        
        if generate_clean_audio(text, output_path, voice=voice, speed=args.speed):
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
                duration = len(text) * 0.06  # rough estimate: 60ms per character
                total_duration += duration
            
            metadata["train"].append({
                "id": sample_id,
                "text": text
            })
            
            audio_files["train"][sample_id] = output_path
            sentences_used += 1
            successful_train += 1
            
            # Check if we've reached our target duration
            if total_duration >= target_seconds:
                print(f"Reached target duration of {target_hours} hours ({total_duration/3600:.2f} hours)")
                break
    
    # If we still need more audio, continue with validation and test samples
    if total_duration < target_seconds and sentences_used < len(all_sentences):
        remaining_sentences = all_sentences[sentences_used:]
        val_test_split = int(len(remaining_sentences) * 0.5)
        
        # Generate validation samples
        val_sentences = remaining_sentences[:val_test_split]
        print(f"Generating {len(val_sentences)} validation samples...")
        for i, text in enumerate(tqdm.tqdm(val_sentences)):
            sample_id = f"gutenberg_val_{i:06d}"
            output_path = os.path.join(audio_dir, f"{sample_id}.wav")
            
            if generate_clean_audio(text, output_path, voice=voice, speed=args.speed):
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
                    duration = len(text) * 0.06
                    total_duration += duration
                
                metadata["val"].append({
                    "id": sample_id,
                    "text": text
                })
                
                audio_files["val"][sample_id] = output_path
                successful_val += 1
                
                # Check if we've reached our target duration
                if total_duration >= target_seconds:
                    print(f"Reached target duration of {target_hours} hours ({total_duration/3600:.2f} hours)")
                    break
        
        # Generate test samples
        test_sentences = remaining_sentences[val_test_split:]
        print(f"Generating {len(test_sentences)} test samples...")
        for i, text in enumerate(tqdm.tqdm(test_sentences)):
            if total_duration >= target_seconds:
                break
                
            sample_id = f"gutenberg_test_{i:06d}"
            output_path = os.path.join(audio_dir, f"{sample_id}.wav")
            
            if generate_clean_audio(text, output_path, voice=voice, speed=args.speed):
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
                    duration = len(text) * 0.06
                    total_duration += duration
                
                metadata["test"].append({
                    "id": sample_id,
                    "text": text
                })
                
                audio_files["test"][sample_id] = output_path
                successful_test += 1
                
                # Check if we've reached our target duration
                if total_duration >= target_seconds:
                    print(f"Reached target duration of {target_hours} hours ({total_duration/3600:.2f} hours)")
                    break
    
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
        print("Consider adding more books or increasing the total_samples parameter")
    
    # Print instructions for using with RSTnet
    print("\nInstructions for using with RSTnet:")
    print("1. Update your extract_token.sh script to use the MimiCodec tokenizer:")
    print("   --tokenizer mimi instead of --tokenizer ssl")
    print(f"2. Set DATA_ROOT='{os.path.abspath(args.output_dir)}' in extract_token.sh")
    print("3. Run extract_token.sh to process the dataset")
    print("4. Run the training script with the processed dataset")

if __name__ == "__main__":
    import shutil
    
    parser = argparse.ArgumentParser(description="Generate clean synthetic speech dataset using espeak with Gutenberg books")
    parser.add_argument("--output_dir", type=str, default="gutenberg_espeak_dataset_clean", 
                        help="Directory to save the dataset")
    parser.add_argument("--total_samples", type=int, default=20000, 
                        help="Maximum number of samples to generate")
    parser.add_argument("--target_hours", type=float, default=7.0, 
                        help="Target hours of audio to generate")
    parser.add_argument("--ngpu", type=int, default=1, 
                        help="Number of GPUs to split data for")
    parser.add_argument("--voice", type=str, default="en-us+m3", 
                        help="espeak voice to use (en-us+m3 for male voice)")
    parser.add_argument("--speed", type=int, default=150, 
                        help="Speech rate (words per minute)")
    parser.add_argument("--book_ids", type=int, nargs="+",
                        help="Specific Project Gutenberg book IDs to use")
    
    args = parser.parse_args()
    main(args)
