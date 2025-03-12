# Espeak Speech AI

[![Status](https://img.shields.io/badge/Status-Experimental-yellow.svg)](https://github.com/yourusername/espeak_speech_ai)
## Overview

Espeak Speech AI allows you to build synthetic speech datasets using eSpeak and finetune an LM for audio output via MIMI. 

## Setup

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for training)
- eSpeak installed on your system

### Environment Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/ruapotato/espeak_speech_ai.git
   cd espeak_speech_ai
   ```

2. Create a virtual environment:
   ```bash
   python3 -m venv pyenv
   source ./pyenv/bin/activate
   ```

3. Install required packages:
   ```bash
   # PyTorch (with CUDA support)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   
   # Core dependencies
   pip install librosa==0.9.1
   pip install tqdm matplotlib omegaconf einops
   pip install vector_quantize_pytorch
   pip install tensorboard deepspeed peft
   pip install transformers
   pip install datasets[audio]
   pip install soundfile
   pip install webdataset
   pip install -r requirements.txt
   ```

4. Follow https://huggingface.co/kyutai/mimi setup steps


Use
---

# Build espeak data
python build_dataset_espeak.py \
  --output_dir ./complex_web_questions_dataset \
  --total_samples 27639 \
  --target_hours 10 \
  --voice "en-us+m3" \

# Build dataset for decoder model
python codebook-dataset-generator.py --input_dir ./gutenberg_espeak_dataset_clean/audio --output_dir ./codebook_dataset

# CPU based
python codebook-decoder-model_old.py --data_dir ./codebook_dataset --output_dir ./big_mapper_model --embed_dim 256 --hidden_dim 512 --epochs 3
Use inference-pipeline_old.py if trained this way

# GPU based
python codebook-decoder-model.py --data_dir ./codebook_dataset --output_dir ./big_mapper_model --embed_dim 256 --hidden_dim 512 --epochs 20 --batch_size 32

# Test output.wav sounds like gutenberg_train_000002.wav
python codebook-zero-decoder.py --input_file ./gutenberg_espeak_dataset_clean/audio/gutenberg_train_000002.wav --output_dir ./output --save_codebooks
python inference-pipeline.py --model_path ./big_mapper_model/checkpoint_epoch_19.pt --input_file ./output/zeroth_codebook.txt --output_file output.wav --use_gpu


# pull model to finetune
litgpt download Qwen/Qwen2.5-0.5B-Instruct


# TODO update to only train on 0th codebook
#python tokenizer.py --analyze --metadata_path ./complex_web_questions_dataset/metadata.json --audio_dir ./complex_web_questions_dataset/audio
python tokenizer.py --metadata_path ./complex_web_questions_dataset/metadata.json --audio_dir ./complex_web_questions_dataset/audio
bash ./train.sh
generate ./audio_lm_model/final --prompt "Generate speech audio for the following text.\n\nInput: Test.\n\nOutput:" --max_new_tokens 2000 --temperature 0.8
bash ./inference.sh "Say 'potato'" test.wav


No longer based on RSTnet


Conversational fine tune
------------------------

cd ./text_finetune/
python ./download_chat_data.py 
litgpt download  mistralai/Mistral-7B-Instruct-v0.3
#rm -r out/chat-finetuned #(If retraining)

litgpt finetune_lora mistralai/Mistral-7B-Instruct-v0.3 --data JSON --data.json_path data/chat_instructions.json --data.val_split_fraction 0.05 --precision bf16-true --quantize "bnb.nf4" --train.micro_batch_size 2 --train.global_batch_size 128 --train.max_steps 1000 --train.lr_warmup_steps 100 --optimizer AdamW --optimizer.learning_rate 2e-5 --eval.interval 100 --train.save_interval 100 --out_dir out/chat-finetuned --optimizer AdamW

litgpt chat out/chat-finetuned/final/

## License

This project under the GPL3
