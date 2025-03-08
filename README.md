# Espeak Speech AI

[![Status](https://img.shields.io/badge/Status-Experimental-yellow.svg)](https://github.com/yourusername/espeak_speech_ai)


python build_dataset_espeak.py \
  --output_dir ./complex_web_questions_dataset \
  --total_samples 27639 \
  --target_hours 10 \
  --voice "en-us+m3" \


#python tokenizer.py --analyze --metadata_path ./complex_web_questions_dataset/metadata.json --audio_dir ./complex_web_questions_dataset/audio
python tokenizer.py --metadata_path ./complex_web_questions_dataset/metadata.json --audio_dir ./complex_web_questions_dataset/audio
bash ./train.sh
bash ./inference.sh "Say 'potato'" test.wav


No longer based on RSTnet
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

## License

This project under the GPL3
