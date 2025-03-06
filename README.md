# Espeak Speech AI

[![Status](https://img.shields.io/badge/Status-Experimental-yellow.svg)](https://github.com/yourusername/espeak_speech_ai)

A fork of [RSTnet](https://github.com/yangdongchao/RSTnet) modified to work with eSpeak-generated speech datasets.

## Overview

Espeak Speech AI allows you to build synthetic speech datasets using eSpeak and train speech-text foundation models on this data. It provides scripts for generating, processing, and tokenizing speech data to be used with RSTnet's speech-text foundation model architecture.

## Features

- Generate synthetic speech datasets using eSpeak text-to-speech
- Process and tokenize audio files with fixed compatibility for various WAV formats
- Train speech-text foundation models with the processed dataset
- Leverage the capabilities of the original RSTnet architecture

## Setup

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for training)
- eSpeak installed on your system

### Environment Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/espeak_speech_ai.git
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

## Project Structure

```
espeak_speech_ai/
├── RSTnet/                     # Original RSTnet codebase
│   ├── AudioCodec/             # Audio codec components
│   ├── DataPipeline/           # Data processing pipeline
│   ├── Evaluation/             # Evaluation scripts
│   ├── MLLM/                   # Original MLLM implementation
│   └── MLLM_v2/                # Improved MLLM implementation
│       ├── configs/            # Configuration files
│       ├── egs/                # Example scripts and training recipes
│       │   └── pretraining/    # Pretraining scripts
│       │       ├── data_scripts/     # Data processing scripts
│       │       ├── extract_token.sh  # Script for audio tokenization
│       │       ├── local/            # Local utility scripts
│       │       │   └── offline_codec_tokenization_fixed.py  # Fixed MimiCodec tokenization
│       │       ├── run.sh            # Main training script
│       │       └── infer.sh          # Inference script
│       ├── models/             # Model implementations
│       └── modules/            # Model components
├── build_dataset_espeak.py     # Script to generate eSpeak datasets
├── fix_wav_files.sh            # Script to fix WAV file format issues
├── gutenberg_espeak_dataset_clean/  # Generated eSpeak dataset
├── checkpoints/                # Model checkpoints directory (to be populated)
└── README.md                   # This documentation
```

## Current Progress and Workflow

This project is in its early stages. Currently, steps 1 and 2 (dataset creation and data preprocessing) have been implemented.

### Multi-modal LLM (Speech-Text Foundation Models)

The intended workflow follows the RSTnet approach for training speech-text foundation models:

### Step 0: LLM Checkpoint Preparation
Download the Llama-3.2 model checkpoint using:

```bash
litgpt download meta-llama/Llama-3.2-1B-Instruct
```

### Step 1: Dataset Creation (Implemented)
Generate a synthetic speech dataset using eSpeak:

```bash
python build_dataset_espeak.py --output_dir ./gutenberg_espeak_dataset_clean --target_hours 20.0 --voice "en-us+m3"
```

Parameters:
- `--output_dir`: Directory to save the generated dataset
- `--target_hours`: Target dataset size in hours
- `--voice`: eSpeak voice to use for synthesis

If you encounter WAV file compatibility issues, use:
```bash
bash ./fix_wav_files.sh
```

### Step 2: Data Preprocessing (Implemented/Redoing)
Process and tokenize the generated dataset using:
```bash
cd RSTnet/MLLM_v2/egs/pretraining
bash ./extract_token.sh
```

This script handles:
- Audio format standardization
- Tokenization using MimiCodec
- Text tokenization with LLM
- Creation of data JSON files for training
- Preparation for model training

Sample output:
```
2025-03-05 11:43:25,956 INFO [offline_codec_tokenization_fixed.py:40] max gpu 1
2025-03-05 11:43:25,957 INFO [offline_codec_tokenization_fixed.py:44] Using device: cuda:0
2025-03-05 11:43:26,662 INFO [offline_codec_tokenization_fixed.py:48] tokenizer built
...
Processing text tokenization with LLM...
...
Creating data JSON files...
Creating data JSON for train split 1...
Creating data JSON for val split 1...
Creating data JSON for test split 1...
```

### Step 3: Model Pre-training (Implemented)
Train the model using RSTnet training scripts:
```bash
cd RSTnet/MLLM_v2/egs/pretraining
# For training on about 15 GB vram
bash ./run_memory_optimized.sh #Tested on my 24GB vram on my RTX 3090
# For training on whatever this take > 24GB vram (Much more ideal)
bash ./run.sh #I can't run this on my GPU
```

### Step 4: Inference (TESTING/BROKEN)
Run inference using the trained model:
```bash
cd espeak_speech_ai
./direct_inference.py --text "This is a test of the speech system."
```

## Troubleshooting

### Common Issues

1. **WAV File Format Issues**:
   - Use the provided `fix_wav_files.sh` script
   - Or use the modified `offline_codec_tokenization_fixed.py` instead of the original

2. **CUDA Out of Memory**:
   - Reduce batch size in training configuration
   - Use DeepSpeed or gradient accumulation

3. **Path Issues**:
   - Ensure all paths in scripts are correctly set to your environment

## Research and Development

This project is in the experimental stage. Contributions and suggestions are welcome. Areas for improvement:

- Voice variety and quality in synthetic datasets
- Fine-tuning techniques for specific applications
- Performance optimization

## Acknowledgements

This project is based on the [RSTnet](https://github.com/yangdongchao/RSTnet) speech-text foundation model by Yang Dongchao et al. The original paper can be found [here](https://arxiv.org/abs/2308.07941).

## License

This project is released under the same license as the original RSTnet.

## Citation

If you use this work, please cite:

```bibtex
@article{yang2023uniaudio,
  title={UniAudio: An Audio Foundation Model Toward Universal Audio Generation},
  author={Dongchao Yang, Jinchuan Tian, Xu Tan, Rongjie Huang, Songxiang Liu, Xuankai Chang, Jiatong Shi, Sheng Zhao, Jiang Bian, Xixin Wu, Zhou Zhao, Helen Meng},
  journal={arXiv preprint arXiv:2310.00704},
  year={2023}
}
}
```
