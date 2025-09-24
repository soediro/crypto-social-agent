# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a machine learning project focused on fine-tuning language models for crypto-related text generation. The project uses PyTorch and Transformers to create specialized models that understand cryptocurrency and blockchain terminology.

## Core Architecture

### Directory Structure
- `data/` - Training data storage with `raw/` and `processed/` subdirectories
- `models/fine_tuned/` - Output directory for trained models
- `scripts/` - Standalone Python scripts for training
- `notebooks/` - Jupyter notebooks for interactive development and experimentation

### Key Components

**Training Script (`scripts/train_simple.py`)**
- Main entry point for model training
- Uses facebook/opt-350m as the base model (350M parameters)
- Implements basic fine-tuning with crypto-specific text data
- Outputs trained models to `./hello_world_model` directory

**Jupyter Notebook (`notebooks/01_hello_world_fine_tuning.ipynb`)**
- Interactive version of the training process
- Includes detailed experimentation and testing
- Saves models to `./models/fine_tuned/hello_world`

## Development Commands

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Start Jupyter for interactive development
jupyter notebook
```

### Training
```bash
# Run simple training script
python3 scripts/train_simple.py

# Or use the script directly (executable)
./scripts/train_simple.py
```

### Dependencies
The project uses:
- `torch>=2.0.0` - PyTorch for deep learning
- `transformers>=4.30.0` - Hugging Face transformers library
- `datasets>=2.12.0` - Data loading and processing
- `accelerate>=0.20.0` - Training acceleration
- `jupyter>=1.0.0` - Interactive notebooks

## Model Details

- **Base Model**: facebook/opt-350m (350M parameter OPT model)
- **Training Data**: Crypto-focused sentence completions
- **Output**: Fine-tuned models specialized for cryptocurrency content generation
- **Training Duration**: Minimal epochs (2-3) for quick experimentation

## File Locations

- Training outputs: `./hello_world_model` (script) or `./models/fine_tuned/hello_world` (notebook)
- Data processing happens in-memory with hardcoded examples
- No external data files currently required