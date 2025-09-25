# CUDA 12.4 Setup Instructions

## System Requirements
- NVIDIA GTX 1050 (Mobile) or compatible GPU
- NVIDIA Driver 550.163.01 or later
- CUDA 12.4

## Installation Steps

### 1. Install PyTorch with CUDA 12.4 Support
```bash
# Install PyTorch with CUDA 12.4 support
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124

# Or install all requirements
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu124
```

### 2. Verify CUDA Installation
```bash
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}') if torch.cuda.is_available() else None"
```

### 3. Memory Optimization for GTX 1050 Mobile
The codebase has been optimized for GTX 1050 Mobile with:
- Reduced batch size (1 instead of 2)
- Gradient accumulation (2 steps)
- Half precision (fp16) training when CUDA is available
- Disabled memory pinning to reduce memory usage

### 4. Training Performance Tips
- Close unnecessary applications before training
- Monitor GPU memory usage with `nvidia-smi`
- Consider reducing model size if memory issues persist
- Use gradient checkpointing for larger models if needed

### 5. Troubleshooting
If you encounter CUDA out of memory errors:
1. Reduce `per_device_train_batch_size` to 1
2. Increase `gradient_accumulation_steps`
3. Use `torch.float32` instead of `torch.float16` if stability issues occur
4. Clear CUDA cache: `torch.cuda.empty_cache()`

## Current Configuration
The project is configured to automatically:
- Detect CUDA availability
- Use appropriate data types (fp16 for CUDA, fp32 for CPU)
- Optimize memory usage for mobile GPUs
- Display GPU information at startup