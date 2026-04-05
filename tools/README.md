# Tools Directory

This directory contains utility scripts for MicroLLM.

## Files

### export.py
Source: [https://github.com/karpathy/llama2.c/blob/master/export.py](https://github.com/karpathy/llama2.c/blob/master/export.py)

This script is used for exporting model weights to a format compatible with the MicroLLM implementation.

### model.py
Source: [https://github.com/karpathy/llama2.c/blob/master/model.py](https://github.com/karpathy/llama2.c/blob/master/model.py)

This script defines the model architecture and inference logic, adapted for use with MicroLLM.

### export_llama3.py
This script fixes some issues with the export.py script.

### hf_*.py
These scripts are used to download model weights.

## Usage

### Quantization
```bash
python3 export_llama3.py --version 3 --hf my_tinyllama/AI-ModelScope/TinyLlama-1.1B-Chat-v1.0 chat_q8.bin
```

## Notes

- These scripts are adapted from the llama2.c project by Andrej Karpathy.
- They have been modified to work with the MicroLLM architecture and requirements.
- For more information about the original implementation, please refer to the [llama2.c](https://github.com/karpathy/llama2.c) repository.
