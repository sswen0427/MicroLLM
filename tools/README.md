# Tools Directory

This directory contains utility scripts for MicroLLM.

## Files

### export.py
Source: [https://github.com/karpathy/llama2.c/blob/master/export.py](https://github.com/karpathy/llama2.c/blob/master/export.py)

This script is used for exporting model weights to a format compatible with the MicroLLM implementation.

### model.py
Source: [https://github.com/karpathy/llama2.c/blob/master/model.py](https://github.com/karpathy/llama2.c/blob/master/model.py)

This script defines the model architecture and inference logic, adapted for use with MicroLLM.

## Usage

### Exporting Model Weights
```bash
python export.py --checkpoint <path_to_checkpoint> --output <output_file>
```

### Running Model Inference
```bash
python model.py --checkpoint <path_to_checkpoint> --prompt "Your prompt here"
```

## Notes

- These scripts are adapted from the llama2.c project by Andrej Karpathy.
- They have been modified to work with the MicroLLM architecture and requirements.
- For more information about the original implementation, please refer to the [llama2.c repository](https://github.com/karpathy/llama2.c).
