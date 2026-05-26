#!/usr/bin/env python3
"""Generate HuggingFace reference tokens/logits for MicroLLM alignment.

The script runs greedy generation against a local HuggingFace model directory
and writes a compact JSON trace that MicroLLM can compare against.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a HuggingFace greedy-generation reference trace."
    )
    parser.add_argument(
        "--model_dir",
        required=True,
        type=Path,
        help="Local HuggingFace model directory.",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Prompt text to encode and generate from.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=8,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top logits to record per step.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output JSON path. Defaults to stdout.",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help="Torch dtype used for the reference model.",
    )
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype | str:
    if name == "auto":
        return "auto"
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def topk_trace(logits: torch.Tensor, top_k: int) -> list[dict[str, Any]]:
    values, indices = torch.topk(logits.float(), k=top_k)
    return [
        {"token_id": int(token_id), "logit": float(logit)}
        for token_id, logit in zip(indices.tolist(), values.tolist())
    ]


@torch.inference_mode()
def build_trace(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_new_tokens <= 0:
        raise ValueError("--max_new_tokens must be greater than 0")
    if args.top_k <= 0:
        raise ValueError("--top_k must be greater than 0")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, local_files_only=True, use_fast=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        local_files_only=True,
        torch_dtype=torch_dtype(args.dtype),
        device_map="cpu",
    )
    model.eval()

    input_ids = tokenizer.encode(
        args.prompt, add_special_tokens=True, return_tensors="pt"
    )
    generated: list[int] = []
    steps: list[dict[str, Any]] = []
    current_ids = input_ids

    for step in range(args.max_new_tokens):
        outputs = model(input_ids=current_ids, use_cache=False)
        next_logits = outputs.logits[0, -1]
        next_token = int(torch.argmax(next_logits).item())
        steps.append(
            {
                "step": step,
                "position": int(current_ids.shape[1] - 1),
                "input_token_id": int(current_ids[0, -1].item()),
                "next_token_id": next_token,
                "top_logits": topk_trace(next_logits, args.top_k),
            }
        )
        if tokenizer.eos_token_id is not None and next_token == tokenizer.eos_token_id:
            break
        generated.append(next_token)
        next_tensor = torch.tensor([[next_token]], dtype=current_ids.dtype)
        current_ids = torch.cat([current_ids, next_tensor], dim=1)

    return {
        "model_dir": str(args.model_dir),
        "prompt": args.prompt,
        "prompt_token_ids": [int(token) for token in input_ids[0].tolist()],
        "max_new_tokens": args.max_new_tokens,
        "top_k": args.top_k,
        "generated_token_ids": generated,
        "generated_text": tokenizer.decode(generated),
        "steps": steps,
    }


def main() -> None:
    args = parse_args()
    trace = build_trace(args)
    content = json.dumps(trace, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content + "\n", encoding="utf-8")
    else:
        print(content)

# python3 hf_reference_generate.py \
#   --model_dir my_tinyllama/AI-ModelScope/TinyLlama-1.1B-Chat-v1.0  \
#   --prompt "I am a" \
#   --max_new_tokens 8 \
#   --top_k 5 \
#   --dtype auto \
#   --output tinyllama_trace.json
if __name__ == "__main__":
    main()
