# MicroLLM

MicroLLM is a small C++/CUDA large language model inference runtime. It is
currently a work in progress: the repository focuses on building the core
runtime pieces by hand, including tensor storage, model loading, transformer
layers, CPU/CUDA operators, KV cache management, tokenization, and simple
sampling.

The current runnable path is closest to TinyLlama/LLaMA-style decoder-only
models exported into MicroLLM's custom binary format. Qwen2 and Qwen3 model
implementations are present in the codebase, but the public entrypoint and
model-type plumbing are still evolving.

## Goals

- Provide a compact LLM inference engine that is easy to inspect and modify.
- Implement the main transformer inference path in C++ with CUDA acceleration.
- Support exported HuggingFace-style model weights through a simple binary
  format.
- Keep the codebase useful for learning, debugging, and experimenting with LLM
  runtime internals.

## Current Features

- C++20 and CUDA-based runtime.
- Tensor and buffer abstractions for CPU/CUDA memory.
- Transformer operators:
  - embedding
  - matmul
  - RMSNorm
  - RoPE
  - multi-head attention
  - SwiGLU
  - vector add
- KV cache allocation and slicing.
- SentencePiece tokenizer support.
- Argmax sampler.
- FP32 model loading through memory-mapped checkpoint files.
- INT8 Q8_0-style quantized weight path for CUDA matmul.
- Unit tests for core tensor and operator behavior.
- Export utilities adapted from `llama2.c`.

## Repository Layout

```text
.
├── include/              # Public headers
├── src/
│   ├── base/             # Allocators, buffers, status helpers
│   ├── model/            # LLaMA/Qwen model implementations and model loader
│   ├── op/               # Runtime operators and CPU/CUDA kernels
│   ├── sampler/          # Token sampling
│   └── tensor/           # Tensor abstraction
├── test/                 # GTest-based operator and tensor tests
├── tools/                # Model download/export helpers
├── data/                 # Small test data and model download helper
├── CMakeLists.txt
├── build.sh
├── build_deps.sh
└── main.cpp              # Minimal generation demo entrypoint
```

## Dependencies

MicroLLM expects a Linux environment with CUDA and CMake. The current CMake
configuration targets CUDA architecture `80` by default, which is suitable for
Ampere GPUs such as A100/A800.

Main native dependencies are expected under `third_party/`:

- abseil
- armadillo
- boost
- gflags
- glog
- gtest
- nlohmann_json
- OpenBLAS
- re2
- sentencepiece
- unordered_dense
- libunwind
- CUDA Toolkit

The helper script downloads a prebuilt third-party bundle:

```bash
./build_deps.sh
```

For CUDA/CMake environment notes, see:

```bash
./build_env.sh
```

## Build

Build in release mode:

```bash
./build.sh release
```

Build in debug mode:

```bash
./build.sh debug
```

The build produces:

- `build/MicroLLM`
- `build/MicroLLM_test`

## Run Tests

After building:

```bash
cd build
ctest --output-on-failure
```

Some tests exercise CUDA operators, so they require a working CUDA runtime and
compatible GPU.

## Export a Model

The export tools live in `tools/`. Example TinyLlama export:

```bash
python3 tools/export_llama3.py \
  --version 3 \
  --hf tools/my_tinyllama/AI-ModelScope/TinyLlama-1.1B-Chat-v1.0 \
  tools/chat_q8.bin
```

See [tools/README.md](tools/README.md) for more details.

## Run Inference

The current demo entrypoint expects:

```bash
./build/MicroLLM --checkpoint <path> --tokenizer <path> [options]
```

Example:

```bash
./build/MicroLLM \
  --model-type llama2 \
  --checkpoint tools/chat_q8.bin \
  --tokenizer tools/my_tinyllama/AI-ModelScope/TinyLlama-1.1B-Chat-v1.0/tokenizer.model \
  --prompt "Write a short poem about CUDA" \
  --steps 128 \
  --device cuda \
  --quantized
```

Useful options:

```text
--model-type <llama2>      Model family. qwen2/qwen3 are not wired into the CLI yet.
--checkpoint <path>        MicroLLM checkpoint file.
--tokenizer <path>         Tokenizer model path.
--tokenizer-type <spe>     Tokenizer type. Currently spe is the stable path.
--prompt <text>            Prompt text. Default: hello
--steps <n>                Maximum generation steps. Default: 128
--device <cpu|cuda>        Runtime device. Default: cuda
--quantized                Load checkpoint as int8 Q8_0 weights.
```

The legacy positional form is still supported:

```bash
./build/MicroLLM <checkpoint_path> <tokenizer_path>
```

At the moment, `main.cpp` initializes `LLama2Model` for the stable path. The CLI
already exposes `--model-type` so Qwen2/Qwen3 can be wired into the same
entrypoint later.

## Model File Format

Model checkpoints begin with a fixed `ModelConfig` header:

```cpp
struct ModelConfig {
  int32_t dim;
  int32_t hidden_dim;
  int32_t layer_num;
  int32_t head_num;
  int32_t kv_head_num;
  int32_t vocab_size;
  int32_t seq_len;
};
```

Quantized checkpoints additionally store an `int32_t group_size` after the
header. The remaining bytes are interpreted as model weights in the order
expected by the corresponding model implementation.

Do not change the layout of `ModelConfig` without also updating the export and
loading logic.

## Current Limitations

- The main executable currently only instantiates `LLama2Model`.
- Qwen2/Qwen3 code exists, but model selection and end-to-end examples are not
  fully wired into the CLI.
- Sampling is currently argmax-only.
- CPU inference exists for parts of the runtime, but quantized inference is
  CUDA-only.
- Error handling and configuration are still rough in several places.
- The custom checkpoint format is tightly coupled to the current loader.
- There is no stable user-facing CLI yet.

## Roadmap Ideas

- Add a real CLI for model type, device, prompt, generation length, and
  quantization mode.
- Unify model selection across LLaMA, Qwen2, and Qwen3.
- Add top-k/top-p/temperature sampling.
- Improve export documentation and checkpoint compatibility checks.
- Add benchmark scripts for tokens/s, memory usage, and kernel timings.
- Expand tests from operator-level checks to end-to-end small-model inference.
- Improve CUDA kernels and reduce host/device synchronization.
- Document the binary weight layout per supported model family.

## Credits

The export utilities are adapted from Andrej Karpathy's
[llama2.c](https://github.com/karpathy/llama2.c). MicroLLM builds on that idea
while experimenting with a C++/CUDA runtime structure.
