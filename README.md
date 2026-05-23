# MicroLLM

MicroLLM is a small C++/CUDA large language model inference runtime. It is
currently a work in progress: the repository focuses on building the core
runtime pieces by hand, including tensor storage, model loading, transformer
layers, CPU/CUDA operators, KV cache management, tokenization, and simple
sampling.

The current public entrypoint targets HuggingFace-style LLaMA/TinyLlama model
directories and reads model structure directly from `config.json` and
`.safetensors` files. Qwen2 and Qwen3 model implementations are present in the
codebase, but the public entrypoint and model-type plumbing are still evolving.

## Goals

- Provide a compact LLM inference engine that is easy to inspect and modify.
- Implement the main transformer inference path in C++ with CUDA acceleration.
- Load model metadata and weights from HuggingFace-style model directories.
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
- HuggingFace `config.json` parsing for LLaMA-style models.
- safetensors metadata inspection and tensor shape validation.
- SentencePiece tokenizer support in the lower-level runtime.
- Argmax sampler.
- Legacy FP32 and INT8 binary weight loaders remain in the lower-level runtime,
  but they are no longer exposed by the public executable.
- Unit tests for core tensor and operator behavior.
- Export utilities adapted from `llama2.c`.

## Repository Layout

```text
.
├── src/
│   ├── base/             # Allocators, buffers, status helpers
│   ├── cli/              # Executable command-line parsing
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

## Inspect a Model

The executable currently accepts a HuggingFace model directory and validates
that its `config.json` matches the tensor names and shapes in the safetensors
weights:

```bash
./build/MicroLLM --model_dir <hf_model_dir>
```

Example:

```bash
./build/MicroLLM \
  --model_dir data/my_tinyllama/AI-ModelScope/TinyLlama-1___1B-Chat-v1___0
```

The output includes the model type, hidden size, attention head count, KV head
count, vocabulary size, tensor count, and every validated LLaMA weight tensor.

```text
--model_dir <path>         HuggingFace model directory.
```

## Legacy Model File Format

The older internal binary loader is still present in the lower-level runtime,
but it is no longer exposed through the public executable. Legacy binary model
files begin with a fixed `ModelConfig` header:

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

Quantized legacy files additionally store an `int32_t group_size` after the
header. The remaining bytes are interpreted as model weights in the order
expected by the corresponding model implementation.

The project is moving toward direct HuggingFace safetensors loading so users can
run against downloaded model directories without an offline export step.

## Current Limitations

- The main executable currently inspects HuggingFace LLaMA/TinyLlama model
  directories; safetensors-backed inference loading is still being wired in.
- Qwen2/Qwen3 code exists, but model selection and end-to-end examples are not
  fully wired into the public entrypoint.
- Sampling is currently argmax-only.
- CPU inference exists for parts of the runtime, but quantized inference is
  CUDA-only.
- Error handling and configuration are still rough in several places.
- The legacy custom checkpoint format is tightly coupled to the old loader.

## Roadmap Ideas

- Load LLaMA/TinyLlama weights directly from safetensors into the runtime.
- Add a real CLI for prompt, device, generation length, and sampling settings.
- Unify model selection across LLaMA, Qwen2, and Qwen3.
- Add top-k/top-p/temperature sampling.
- Add safetensors compatibility checks for sharded models and additional model
  families.
- Add benchmark scripts for tokens/s, memory usage, and kernel timings.
- Expand tests from operator-level checks to end-to-end small-model inference.
- Improve CUDA kernels and reduce host/device synchronization.
- Document the binary weight layout per supported model family.

## Credits

The export utilities are adapted from Andrej Karpathy's
[llama2.c](https://github.com/karpathy/llama2.c). MicroLLM builds on that idea
while experimenting with a C++/CUDA runtime structure.
