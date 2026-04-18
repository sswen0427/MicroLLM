import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from modelscope import snapshot_download

# ==========================================
# 1. Download TinyLlama using ModelScope
# ==========================================
print("Starting to download TinyLlama-1.1B-Chat-v1.0 from ModelScope...")

# Specify the repo ID on ModelScope and the local cache directory
repo_id = 'AI-ModelScope/TinyLlama-1.1B-Chat-v1.0'
local_model_path = snapshot_download(repo_id, cache_dir='./my_tinyllama')

print(f"Download complete! Local model path: {local_model_path}")
print("-" * 50)

# ==========================================
# 2. Load the model using Transformers
# ==========================================
print("Loading tokenizer and model into memory/VRAM...")

tokenizer = AutoTokenizer.from_pretrained(
    local_model_path,
    trust_remote_code=True
)

# Load the model in float16 to save memory (highly recommended)
tiny_model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
).eval()

print("Model loaded successfully! Ready for inference...")
print("-" * 50)

# ==========================================
# 3. Inference (Standard Chat Format)
# ==========================================
# Define the conversation history using the standard messages format
messages = [
    {"role": "system", "content": "You are a helpful, respectful and honest AI assistant."},
    {"role": "user", "content": "Please write a very short poem about writing C++ code."}
]

# Apply the chat template specific to TinyLlama
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# Tokenize the prompt and move it to the correct device (CPU/GPU)
inputs = tokenizer(prompt, return_tensors="pt").to(tiny_model.device)

print("TinyLlama is generating a response...\n")

# Generate the output
with torch.no_grad():
    pred = tiny_model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,        # Enable sampling for more natural text
        temperature=0.7,       # Control creativity (0.0 to 1.0)
        top_p=0.9,
        repetition_penalty=1.1 # Penalize repetitive text
    )

# Extract only the newly generated tokens (ignore the input prompt)
input_length = inputs.input_ids.shape[1]
generated_tokens = pred[0][input_length:]

# Decode the tokens back to human-readable text
response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

print("TinyLlama's Response:")
print(response)
