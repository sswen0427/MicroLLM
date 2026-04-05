from transformers import AutoModelForCausalLM, AutoTokenizer
from modelscope import snapshot_download

print('downloading qwen2 model from ModelScope...')
local_model_path = snapshot_download(
    'qwen/Qwen2.5-0.5B',
    cache_dir='./my_qwen_model'
)

tokenizer = AutoTokenizer.from_pretrained(
    local_model_path,
    trust_remote_code=True
)

qwen_model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    device_map="auto",
    trust_remote_code=True
).eval()

print(qwen_model)

inputs = tokenizer("你好", return_tensors="pt")
inputs = inputs.to(qwen_model.device)
pred = qwen_model.generate(
    **inputs,
    max_new_tokens=128,
    do_sample=False,
    repetition_penalty=1.0
)
# 你好，我需要预订一张从纽约到洛杉矶的机票。 你好！请问您需要从纽约到洛杉矶的机票，还是从洛杉矶到纽约的机票？

# 从洛杉矶到纽约。 了解了，我为您预订了一张从洛杉矶到纽约的机票，价格为$1,200。您需要确认一下吗？

# 好的，我确认了。 请告诉我您的姓名和联系方式，以便我为您预订机票。<|endoftext|>

test = tokenizer.decode(pred.cpu()[0], skip_special_tokens=False)
print(test)