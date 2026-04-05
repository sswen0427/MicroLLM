from transformers import AutoModelForCausalLM, AutoTokenizer
from modelscope import snapshot_download

print('downloading llama3 model from ModelScope...')
local_model_path = snapshot_download('LLM-Research/Llama-3.2-1B', cache_dir='./my_llama3_model')

tokenizer = AutoTokenizer.from_pretrained(
    local_model_path,
    trust_remote_code=True
)

llm_model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    device_map="auto",
    trust_remote_code=True,
).eval()

print(llm_model)

inputs = tokenizer("hello", return_tensors="pt")
inputs = inputs.to(llm_model.device)

pred = llm_model.generate(
    **inputs,
    max_new_tokens=128,
    do_sample=False,
    repetition_penalty=1.0
)

# hello, i am new to this forum and i am trying to get a better understanding of the different types of data that can be used in a regression model. i have a dataset that contains 3 variables: age, sex, and income. i am trying to determine if there is a relationship between age and income. i have tried to use the lm() function in R but i am not sure how to interpret the results. i have also tried to use the lm() function in SPSS but i am not sure how to interpret the results. can someone please help me understand the results of the lm() function in R and SPSS?
test = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
print(test)
