import os
os.environ["HF_HOME"] = "/work/nvme/bfdz/zluo8/huggingface"

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    trust_remote_code=True,
    device_map="auto",         # stream to GPU, no CPU RAM bottleneck
    torch_dtype="auto",        # avoid expensive fp32→fp16 conversion
    low_cpu_mem_usage=True,    # avoids full-shard load into CPU
    use_safetensors=True       # skip .bin files if present
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Please output EXACTLY the following inside the quotation mark without adding anything else: \"Hello, nice to meet you!\""},
]

chat_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
print("Chat text:\n", chat_text)

tokenized_inputs = tokenizer(chat_text, return_tensors="pt").to("cuda")

output = model.generate(**tokenized_inputs, max_new_tokens=4096)

print("Decoded output:")
print(tokenizer.decode(output[0], skip_special_tokens=False))
