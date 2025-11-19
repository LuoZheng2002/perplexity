import os
os.environ["HF_HOME"] = "/work/nvme/bfdz/zluo8/huggingface"

from transformers import AutoTokenizer, AutoModelForCausalLM

import torch
import math
import re


def find_assistant_answer_start(full_ids, prefix_ids):
    L = len(prefix_ids)
    for i in range(len(full_ids) - L + 1):
        if full_ids[i:i+L] == prefix_ids:
            return i + L  # the first token *after* the prefix
    raise ValueError("Assistant prefix not found in full_ids.")


def compute_assistant_answer_ppl(decoded_output, tokenizer, model):
    """
    Extracts the assistant's answer from a Qwen chat output, computes
    perplexity of the answer under the model by masking out all non-answer tokens.
    """

    # 1. Extract assistant content between <|im_start|>assistant ... <|im_end|>
    match = re.search(r"<\|im_start\|\>assistant\n(.*?)(?=<\|im_end\|\>)", 
                      decoded_output, 
                      flags=re.DOTALL)

    if not match:
        raise ValueError("Could not locate assistant answer block in the decoded output.")

    assistant_answer = match.group(1).strip()

    full_chat_text = decoded_output

    # 3. Tokenize the full text
    tokenized = tokenizer(full_chat_text, return_tensors="pt")
    input_ids = tokenized.input_ids.to(model.device)

    # 4. Identify token range corresponding ONLY to assistant answer
    # tokenize the assistant answer separately to find its length
    answer_tokens = tokenizer(assistant_answer, add_special_tokens=False).input_ids
    answer_len = len(answer_tokens)

    # Find where answer tokens start in the full sequence
    # We search from the end because answer appears last
    full_ids = input_ids[0].tolist()
    print("full_ids length:", len(full_ids))
    answer_start = None

    assistant_prefix = "<|im_start|>assistant\n"
    prefix_ids = tokenizer(assistant_prefix, add_special_tokens=False).input_ids

    # for i in range(len(full_ids) - answer_len + 1):
    #     if full_ids[i:i + answer_len] == answer_tokens:
    #         answer_start = i
    #         break

    answer_start = find_assistant_answer_start(full_ids, prefix_ids)
    print("answer_start:", answer_start)

    if answer_start is None:
        raise ValueError("Assistant answer tokens not found in full tokenized output.")

    answer_end = answer_start + answer_len  # exclusive
    print("answer_end:", answer_end)

    # 5. Run model forward pass (no sampling)
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # shape: [1, seq_len, vocab_size]

    # 6. Compute log-likelihood ONLY for answer tokens
    # shift logits by 1 to align with the next-token predictions
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    # mask: we only keep positions inside the assistant answer span
    mask = torch.zeros_like(shift_labels, dtype=torch.bool)
    mask[0, answer_start:answer_end] = True

    # cross entropy per token
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    selected_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

    # filter only answer tokens
    answer_log_probs = selected_log_probs[mask]

    avg_log_prob = answer_log_probs.mean().item()
    perplexity = math.exp(-avg_log_prob)

    return perplexity


# decoded_output ='''<|im_start|>system
# You are a helpful assistant.<|im_end|>
# <|im_start|>user
# Explain gravity simply.<|im_end|>
# <|im_start|>assistant
# Sure! Gravity is a fundamental force of nature that causes all objects with mass to be attracted to each other. Here’s a simple explanation:

# 1. **Attraction**: Objects with mass pull on each other, and the strength of this pull depends on their masses and the distance between them.
# 2. **Everyday Experience**: On Earth, we feel the pull of gravity as the force that keeps us on the ground and makes things fall when dropped.
# 3. **Universal Constant**: Gravity is always present, but its effects can vary depending on the mass and size of objects.

# In essence, gravity is what makes apples fall from trees and keeps our feet on the ground!<|im_end|>'''

decoded_output ='''<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Please output EXACTLY the following inside the quotation mark without adding anything else: "Hello, nice to meet you!"<|im_end|>
<|im_start|>assistant
"Hello, nice to meet you!"<|im_end|>'''

# print(decoded_output)

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

perplexity = compute_assistant_answer_ppl(decoded_output, tokenizer, model)
print("Perplexity:", perplexity)