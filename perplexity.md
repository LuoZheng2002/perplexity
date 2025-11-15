The way of calculating perplexity of an API model:

OpenAI's models support the "logprobs" (True or False) and "top_logprobs" (0-5) parameters:
'''
params = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": stop,
        "seed": seed,
        "logprobs": logprobs,
        "top_logprobs": top_logprobs,
    }
    if tools:
        params["tools"] = tools

    completion = client.chat.completions.create(**params)
'''

Then we can send in 