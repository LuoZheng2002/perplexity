import json
import os
import re
def collect_preference_local_direct(pairs, model, tokenizer, model_name, model_interface, output_file="preferences_local.jsonl", device="cuda"):
    """
    Use a local LLM to judge which answer is better.

    Args:
        pairs: List of question-answer pairs
        model: Pre-loaded model instance
        tokenizer: Pre-loaded tokenizer instance
        model_name: Hugging Face model name (for logging/identification)
        model_interface: ModelInterface instance for model-specific behavior
        output_file: Output file for results
        device: Device to use ("cuda" or "cpu")

    Returns:
        List of preferences (1 if answer from lang1 is better, 2 if answer from lang2 is better)
    """
    import torch

    # Load already processed samples if file exists
    processed_indices = set()
    results_dict = {}

    if os.path.exists(output_file):
        print(f"Loading existing results from {output_file}...")
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    result = json.loads(line)
                    idx = result['index']
                    processed_indices.add(idx)
                    results_dict[idx] = result['preference']
        print(f"Found {len(processed_indices)} already processed samples")

    print(f"\nCollecting preferences using local LLM: {model_name}")
    print(f"Results will be written to {output_file}")

    # Open file in append mode
    with open(output_file, 'a', encoding='utf-8') as f:
        for i, pair in enumerate(pairs):
            # Skip if already processed
            if i in processed_indices:
                continue

            try:
                # Build formatted prompt using model-specific interface
                # This includes the chat template and "\\box{" prefix to encourage direct output
                formatted_prompt = model_interface.build_messages_for_compare_directly(
                    tokenizer,
                    pair['question'],
                    pair['answer_lang1'],
                    pair['answer_lang2']
                )

                # Tokenize the formatted prompt
                inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=1024)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Generate response
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=200,
                        temperature=0.0,
                        top_p=1.0,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )

                # Decode the output (skip the input tokens)
                raw_answer = tokenizer.decode(
                    output_ids[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                ).strip()

                # Extract the boxed answer
                match = re.search(r'\\boxed\{(\d+)\}', raw_answer)
                error_msg = None

                if match:
                    preference = int(match.group(1))
                    if preference not in [1, 2]:
                        error_msg = f"Invalid preference value in boxed answer: {preference}"
                        preference = 0
                else:
                    # Could not find boxed answer
                    error_msg = "LLM did not include decision in \\boxed{} format"
                    preference = 0
                    print(f"  Warning: Could not parse answer for sample {i}, setting preference to 0")

                # Write result immediately
                result = {
                    'index': i,
                    'preference': preference,
                    'raw_answer': raw_answer,
                    'question': pair['question'],
                    'answer_lang1': pair['answer_lang1'],
                    'answer_lang2': pair['answer_lang2'],
                    'lang1': pair['lang1'],
                    'lang2': pair['lang2'],
                    'model': model_name
                }

                if error_msg:
                    result['error'] = error_msg

                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                f.flush()

                results_dict[i] = preference

                if (len(results_dict)) % 5 == 0:
                    print(f"  Processed {len(results_dict)}/{len(pairs)} samples")

            except Exception as e:
                print(f"Error on sample {i}: {e}")
                # Write error result
                result = {
                    'index': i,
                    'preference': None,
                    'raw_answer': None,
                    'error': str(e),
                    'model': model_name
                }
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                f.flush()
                results_dict[i] = None

    # Build final list in order
    preferences = [results_dict.get(i, None) for i in range(len(pairs))]
    print(f"Collected {len([p for p in preferences if p is not None])} valid preferences")
    return preferences
