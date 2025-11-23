import json
import os

from config import ResultType


def collect_preference_local_direct(
        pairs,
        model,
        tokenizer,
        model_name,
        model_interface,
        output_file="preferences_local.jsonl",
        device="cuda"):
    """
    Use a local LLM to judge which answer is better by comparing log probabilities.

    This function uses the model's forward pass to compute log probabilities for
    choosing answer 1 vs answer 2, rather than generating text. The preference
    is determined by which answer has the higher log probability.

    Args:
        pairs: List of question-answer pairs
        model: Pre-loaded model instance
        tokenizer: Pre-loaded tokenizer instance
        model_name: Hugging Face model name (for logging/identification)
        model_interface: ModelInterface instance for model-specific behavior
        output_file: Output file for results
        device: Device to use ("cuda" or "cpu")

    Returns:
        List of preferences (1 if answer1 is better, 2 if answer2 is better)
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
    if len(processed_indices) == len(pairs):
        print("All samples already processed. Exiting.")
        return

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
                    pair['answer1'],
                    pair['answer2']
                )

                # Tokenize the formatted prompt
                inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=1024)
                input_ids = inputs['input_ids'].to(device)

                # Get token IDs for "1" and "2"
                token_1 = tokenizer.encode("1", add_special_tokens=False)[0]
                token_2 = tokenizer.encode("2", add_special_tokens=False)[0]

                # Run forward pass to get logits for the next token
                with torch.no_grad():
                    outputs = model(input_ids)
                    logits = outputs.logits  # shape: [1, seq_len, vocab_size]

                    # Get logits for the next token (after the prompt)
                    next_token_logits = logits[0, -1, :]  # shape: [vocab_size]

                    # Compute log probabilities
                    log_probs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)

                    # Extract log probabilities for tokens "1" and "2"
                    log_prob_1 = log_probs[token_1].item()
                    log_prob_2 = log_probs[token_2].item()

                    # Calculate difference (log_prob_1 - log_prob_2)
                    log_prob_diff = log_prob_1 - log_prob_2

                    # Determine preference based on higher log probability
                    if log_prob_1 > log_prob_2:
                        preference = 1
                    else:
                        preference = 2

                # Write result immediately
                result = {
                    'index': i,
                    'preference': preference,
                    'log_prob_1': log_prob_1,
                    'log_prob_2': log_prob_2,
                    'log_prob_diff': log_prob_diff,
                    'question': pair['question'],
                    'answer1': pair['answer1'],
                    'answer2': pair['answer2'],
                    'lang1': pair['lang1'],
                    'lang2': pair['lang2'],
                    'subject': pair.get('subject', ''),
                    'model': model_name
                }

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
                    'log_prob_1': None,
                    'log_prob_2': None,
                    'log_prob_diff': None,
                    'error': str(e),
                    'model': model_name
                }
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                f.flush()
                results_dict[i] = None

    # Build final list in order
    print("\nPreference collection completed.")
