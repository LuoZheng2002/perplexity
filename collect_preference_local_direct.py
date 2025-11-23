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
        device="cuda",
        batch_size=8):
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
        batch_size: Number of samples to process in parallel (default: 8)

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
    print(f"Batch size: {batch_size}")

    # Collect unprocessed samples
    unprocessed_samples = []
    for i, pair in enumerate(pairs):
        if i not in processed_indices:
            unprocessed_samples.append((i, pair))

    total_to_process = len(unprocessed_samples)
    print(f"Samples to process: {total_to_process}")

    # Get token IDs for "1" and "2"
    token_1 = tokenizer.encode("1", add_special_tokens=False)[0]
    token_2 = tokenizer.encode("2", add_special_tokens=False)[0]

    # Save original padding side
    original_padding_side = tokenizer.padding_side

    # Open file in append mode
    with open(output_file, 'a', encoding='utf-8') as f:
        # Process in batches
        for batch_start in range(0, len(unprocessed_samples), batch_size):
            batch_end = min(batch_start + batch_size, len(unprocessed_samples))
            batch = unprocessed_samples[batch_start:batch_end]

            # Prepare batch data
            batch_indices = [item[0] for item in batch]
            batch_pairs = [item[1] for item in batch]

            try:
                # Build formatted prompts for all samples in batch
                formatted_prompts = []
                for pair in batch_pairs:
                    formatted_prompt = model_interface.build_messages_for_compare_directly(
                        tokenizer,
                        pair['question'],
                        pair['answer1'],
                        pair['answer2']
                    )
                    formatted_prompts.append(formatted_prompt)

                # Set padding side to left for decoder-only models
                tokenizer.padding_side = 'left'

                # Tokenize the batch with padding
                inputs = tokenizer(
                    formatted_prompts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1024,
                    padding=True
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Run forward pass to get logits for the next token
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits  # shape: [batch_size, seq_len, vocab_size]

                    # For each sample in batch, get the logits at the last non-padded position
                    # With left padding, the last token is always the actual last token
                    batch_results = []

                    for batch_idx in range(len(batch)):
                        # Get logits for the last token
                        next_token_logits = logits[batch_idx, -1, :]  # shape: [vocab_size]

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

                        batch_results.append({
                            'preference': preference,
                            'log_prob_1': log_prob_1,
                            'log_prob_2': log_prob_2,
                            'log_prob_diff': log_prob_diff
                        })

                # Write results for this batch
                for batch_idx, (i, pair) in enumerate(batch):
                    br = batch_results[batch_idx]
                    result = {
                        'index': i,
                        'preference': br['preference'],
                        'log_prob_1': br['log_prob_1'],
                        'log_prob_2': br['log_prob_2'],
                        'log_prob_diff': br['log_prob_diff'],
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

                    results_dict[i] = br['preference']

                # Progress update after each batch
                if len(results_dict) % 10 == 0 or batch_end == len(unprocessed_samples):
                    print(f"  Processed {len(results_dict)}/{total_to_process} samples")

            except Exception as e:
                print(f"Error processing batch starting at index {batch_start}: {e}")
                exit(1)
                # print(f"Error processing batch starting at index {batch_start}: {e}")
                # # Write error results for all samples in the failed batch
                # for i, pair in batch:
                #     result = {
                #         'index': i,
                #         'preference': None,
                #         'log_prob_1': None,
                #         'log_prob_2': None,
                #         'log_prob_diff': None,
                #         'error': str(e),
                #         'model': model_name,
                #         'subject': pair.get('subject', '')
                #     }
                #     f.write(json.dumps(result, ensure_ascii=False) + '\n')
                #     f.flush()
                #     results_dict[i] = None

            finally:
                # Restore original padding side
                tokenizer.padding_side = original_padding_side

    # Build final list in order
    print("\nPreference collection completed.")
