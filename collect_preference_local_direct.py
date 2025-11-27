import json
import os
import asyncio
import torch

from config import ResultType


def collect_preference_local_direct(
        pairs,
        backend,
        model_interface,
        output_file="preferences_local.jsonl",
        batch_size=8):
    """
    Use a local LLM to judge which answer is better by comparing log probabilities.

    This function uses concurrent async requests to the model backend to compute
    log probabilities for choosing answer 1 vs answer 2. The preference is
    determined by which answer has the higher log probability.

    IMPORTANT: This function ONLY supports HuggingFace backend because it requires
    forward pass operations to get log probabilities, which vLLM does not support.

    Args:
        pairs: List of question-answer pairs
        backend: AsyncModelBackend instance (must be HuggingFace backend)
        model_interface: ModelInterface instance for model-specific behavior
        output_file: Output file for results
        batch_size: Number of concurrent requests (default: 8)

    Returns:
        None (results are written to output_file)
    """

    # Run async implementation
    asyncio.run(_collect_preference_local_direct_async(
        pairs=pairs,
        backend=backend,
        model_interface=model_interface,
        output_file=output_file,
        batch_size=batch_size
    ))


async def _collect_preference_local_direct_async(
        pairs,
        backend,
        model_interface,
        output_file,
        batch_size):
    """Async implementation of collect_preference_local_direct."""

    tokenizer = backend.tokenizer
    model_name = getattr(backend, 'model_name', 'unknown')

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

    print(f"\nCollecting preferences using local LLM")
    print(f"Results will be written to {output_file}")
    print(f"Concurrent requests: {batch_size}")

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

    # Process samples with concurrency control
    semaphore = asyncio.Semaphore(batch_size)
    lock = asyncio.Lock()
    processed_count = 0

    async def process_single_sample(i, pair):
        """Process a single sample asynchronously."""
        nonlocal processed_count

        async with semaphore:
            try:
                # Build formatted prompt
                formatted_prompt = model_interface.build_messages_for_compare_directly(
                    tokenizer,
                    pair['question'],
                    pair['answer1'],
                    pair['answer2']
                )

                # Run forward pass
                result = await backend.forward_async(formatted_prompt, max_length=1024)

                # Extract logits for the last token
                logits = result.logits  # [seq_len, vocab_size]
                next_token_logits = logits[-1, :]  # [vocab_size]

                # Compute log probabilities
                log_probs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)

                # Extract log probabilities for tokens "1" and "2"
                log_prob_1 = log_probs[token_1].item()
                log_prob_2 = log_probs[token_2].item()

                # Calculate difference
                log_prob_diff = log_prob_1 - log_prob_2

                # Determine preference
                preference = 1 if log_prob_1 > log_prob_2 else 2

                # Write result
                output_result = {
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

                async with lock:
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(output_result, ensure_ascii=False) + '\n')
                        f.flush()

                    results_dict[i] = preference
                    processed_count += 1

                    if processed_count % 10 == 0 or processed_count == total_to_process:
                        print(f"  Processed {processed_count}/{total_to_process} samples")

            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                raise

    # Process all unprocessed samples concurrently
    tasks = [process_single_sample(i, pair) for i, pair in unprocessed_samples]
    await asyncio.gather(*tasks)

    print("\nPreference collection completed.")
