import json
import os
import re
import asyncio

from config import ResultType


def collect_preference_local_cot(
        pairs,
        backend,
        model_interface,
        output_file="preferences_local_cot.jsonl",
        batch_size=8):
    """
    Use a local LLM to judge which answer is better with reasoning.

    This function uses concurrent async requests to prompt the model to analyze
    and explain its reasoning before making a decision. The model generates a
    response that includes its thought process and final answer in \\boxed{} format.

    Args:
        pairs: List of question-answer pairs
        backend: AsyncModelBackend instance (HuggingFace or vLLM)
        model_interface: ModelInterface instance for model-specific behavior
        output_file: Output file for results
        batch_size: Number of concurrent requests (default: 8)

    Returns:
        None (results are written to output_file)
    """

    # Run async implementation
    asyncio.run(_collect_preference_local_cot_async(
        pairs=pairs,
        backend=backend,
        model_interface=model_interface,
        output_file=output_file,
        batch_size=batch_size
    ))


async def _collect_preference_local_cot_async(
        pairs,
        backend,
        model_interface,
        output_file,
        batch_size):
    """Async implementation of collect_preference_local_cot."""

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

    print(f"\nCollecting preferences with reasoning using local LLM")
    print(f"Results will be written to {output_file}")
    print(f"Concurrent requests: {batch_size}")

    # Collect unprocessed samples
    unprocessed_samples = []
    for i, pair in enumerate(pairs):
        if i not in processed_indices:
            unprocessed_samples.append((i, pair))

    total_to_process = len(unprocessed_samples)
    print(f"Samples to process: {total_to_process}")

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
                formatted_prompt = model_interface.build_messages_for_compare_cot(
                    tokenizer,
                    pair['question'],
                    pair['answer1'],
                    pair['answer2']
                )

                # Generate response
                result = await backend.generate_async(
                    formatted_prompt,
                    max_new_tokens=500,
                    temperature=0.0,
                    do_sample=False
                )

                raw_answer = result.generated_text

                # Extract the boxed answer
                match = re.search(r'\\boxed\{(\d+)\}', raw_answer)
                error_msg = None

                if match:
                    preference = int(match.group(1))
                    if preference not in [1, 2]:
                        error_msg = f"Invalid preference value in boxed answer: {preference}"
                        preference = 0
                else:
                    error_msg = "LLM did not include decision in \\boxed{} format"
                    preference = 0
                    print(f"  Warning: Could not parse answer for sample {i}, setting preference to 0")

                # Write result
                output_result = {
                    'index': i,
                    'preference': preference,
                    'reasoning': raw_answer,
                    'question': pair['question'],
                    'answer1': pair['answer1'],
                    'answer2': pair['answer2'],
                    'lang1': pair['lang1'],
                    'lang2': pair['lang2'],
                    'subject': pair.get('subject', ''),
                    'model': model_name
                }

                if error_msg:
                    output_result['error'] = error_msg

                async with lock:
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(output_result, ensure_ascii=False) + '\n')
                        f.flush()

                    results_dict[i] = preference
                    processed_count += 1

                    if processed_count % 5 == 0 or processed_count == total_to_process:
                        print(f"  Processed {processed_count}/{total_to_process} samples")

            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                raise

    # Process all unprocessed samples concurrently
    tasks = [process_single_sample(i, pair) for i, pair in unprocessed_samples]
    await asyncio.gather(*tasks)

    print("\nPreference collection completed.")