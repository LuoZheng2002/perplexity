import os
import json
import math
import asyncio
import torch


def language_abbreviation_to_name(abbreviation):
    """
    Map language abbreviation to full language name.
    """
    lang_map = {
        'en': 'English',
        'fr': 'French',
        'de': 'German',
        'es': 'Spanish',
        'it': 'Italian',
        'pt': 'Portuguese',
        'zh_cn': 'Chinese',
        'ja': 'Japanese',
        'ko': 'Korean',
        # Add more mappings as needed
    }
    assert isinstance(abbreviation, str), "Language abbreviation must be a string"
    assert abbreviation in lang_map or len(abbreviation) > 2, f"Unknown language abbreviation: {abbreviation}"
    return lang_map.get(abbreviation, abbreviation)


def collect_perplexity_local(entries, backend, model_interface, output_file="perplexities_local.jsonl", batch_size=8):
    """
    Calculate the perplexity of each answer entry using concurrent async requests.

    Perplexity is calculated by getting the average log probability of tokens in the answer
    given the question context. Lower perplexity indicates the model finds the answer more likely.
    Also generates an answer for comparison.

    IMPORTANT: This function ONLY supports HuggingFace backend because it requires
    forward pass operations to compute perplexity, which vLLM does not support.

    Args:
        entries: List of individual answer entries, each containing:
            - 'index': int
            - 'question': str
            - 'answer': str
            - 'lang': str
            - 'is_correct': bool
            - 'subject': str
        backend: AsyncModelBackend instance (must be HuggingFace backend)
        model_interface: ModelInterface instance for model-specific behavior
        output_file: Output file for results
        batch_size: Number of concurrent requests (default: 8)

    Returns:
        None (results are written to output_file)
    """

    # Run async implementation
    asyncio.run(_collect_perplexity_local_async(
        entries=entries,
        backend=backend,
        model_interface=model_interface,
        output_file=output_file,
        batch_size=batch_size
    ))


async def _collect_perplexity_local_async(
        entries,
        backend,
        model_interface,
        output_file,
        batch_size):
    """Async implementation of collect_perplexity_local."""

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
                    results_dict[idx] = {
                        'perplexity': result.get('perplexity'),
                        'generated_answer': result.get('generated_answer')
                    }
        print(f"Found {len(processed_indices)} already processed samples")
    if len(processed_indices) == len(entries):
        print("All samples already processed. Exiting.")
        return

    print(f"\nCalculating perplexities using local LLM")
    print(f"Results will be written to {output_file}")
    print(f"Concurrent requests: {batch_size}")

    # Collect unprocessed samples
    unprocessed_samples = []
    for entry in entries:
        if entry['index'] not in processed_indices:
            unprocessed_samples.append(entry)

    total_to_process = len(unprocessed_samples)
    print(f"Samples to process: {total_to_process}")

    # Process samples with concurrency control
    semaphore = asyncio.Semaphore(batch_size)
    lock = asyncio.Lock()
    processed_count = 0

    async def process_single_entry(entry):
        """Process a single entry asynchronously."""
        nonlocal processed_count

        async with semaphore:
            try:
                language_name = language_abbreviation_to_name(entry['lang'])

                # Build prompts for both perplexity and generation
                full_chat_text = model_interface.build_messages_for_perplexity_forward(
                    tokenizer, entry['question'], entry['answer'], language_name
                )
                generation_prompt = model_interface.build_messages_for_perplexity_generate(
                    tokenizer, entry['question'], language_name
                )

                # Run forward pass and generation concurrently
                forward_result, gen_result = await asyncio.gather(
                    backend.forward_async(full_chat_text, max_length=2048),
                    backend.generate_async(generation_prompt, max_new_tokens=100, temperature=0.0, do_sample=False)
                )

                # Calculate perplexity from forward result
                logits = forward_result.logits  # [seq_len, vocab_size]
                input_ids = forward_result.input_ids

                # Get answer tokens
                answer_tokens = tokenizer(entry['answer'], add_special_tokens=False).input_ids

                # Find answer position
                answer_start = model_interface.find_answer_start(
                    tokenizer, input_ids, answer_tokens
                )
                answer_end = answer_start + len(answer_tokens)

                # Shift logits and labels for perplexity calculation
                shift_logits = logits[:-1, :]  # All but last position
                shift_labels = torch.tensor(input_ids[1:])  # All but first position

                # Compute log probabilities
                log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
                selected_log_probs = log_probs.gather(1, shift_labels.unsqueeze(-1)).squeeze(-1)

                # Extract log probs for answer tokens (adjusting for shift)
                mask_start = answer_start - 1
                mask_end = answer_end - 1
                answer_log_probs = selected_log_probs[mask_start:mask_end]

                # Calculate perplexity
                if len(answer_log_probs) > 0:
                    avg_log_prob = answer_log_probs.mean().item()
                    perplexity = math.exp(-avg_log_prob)
                else:
                    perplexity = None

                generated_answer = gen_result.generated_text

                # Write result
                output_result = {
                    'index': entry['index'],
                    'perplexity': perplexity,
                    'question': entry['question'],
                    'answer': entry['answer'],
                    'generated_answer': generated_answer,
                    'lang': entry['lang'],
                    'is_correct': entry['is_correct'],
                    'subject': entry.get('subject', ''),
                    'model': model_name,
                }

                async with lock:
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(output_result, ensure_ascii=False) + '\n')
                        f.flush()

                    results_dict[entry['index']] = {
                        'perplexity': perplexity,
                        'generated_answer': generated_answer
                    }
                    processed_count += 1

                    if processed_count % 10 == 0 or processed_count == total_to_process:
                        print(f"  Processed {processed_count}/{total_to_process} samples")

            except Exception as e:
                print(f"Error processing entry {entry['index']}: {e}")
                raise

    # Process all unprocessed samples concurrently
    tasks = [process_single_entry(entry) for entry in unprocessed_samples]
    await asyncio.gather(*tasks)

    print("\nPerplexity calculation completed.")
