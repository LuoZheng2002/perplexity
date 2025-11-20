import json
import os
import re

from config import ResultType


def collect_preference_local_thinking(
        pairs,
        model,
        tokenizer,
        model_name,
        model_interface,
        output_file="preferences_local_thinking.jsonl",
        device="cuda",
        batch_size=1):
    """
    Use a local LLM to judge which answer is better with reasoning.

    This function prompts the model to analyze and explain its reasoning before
    making a decision. The model generates a response that includes its thought
    process and final answer in \\boxed{} format.

    Args:
        pairs: List of question-answer pairs
        model: Pre-loaded model instance
        tokenizer: Pre-loaded tokenizer instance
        model_name: Hugging Face model name (for logging/identification)
        model_interface: ModelInterface instance for model-specific behavior
        output_file: Output file for results
        device: Device to use ("cuda" or "cpu")
        batch_size: Number of samples to process in parallel (default: 1)

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

    print(f"\nCollecting preferences with reasoning using local LLM: {model_name}")
    print(f"Results will be written to {output_file}")
    print(f"Batch size: {batch_size}")

    # Collect unprocessed samples
    unprocessed_samples = []
    for i, pair in enumerate(pairs):
        if i not in processed_indices:
            unprocessed_samples.append((i, pair))

    total_to_process = len(unprocessed_samples)
    print(f"Samples to process: {total_to_process}")

    # Open file in append mode
    with open(output_file, 'a', encoding='utf-8') as f:
        # Process in batches
        for batch_start in range(0, len(unprocessed_samples), batch_size):
            batch_end = min(batch_start + batch_size, len(unprocessed_samples))
            batch = unprocessed_samples[batch_start:batch_end]

            # Prepare batch data
            batch_indices = [item[0] for item in batch]
            batch_pairs = [item[1] for item in batch]

            # Save original padding side and set to left for decoder-only models
            original_padding_side = tokenizer.padding_side

            try:
                # Build formatted prompts for all samples in batch
                formatted_prompts = []
                for pair in batch_pairs:
                    formatted_prompt = model_interface.build_messages_for_compare_thinking(
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

                # Generate responses for the batch
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=500,  # More tokens to allow for reasoning
                        temperature=0.0,
                        top_p=1.0,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )

                # Process each output in the batch
                for batch_idx, (i, pair) in enumerate(batch):
                    try:
                        # Decode the output (skip the input tokens)
                        input_length = inputs['input_ids'][batch_idx].shape[0]
                        raw_answer = tokenizer.decode(
                            output_ids[batch_idx][input_length:],
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
                            'reasoning': raw_answer,
                            'question': pair['question'],
                            'answer1': pair['answer1'],
                            'answer2': pair['answer2'],
                            'lang1': pair['lang1'],
                            'lang2': pair['lang2'],
                            'model': model_name
                        }

                        if error_msg:
                            result['error'] = error_msg

                        f.write(json.dumps(result, ensure_ascii=False) + '\n')
                        f.flush()

                        results_dict[i] = preference

                    except Exception as e:
                        print(f"Error processing sample {i} in batch: {e}")
                        # Write error result
                        result = {
                            'index': i,
                            'preference': None,
                            'reasoning': None,
                            'error': str(e),
                            'model': model_name
                        }
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')
                        f.flush()
                        results_dict[i] = None

                # Progress update after each batch
                if len(results_dict) % 5 == 0 or batch_end == len(unprocessed_samples):
                    print(f"  Processed {len(results_dict)}/{len(pairs)} samples")

            except Exception as e:
                print(f"Error processing batch starting at index {batch_start}: {e}")
                # Write error results for all samples in the failed batch
                for i, pair in batch:
                    result = {
                        'index': i,
                        'preference': None,
                        'reasoning': None,
                        'error': f"Batch error: {str(e)}",
                        'model': model_name
                    }
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    f.flush()
                    results_dict[i] = None

            finally:
                # Always restore original padding side
                tokenizer.padding_side = original_padding_side

    # Build final list in order
    print("\nPreference collection completed.")