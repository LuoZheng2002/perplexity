import os
import json
import math


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


def collect_perplexity_local(entries, model, tokenizer, model_name, model_interface, output_file="perplexities_local.jsonl", device="cuda", batch_size=4):
    """
    Calculate the perplexity of each answer entry using a local LLM with batch inference.

    Perplexity is calculated by getting the average log probability of tokens in the answer
    given the question context. Lower perplexity indicates the model finds the answer more likely.

    Args:
        entries: List of individual answer entries, each containing:
            - 'index': int
            - 'question': str
            - 'answer': str
            - 'lang': str
            - 'is_correct': bool
            - 'subject': str
        model: Pre-loaded model instance
        tokenizer: Pre-loaded tokenizer instance
        model_name: Hugging Face model name (for logging/identification)
        model_interface: ModelInterface instance for model-specific behavior
        output_file: Output file for results
        device: Device to use ("cuda" or "cpu")
        batch_size: Number of samples to process in parallel (default: 4)

    Returns:
        None (results are written to output_file)
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
                    results_dict[idx] = {
                        'perplexity': result.get('perplexity'),
                        'generated_answer': result.get('generated_answer')
                    }
        print(f"Found {len(processed_indices)} already processed samples")
    if len(processed_indices) == len(entries):
        print("All samples already processed. Exiting.")
        return

    print(f"\nCalculating perplexities using local LLM: {model_name}")
    print(f"Results will be written to {output_file}")
    print(f"Batch size: {batch_size}")

    # Collect unprocessed samples
    unprocessed_samples = []
    for entry in entries:
        if entry['index'] not in processed_indices:
            unprocessed_samples.append(entry)

    total_to_process = len(unprocessed_samples)
    print(f"Samples to process: {total_to_process}")

    # Save original padding side
    original_padding_side = tokenizer.padding_side

    def calculate_batch_perplexities(batch_entries):
        """
        Calculate perplexities for a batch of entries.
        Returns list of perplexity values (or None for errors).
        """
        # Build formatted texts for all entries
        formatted_texts = []
        answer_infos = []  # Store answer token info for each entry

        for entry in batch_entries:
            language_name = language_abbreviation_to_name(entry['lang'])
            try:
                full_chat_text = model_interface.build_messages_for_perplexity_forward(
                    tokenizer, entry['question'], entry['answer'], language_name
                )
                formatted_texts.append(full_chat_text)

                # Get answer tokens
                answer_tokens = tokenizer(entry['answer'], add_special_tokens=False).input_ids
                answer_infos.append({
                    'answer_tokens': answer_tokens,
                    'answer_len': len(answer_tokens)
                })
            except Exception as e:
                print(f"Error building messages for entry {entry['index']}: {e}")
                exit(1)
                # formatted_texts.append(None)
                # answer_infos.append(None)

        # Filter out None entries for batching
        valid_indices = [i for i, t in enumerate(formatted_texts) if t is not None]
        if not valid_indices:
            return [None] * len(batch_entries)

        valid_texts = [formatted_texts[i] for i in valid_indices]

        # Set padding side to right for perplexity calculation (we need to know where answer ends)
        tokenizer.padding_side = 'right'

        # Tokenize batch
        tokenized = tokenizer(
            valid_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        input_ids = tokenized.input_ids.to(device)
        attention_mask = tokenized.attention_mask.to(device)

        # Find answer positions for each valid entry
        answer_positions = []
        for batch_idx, orig_idx in enumerate(valid_indices):
            info = answer_infos[orig_idx]
            full_ids = input_ids[batch_idx].tolist()

            try:
                answer_start = model_interface.find_answer_start(
                    tokenizer, full_ids, info['answer_tokens']
                )
                answer_end = answer_start + info['answer_len']
                answer_positions.append((answer_start, answer_end))
            except (ValueError, AttributeError) as e:
                print(f"Could not find answer start for entry {batch_entries[orig_idx]['index']}: {e}")
                exit(1)
                # answer_positions.append(None)

        # Run model forward pass
        perplexities = [None] * len(batch_entries)

        try:
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits  # shape: [batch_size, seq_len, vocab_size]

                # Shift logits and labels
                shift_logits = logits[:, :-1, :]
                shift_labels = input_ids[:, 1:]

                # Compute log probabilities
                log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
                selected_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

                # Calculate perplexity for each valid entry
                for batch_idx, orig_idx in enumerate(valid_indices):
                    pos = answer_positions[batch_idx]
                    if pos is None:
                        continue

                    answer_start, answer_end = pos

                    # Create mask for answer tokens (adjusting for shift)
                    mask = torch.zeros(shift_labels.shape[1], dtype=torch.bool, device=device)
                    mask[answer_start-1:answer_end-1] = True

                    # Get log probs for answer tokens
                    answer_log_probs = selected_log_probs[batch_idx][mask]

                    if len(answer_log_probs) > 0:
                        avg_log_prob = answer_log_probs.mean().item()
                        perplexity = math.exp(-avg_log_prob)
                        perplexities[orig_idx] = perplexity

        except Exception as e:
            print(f"Error in batch forward pass: {e}")
            exit(1)

        return perplexities

    def generate_batch_answers(batch_entries):
        """
        Generate answers for a batch of entries.
        Returns list of generated answer strings (or None for errors).
        """
        # Build prompts for all entries
        prompts = []
        for entry in batch_entries:
            language_name = language_abbreviation_to_name(entry['lang'])
            try:
                prompt_text = model_interface.build_messages_for_perplexity_generate(
                    tokenizer, entry['question'], language_name
                )
                prompts.append(prompt_text)
            except Exception as e:
                print(f"Error building generation prompt for entry {entry['index']}: {e}")
                exit(1)
                # prompts.append(None)

        # Filter out None entries
        valid_indices = [i for i, p in enumerate(prompts) if p is not None]
        if not valid_indices:
            return [None] * len(batch_entries)

        valid_prompts = [prompts[i] for i in valid_indices]

        # Set padding side to left for generation
        tokenizer.padding_side = 'left'

        # Tokenize batch
        tokenized = tokenizer(
            valid_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        )
        input_ids = tokenized.input_ids.to(device)
        attention_mask = tokenized.attention_mask.to(device)

        generated_answers = [None] * len(batch_entries)

        try:
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=100,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

                # Decode each output
                for batch_idx, orig_idx in enumerate(valid_indices):
                    input_length = input_ids[batch_idx].shape[0]
                    generated_ids = output_ids[batch_idx][input_length:]
                    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                    generated_answers[orig_idx] = generated_text.strip()

        except Exception as e:
            print(f"Error in batch generation: {e}")
            exit(1)

        return generated_answers

    # Open file in append mode
    with open(output_file, 'a', encoding='utf-8') as f:
        # Process in batches
        for batch_start in range(0, len(unprocessed_samples), batch_size):
            batch_end = min(batch_start + batch_size, len(unprocessed_samples))
            batch_entries = unprocessed_samples[batch_start:batch_end]

            try:
                # Calculate perplexities for batch
                perplexities = calculate_batch_perplexities(batch_entries)

                # Generate answers for batch
                generated_answers = generate_batch_answers(batch_entries)

                # Write results
                for i, entry in enumerate(batch_entries):
                    result = {
                        'index': entry['index'],
                        'perplexity': perplexities[i],
                        'question': entry['question'],
                        'answer': entry['answer'],
                        'generated_answer': generated_answers[i],
                        'lang': entry['lang'],
                        'is_correct': entry['is_correct'],
                        'subject': entry.get('subject', ''),
                        'model': model_name,
                    }

                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    f.flush()

                    results_dict[entry['index']] = {
                        'perplexity': perplexities[i],
                        'generated_answer': generated_answers[i]
                    }

                # Progress update
                if len(results_dict) % 10 == 0 or batch_end == len(unprocessed_samples):
                    print(f"  Processed {len(results_dict)}/{total_to_process} samples")

            except Exception as e:
                print(f"Error processing batch starting at index {batch_start}: {e}")
                exit(1)
                # print(f"Error processing batch: {e}")
                # # Write error results for all samples in the failed batch
                # for entry in batch_entries:
                #     result = {
                #         'index': entry['index'],
                #         'perplexity': None,
                #         'generated_answer': None,
                #         'error': str(e),
                #         'model': model_name,
                #         'subject': entry.get('subject', ''),
                #     }
                #     f.write(json.dumps(result, ensure_ascii=False) + '\n')
                #     f.flush()
                #     results_dict[entry['index']] = {
                #         'perplexity': None,
                #         'generated_answer': None
                #     }

            finally:
                # Restore original padding side
                tokenizer.padding_side = original_padding_side

    print("\nPerplexity calculation completed.")
