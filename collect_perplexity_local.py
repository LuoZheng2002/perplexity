import os
import json
import math

def collect_perplexity_local(pairs, model, tokenizer, model_name, model_interface, output_file="perplexities_local.jsonl", device="cuda"):
    """
    Calculate the perplexity of each answer in the pairs using a local LLM.

    Perplexity is calculated by getting the average log probability of tokens in the answer
    given the question context. Lower perplexity indicates the model finds the answer more likely.

    Args:
        pairs: List of question-answer pairs
        model: Pre-loaded model instance
        tokenizer: Pre-loaded tokenizer instance
        model_name: Hugging Face model name (for logging/identification)
        model_interface: ModelInterface instance for model-specific behavior
        output_file: Output file for results
        device: Device to use ("cuda" or "cpu")

    Returns:
        Tuple of (perplexities_lang1, perplexities_lang2)
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
                        'perplexity_lang1': result.get('perplexity_lang1'),
                        'perplexity_lang2': result.get('perplexity_lang2')
                    }
        print(f"Found {len(processed_indices)} already processed samples")
    if len(processed_indices) == len(pairs):
        print("All samples already processed. Exiting.")
        return

    print(f"\nCalculating perplexities using local LLM: {model_name}")
    print(f"Results will be written to {output_file}")

    def calculate_answer_perplexity(question, answer, language_name: str):
        """
        Calculate perplexity of an answer given a question.
        Uses chat template formatting with system, user, and assistant messages.
        Returns perplexity computed as exp(-avg_log_prob).
        """
        # Build formatted conversation text using model-specific interface
        try:
            full_chat_text = model_interface.build_messages_for_perplexity(
                tokenizer, question, answer, language_name
            )
        except Exception as e:
            print(f"Error building messages for perplexity: {e}")
            exit(1)
        print("full chat text:\n", full_chat_text)
        print("answer:\n", answer)
        # Tokenize the full text
        tokenized = tokenizer(full_chat_text, return_tensors="pt")
        input_ids = tokenized.input_ids.to(device)

        # Identify token range corresponding to the assistant answer
        # Tokenize the answer separately to find its length
        answer_tokens = tokenizer(answer, add_special_tokens=False).input_ids
        answer_len = len(answer_tokens)

        # Find where answer tokens start in the full sequence
        full_ids = input_ids[0].tolist()

        try:
            # Use the model interface to find the answer start position
            answer_start = model_interface.find_answer_start(tokenizer, full_ids, answer_tokens)
        except (ValueError, AttributeError) as e:
            print(f"Could not find assistant answer start: {e}")
            exit(1)

        answer_end = answer_start + answer_len  # exclusive

        try:
            with torch.no_grad():
                # Run model forward pass
                outputs = model(input_ids)
                logits = outputs.logits  # shape: [1, seq_len, vocab_size]

                # Shift logits and labels to align with next-token predictions
                shift_logits = logits[:, :-1, :]
                shift_labels = input_ids[:, 1:]

                # Create mask: only keep positions inside the assistant answer span
                mask = torch.zeros_like(shift_labels, dtype=torch.bool)
                mask[0, answer_start:answer_end] = True

                # Compute log probabilities
                log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
                selected_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

                # Filter only answer tokens
                answer_log_probs = selected_log_probs[mask]

                if len(answer_log_probs) > 0:
                    avg_log_prob = answer_log_probs.mean().item()
                    perplexity = math.exp(-avg_log_prob)
                    return perplexity
                else:
                    return None
        except Exception as e:
            print(f"Error calculating perplexity: {e}")
            return None

    # Open file in append mode
    with open(output_file, 'a', encoding='utf-8') as f:
        for i, pair in enumerate(pairs):
            # Skip if already processed
            if i in processed_indices:
                continue

            try:
                # Calculate perplexity for lang1 answer
                perplexity_lang1 = calculate_answer_perplexity(
                    pair['question'],
                    pair['answer1'],
                    pair['lang1']
                )

                # Calculate perplexity for lang2 answer
                perplexity_lang2 = calculate_answer_perplexity(
                    pair['question'],
                    pair['answer2'],
                    pair['lang2']
                )

                # Write result immediately
                result = {
                    'index': i,
                    'perplexity_lang1': perplexity_lang1,
                    'perplexity_lang2': perplexity_lang2,
                    'question': pair['question'],
                    'answer1': pair['answer1'],
                    'answer2': pair['answer2'],
                    'lang1': pair['lang1'],
                    'lang2': pair['lang2'],
                    'model': model_name
                }

                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                f.flush()

                results_dict[i] = {
                    'perplexity_lang1': perplexity_lang1,
                    'perplexity_lang2': perplexity_lang2
                }

                if (len(results_dict)) % 5 == 0:
                    print(f"  Processed {len(results_dict)}/{len(pairs)} samples")

            except Exception as e:
                print(f"Error on sample {i}: {e}")
                # Write error result
                result = {
                    'index': i,
                    'perplexity_lang1': None,
                    'perplexity_lang2': None,
                    'error': str(e),
                    'model': model_name
                }
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                f.flush()
                results_dict[i] = {
                    'perplexity_lang1': None,
                    'perplexity_lang2': None
                }

    # Build final lists in order
    print("\nPerplexity calculation completed.")