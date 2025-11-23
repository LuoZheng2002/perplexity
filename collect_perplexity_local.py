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

def collect_perplexity_local(entries, model, tokenizer, model_name, model_interface, output_file="perplexities_local.jsonl", device="cuda"):
    """
    Calculate the perplexity of each answer entry using a local LLM.

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

    def calculate_answer_perplexity(question, answer, language_name: str):
        """
        Calculate perplexity of an answer given a question.
        Uses chat template formatting with system, user, and assistant messages.
        Returns perplexity computed as exp(-avg_log_prob).
        """
        # Build formatted conversation text using model-specific interface
        try:
            full_chat_text = model_interface.build_messages_for_perplexity_forward(
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
                # Note: answer_start/answer_end are in original input_ids coordinates,
                # but shift_labels is offset by 1, so we need to adjust indices
                mask = torch.zeros_like(shift_labels, dtype=torch.bool)
                mask[0, answer_start-1:answer_end-1] = True

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

    def generate_answer(question, language_name: str):
        """
        Generate an answer to the question using the model.
        Uses chat template formatting with add_generation_prompt=True.
        Returns the generated answer text.
        """
        try:
            # Build formatted conversation text using model-specific interface
            prompt_text = model_interface.build_messages_for_perplexity_generate(
                tokenizer, question, language_name
            )
        except Exception as e:
            print(f"Error building messages for generation: {e}")
            return None

        # Tokenize the prompt
        tokenized = tokenizer(prompt_text, return_tensors="pt")
        input_ids = tokenized.input_ids.to(device)

        try:
            with torch.no_grad():
                # Generate answer
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=100,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

                # Decode only the generated tokens (excluding the prompt)
                generated_ids = output_ids[0][input_ids.shape[1]:]
                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

                return generated_text.strip()
        except Exception as e:
            print(f"Error generating answer: {e}")
            return None

    # Open file in append mode
    with open(output_file, 'a', encoding='utf-8') as f:
        for i, entry in enumerate(entries):
            # Skip if already processed
            if entry['index'] in processed_indices:
                continue

            try:
                language_name = language_abbreviation_to_name(entry['lang'])

                # Calculate perplexity for this answer
                perplexity = calculate_answer_perplexity(
                    entry['question'],
                    entry['answer'],
                    language_name
                )

                # Generate answer for this language
                generated_answer = generate_answer(
                    entry['question'],
                    language_name
                )

                # Write result immediately
                result = {
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

                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                f.flush()

                results_dict[entry['index']] = {
                    'perplexity': perplexity,
                    'generated_answer': generated_answer
                }

                if (len(results_dict)) % 5 == 0:
                    print(f"  Processed {len(results_dict)}/{len(entries)} samples")

            except Exception as e:
                print(f"Error on sample {entry['index']}: {e}")
                # Write error result
                result = {
                    'index': entry['index'],
                    'perplexity': None,
                    'generated_answer': None,
                    'error': str(e),
                    'model': model_name
                }
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                f.flush()
                results_dict[entry['index']] = {
                    'perplexity': None,
                    'generated_answer': None
                }

    # Build final lists in order
    print("\nPerplexity calculation completed.")
