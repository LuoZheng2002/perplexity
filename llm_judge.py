"""
LLM as Judge: Exploring the relationship between perplexity and preference
"""
import os
os.environ["HF_HOME"] = "/work/nvme/bfdz/zluo8/huggingface"
import sys
from datasets import load_dataset
from openai import OpenAI
from dotenv import load_dotenv
import json
import re
import math
from parse_dataset import parse_dataset

# Set UTF-8 encoding for console output (Windows fix)
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Load environment variables
load_dotenv()

def initialize_model(model_name="Qwen/Qwen2.5-7B-Instruct", device="cuda"):
    """
    Initialize model and tokenizer with the same configuration as test_chat_template.py.

    Args:
        model_name: Hugging Face model name
        device: Device to use ("cuda" or "cpu")

    Returns:
        Tuple of (model, tokenizer)
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError:
        print("Error: transformers library not found. Install it with: pip install transformers torch")
        raise

    print(f"Loading model: {model_name}")
    print(f"Using device: {device}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",         # stream to GPU, no CPU RAM bottleneck
            torch_dtype="auto",        # avoid expensive fp32→fp16 conversion
            low_cpu_mem_usage=True,    # avoids full-shard load into CPU
            use_safetensors=True       # skip .bin files if present
        )
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        raise

    # Set pad token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer



def prepare_answer_pairs_bilingual(lang1="zh_cn", lang2="en", subject=None, num_samples=100):
    """
    Load datasets from two languages and create bilingual answer pairs.
    Question is always in English, while answers are extracted from lang1 and lang2 datasets.

    Args:
        lang1: First language code (e.g., "zh_cn")
        lang2: Second language code (e.g., "en")
        num_samples: Number of samples to create

    Returns:
        List of pairs with structure:
        {
            'question': str (English question),
            'answer_lang1': str (correct answer from lang1),
            'answer_lang2': str (correct answer from lang2),
            'lang1': str (language code),
            'lang2': str (language code),
            'subject': str,
            'original_index': int
        }
    """
    print(f"\nLoading and parsing datasets...")

    # Parse datasets using the normalized parse_dataset function
    data_en = parse_dataset("en", num_samples=num_samples, subject=subject)
    data_lang1 = parse_dataset(lang1, num_samples=num_samples, subject=subject)
    data_lang2 = parse_dataset(lang2, num_samples=num_samples, subject=subject)

    print(f"Dataset sizes: en={len(data_en)}, {lang1}={len(data_lang1)}, {lang2}={len(data_lang2)}")

    pairs = []
    misaligned_count = 0

    for i in range(min(num_samples, len(data_en), len(data_lang1), len(data_lang2))):
        sample_en = data_en[i]
        sample_lang1 = data_lang1[i]
        sample_lang2 = data_lang2[i]

        # Verify they're the same question (same original_index)
        if sample_en['original_index'] != sample_lang1['original_index'] or \
           sample_en['original_index'] != sample_lang2['original_index']:
            misaligned_count += 1
            continue

        # Extract correct answers from both language datasets
        answer_lang1 = sample_lang1['answer']  # Already the correct answer text
        answer_lang2 = sample_lang2['answer']  # Already the correct answer text

        pairs.append({
            'question': sample_en['question'],  # Always English
            'answer_lang1': answer_lang1,
            'answer_lang2': answer_lang2,
            'lang1': lang1,
            'lang2': lang2,
            'subject': sample_en['subject'],
            'original_index': sample_en['original_index']
        })

    if misaligned_count > 0:
        print(f"Warning: Skipped {misaligned_count} misaligned samples")

    print(f"Created {len(pairs)} answer pairs comparing {lang1} vs {lang2} (with English questions)")
    return pairs


def collect_preferences(pairs, api_key, output_file="preferences.jsonl"):
    """
    Use GPT-4o-mini to judge which answer is better for each question.
    Returns list of preferences (1 if answer from lang1 is better, 2 if answer from lang2 is better)
    Writes results incrementally to output_file in JSON Lines format.
    """
    client = OpenAI(api_key=api_key)

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

    print(f"\nCollecting preferences from GPT-4o-mini...")
    print(f"Results will be written to {output_file}")

    # Open file in append mode
    with open(output_file, 'a', encoding='utf-8') as f:
        for i, pair in enumerate(pairs):
            # Skip if already processed
            if i in processed_indices:
                continue

            # Present the question in English with answers in different languages (without language labels)
            # Allow reasoning and require final answer in a box
            prompt = f"""Given the following question and two answers, which answer is better?

Question: {pair['question']}

Answer 1: {pair['answer_lang1']}
Answer 2: {pair['answer_lang2']}

Think through your reasoning, then provide your final decision in the following format:
\\boxed{{X}} where X is either 1 or 2."""

            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=500  # Allow more tokens for reasoning
                )

                raw_answer = response.choices[0].message.content.strip()

                # Extract the boxed answer
                import re
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
                    'raw_answer': raw_answer,  # Include full reasoning
                    'question': pair['question'],
                    'answer_lang1': pair['answer_lang1'],
                    'answer_lang2': pair['answer_lang2'],
                    'lang1': pair['lang1'],
                    'lang2': pair['lang2']
                }

                if error_msg:
                    result['error'] = error_msg

                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                f.flush()  # Flush to disk immediately

                results_dict[i] = preference

                if (len(results_dict)) % 10 == 0:
                    print(f"  Processed {len(results_dict)}/{len(pairs)} samples")

            except Exception as e:
                print(f"Error on sample {i}: {e}")
                # Write error result
                result = {
                    'index': i,
                    'preference': None,
                    'raw_answer': None,
                    'error': str(e)
                }
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                f.flush()
                results_dict[i] = None

    # Build final list in order
    preferences = [results_dict.get(i, None) for i in range(len(pairs))]
    print(f"Collected {len([p for p in preferences if p is not None])} valid preferences")
    return preferences


def collect_preferences_local(pairs, model, tokenizer, model_name, output_file="preferences_local.jsonl", device="cuda"):
    """
    Use a local LLM to judge which answer is better.

    Args:
        pairs: List of question-answer pairs
        model: Pre-loaded model instance
        tokenizer: Pre-loaded tokenizer instance
        model_name: Hugging Face model name (for logging/identification)
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

            # Present the question with both answers
            prompt = f"""Given the following question and two answers, which answer is better?

Question: {pair['question']}

Answer 1: {pair['answer_lang1']}
Answer 2: {pair['answer_lang2']}

Think through your reasoning, then provide your final decision in the following format:
\\boxed{{X}} where X is either 1 or 2."""

            try:
                # Tokenize the prompt
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
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


def find_assistant_answer_start(full_ids, prefix_ids):
    """
    Find the start position of assistant answer in the token sequence.

    Args:
        full_ids: List of all token IDs in the full conversation
        prefix_ids: List of token IDs representing the assistant prefix

    Returns:
        Index of the first token after the prefix

    Raises:
        ValueError: If prefix not found in full_ids
    """
    L = len(prefix_ids)
    for i in range(len(full_ids) - L + 1):
        if full_ids[i:i+L] == prefix_ids:
            return i + L  # the first token *after* the prefix
    raise ValueError("Assistant prefix not found in full_ids.")


def calculate_perplexity_local(pairs, model, tokenizer, model_name, output_file="perplexities_local.jsonl", device="cuda"):
    """
    Calculate the perplexity of each answer in the pairs using a local LLM.

    Perplexity is calculated by getting the average log probability of tokens in the answer
    given the question context. Lower perplexity indicates the model finds the answer more likely.

    Args:
        pairs: List of question-answer pairs
        model: Pre-loaded model instance
        tokenizer: Pre-loaded tokenizer instance
        model_name: Hugging Face model name (for logging/identification)
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

    print(f"\nCalculating perplexities using local LLM: {model_name}")
    print(f"Results will be written to {output_file}")

    def calculate_answer_perplexity(question, answer):
        """
        Calculate perplexity of an answer given a question.
        Uses chat template formatting with system, user, and assistant messages.
        Returns perplexity computed as exp(-avg_log_prob).
        """
        # Build messages for the full conversation
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]

        try:
            # Apply chat template to get the full conversation text
            full_chat_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        except Exception as e:
            print(f"Error applying chat template: {e}")
            # Fallback to simple concatenation if chat template not supported
            full_chat_text = f"{question}{answer}"
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
            # Use the assistant prefix to find the answer start position
            assistant_prefix = "<|im_start|>assistant\n"
            prefix_ids = tokenizer(assistant_prefix, add_special_tokens=False).input_ids
            answer_start = find_assistant_answer_start(full_ids, prefix_ids)
        except (ValueError, AttributeError):
            # Fallback: search for exact answer token match if prefix method fails
            # answer_start = None
            # for i in range(len(full_ids) - answer_len + 1):
            #     if full_ids[i:i + answer_len] == answer_tokens:
            #         answer_start = i
            #         break

            if answer_start is None:
                # If exact match not found, log a warning and return None
                print(f"Warning: Answer tokens not found in full tokenized output for question: {question[:50]}...")
                return None

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
                    pair['answer_lang1']
                )

                # Calculate perplexity for lang2 answer
                perplexity_lang2 = calculate_answer_perplexity(
                    pair['question'],
                    pair['answer_lang2']
                )

                # Write result immediately
                result = {
                    'index': i,
                    'perplexity_lang1': perplexity_lang1,
                    'perplexity_lang2': perplexity_lang2,
                    'question': pair['question'],
                    'answer_lang1': pair['answer_lang1'],
                    'answer_lang2': pair['answer_lang2'],
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
    perplexities_lang1 = [results_dict.get(i, {}).get('perplexity_lang1') for i in range(len(pairs))]
    perplexities_lang2 = [results_dict.get(i, {}).get('perplexity_lang2') for i in range(len(pairs))]

    print(f"Calculated perplexities for {len([p for p in perplexities_lang1 if p is not None])} samples")
    return perplexities_lang1, perplexities_lang2


def compare_results(preferences, perplexities_lang1, perplexities_lang2, lang1, lang2):
    """
    Compare preferences with perplexity rankings and calculate correlation.
    """
    print("\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)

    # Determine which answer has lower perplexity (better)
    perplexity_preferences = []
    for p1, p2 in zip(perplexities_lang1, perplexities_lang2):
        if p1 is not None and p2 is not None:
            perplexity_preferences.append(1 if p1 < p2 else 2)
        else:
            perplexity_preferences.append(None)

    # Calculate agreement
    valid_comparisons = 0
    agreement = 0
    lang1_preference_count = 0
    lang2_preference_count = 0
    lang1_perplexity_count = 0
    lang2_perplexity_count = 0

    for pref, perp_pref in zip(preferences, perplexity_preferences):
        if pref is not None and perp_pref is not None:
            valid_comparisons += 1

            if pref == 1:
                lang1_preference_count += 1
            else:
                lang2_preference_count += 1

            if perp_pref == 1:
                lang1_perplexity_count += 1
            else:
                lang2_perplexity_count += 1

            if pref == perp_pref:
                agreement += 1

    print(f"\nTotal valid comparisons: {valid_comparisons}")
    print(f"\nPreference Method (which answer LLM judges as better):")
    print(f"  {lang1} answers preferred: {lang1_preference_count}/{valid_comparisons} = {100*lang1_preference_count/valid_comparisons:.2f}%")
    print(f"  {lang2} answers preferred: {lang2_preference_count}/{valid_comparisons} = {100*lang2_preference_count/valid_comparisons:.2f}%")

    print(f"\nPerplexity Method (which answer has lower perplexity):")
    print(f"  {lang1} answers have lower perplexity: {lang1_perplexity_count}/{valid_comparisons} = {100*lang1_perplexity_count/valid_comparisons:.2f}%")
    print(f"  {lang2} answers have lower perplexity: {lang2_perplexity_count}/{valid_comparisons} = {100*lang2_perplexity_count/valid_comparisons:.2f}%")

    print(f"\nAgreement between methods:")
    print(f"  {agreement}/{valid_comparisons} = {100*agreement/valid_comparisons:.2f}%")

    # Calculate correlation
    from scipy.stats import pearsonr, spearmanr

    # Filter out None values
    valid_indices = [i for i in range(len(preferences))
                     if preferences[i] is not None and perplexity_preferences[i] is not None]

    if len(valid_indices) > 1:
        pref_values = [preferences[i] for i in valid_indices]
        perp_values = [perplexity_preferences[i] for i in valid_indices]

        try:
            # Check if there's variance
            if len(set(pref_values)) > 1 and len(set(perp_values)) > 1:
                pearson_corr, pearson_p = pearsonr(pref_values, perp_values)
                spearman_corr, spearman_p = spearmanr(pref_values, perp_values)

                print(f"\nCorrelation statistics:")
                print(f"  Pearson correlation: {pearson_corr:.3f} (p={pearson_p:.4f})")
                print(f"  Spearman correlation: {spearman_corr:.3f} (p={spearman_p:.4f})")
            else:
                print("\nCould not calculate correlation (insufficient variance)")
        except Exception as e:
            print(f"\nCould not calculate correlation: {e}")

    print("\n" + "="*60)

    # Show some examples
    print("\nExample comparisons (first 5):")
    for i in range(min(5, len(preferences))):
        if preferences[i] is not None and perplexity_preferences[i] is not None:
            print(f"\nSample {i}:")
            print(f"  Preference method chose: {lang1 if preferences[i] == 1 else lang2}")
            print(f"  Perplexity method chose: {lang1 if perplexity_preferences[i] == 1 else lang2}")
            print(f"  Match: {'✓' if preferences[i] == perplexity_preferences[i] else '✗'}")


if __name__ == "__main__":
    # # Load API key
    # api_key = os.getenv("OPENAI_API_KEY")
    # if not api_key:
    #     print("Error: OPENAI_API_KEY not found in .env file")
    #     exit(1)



    # Prepare answer pairs (comparing zh_cn vs en)
    print("\n" + "="*60)
    print("STEP 2: Preparing Answer Pairs")
    print("="*60)
    pairs = prepare_answer_pairs_bilingual(lang1="zh_cn", lang2="en", subject="philosophy", num_samples=20)

    # Show example pairs
    if pairs:
        print("\nExample pair:")
        print(f"Question (en): {pairs[0]['question']}")
        print(f"Answer 1 ({pairs[0]['lang1']}): {pairs[0]['answer_lang1']}")
        print(f"Answer 2 ({pairs[0]['lang2']}): {pairs[0]['answer_lang2']}")

    # Initialize model and tokenizer
    print("\n" + "="*60)
    print("STEP 3: Initializing Model and Tokenizer")
    print("="*60)
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    model, tokenizer = initialize_model(model_name=model_name, device="cuda")

    # Collect preferences
    print("\n" + "="*60)
    print("STEP 4: Collecting Preferences")
    print("="*60)
    preferences = collect_preferences_local(pairs, model, tokenizer, model_name, output_file="preferences_local.jsonl")

    # Calculate perplexities
    print("\n" + "="*60)
    print("STEP 5: Calculating Perplexities")
    print("="*60)
    perplexities_lang1, perplexities_lang2 = calculate_perplexity_local(pairs, model, tokenizer, model_name, output_file="perplexities_local.jsonl")

    # Compare results
    print("\n" + "="*60)
    print("STEP 6: Analyzing Results")
    print("="*60)
    compare_results(preferences, perplexities_lang1, perplexities_lang2, "zh_cn", "en")
