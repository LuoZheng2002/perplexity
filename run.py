"""
LLM as Judge: Exploring the relationship between perplexity and preference
"""
import os

from collect_preference_local_direct import collect_preference_local_direct
from collect_preference_local_thinking import collect_preference_local_thinking
from generate_dataset import generate_answer_pair_datasets
os.environ["HF_HOME"] = "/work/nvme/bfdz/zluo8/huggingface"
import sys
from datasets import load_dataset
from openai import OpenAI
from dotenv import load_dotenv
import json
import re
import math
from parse_dataset import parse_dataset, prepare_answer_pairs_bilingual

from config import *
from models import create_model_interface

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

    print(f"\nCalculating perplexities using local LLM: {model_name}")
    print(f"Results will be written to {output_file}")

    def calculate_answer_perplexity(question, answer):
        """
        Calculate perplexity of an answer given a question.
        Uses chat template formatting with system, user, and assistant messages.
        Returns perplexity computed as exp(-avg_log_prob).
        """
        # Build formatted conversation text using model-specific interface
        try:
            full_chat_text = model_interface.build_messages_for_perplexity(
                tokenizer, question, answer
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

_global_model_name = None
_global_model = None
_global_tokenizer = None

if __name__ == "__main__":
    for config in configs:
        print("Processing configuration: ", config)
        pairs = generate_answer_pair_datasets(lang1=config.lang1, lang2=config.lang2, subject=config.subject)

        # Initialize model and tokenizer
        print("\n" + "="*60)
        print("STEP 3: Initializing Model and Tokenizer")
        print("="*60)

        # Get or create model and tokenizer with caching
        model_name = config.model.value
        if _global_model_name == model_name:
            # Reuse cached model and tokenizer
            print(f"Reusing cached model: {model_name}")
            model = _global_model
            tokenizer = _global_tokenizer
        else:
            # Initialize new model and tokenizer, update cache
            print(f"Initializing new model: {model_name}")
            model, tokenizer = initialize_model(model_name=model_name, device="cuda")
            _global_model_name = model_name
            _global_model = model
            _global_tokenizer = tokenizer

        # Create model interface for model-specific behavior
        model_interface = create_model_interface(model_name)
        print(f"Using model interface: {model_interface.__class__.__name__}")
        match config.result_type:
            case ResultType.PREFERENCE_DIRECT:
                collect_preference_local_direct(
                    pairs=pairs,
                    model=model,
                    tokenizer=tokenizer,
                    model_name=model_name,
                    model_interface=model_interface,
                    output_file=f"preferences_local_direct_{config.lang1}_{config.lang2}_{config.subject}.jsonl",
                    device="cuda"
                )
            case ResultType.PREFERENCE_THINKING:
                collect_preference_local_thinking(
                    pairs=pairs,
                    model=model,
                    tokenizer=tokenizer,
                    model_name=model_name,
                    model_interface=model_interface,
                    output_file=f"preferences_local_thinking_{config.lang1}_{config.lang2}_{config.subject}.jsonl",
                    device="cuda"
                )

