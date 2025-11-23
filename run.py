"""
LLM as Judge: Exploring the relationship between perplexity and preference
"""
import os

from collect_perplexity_local import collect_perplexity_local
from collect_preference_local_direct import collect_preference_local_direct
from collect_preference_local_thinking import collect_preference_local_thinking
from generate_dataset import generate_answer_datasets
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


def load_entries(file_path):
    """Load entries from a JSONL file."""
    entries = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def combine_entries_to_pairs(entries1, entries2, lang1, lang2):
    """
    Combine two lists of entries into pairs by matching indices.

    Args:
        entries1: List of entries for answer1 (lang1)
        entries2: List of entries for answer2 (lang2)
        lang1: Language code for answer1
        lang2: Language code for answer2

    Returns:
        List of pairs with structure:
        {
            'index': int,
            'question': str,
            'answer1': str,
            'answer2': str,
            'lang1': str,
            'lang2': str,
            'is_correct1': bool,
            'is_correct2': bool,
            'subject': str,
        }
    """
    # Index entries by their index field
    entries1_by_index = {e['index']: e for e in entries1}
    entries2_by_index = {e['index']: e for e in entries2}

    # Find common indices
    common_indices = set(entries1_by_index.keys()) & set(entries2_by_index.keys())

    pairs = []
    for idx in sorted(common_indices):
        e1 = entries1_by_index[idx]
        e2 = entries2_by_index[idx]

        pair = {
            'index': idx,
            'question': e1['question'],
            'answer1': e1['answer'],
            'answer2': e2['answer'],
            'lang1': lang1,
            'lang2': lang2,
            'is_correct1': e1['is_correct'],
            'is_correct2': e2['is_correct'],
            'subject': e1.get('subject', ''),
        }
        pairs.append(pair)

    return pairs


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
    # Create result directory if it doesn't exist
    os.makedirs("result", exist_ok=True)

    for config in configs:
        print("Processing configuration: ", config)

        # Determine alphabetical order for language codes
        sorted_langs = sorted([config.lang1, config.lang2])
        first_lang = sorted_langs[0]
        second_lang = sorted_langs[1]

        # Load individual entry datasets
        entries_lang1_correct = load_entries(f"datasets/{first_lang}_correct.jsonl")
        entries_lang1_incorrect = load_entries(f"datasets/{first_lang}_incorrect.jsonl")
        entries_lang2_correct = load_entries(f"datasets/{second_lang}_correct.jsonl")
        entries_lang2_incorrect = load_entries(f"datasets/{second_lang}_incorrect.jsonl")

        print(f"Loaded {len(entries_lang1_correct)} entries from datasets/{first_lang}_correct.jsonl")
        print(f"Loaded {len(entries_lang1_incorrect)} entries from datasets/{first_lang}_incorrect.jsonl")
        print(f"Loaded {len(entries_lang2_correct)} entries from datasets/{second_lang}_correct.jsonl")
        print(f"Loaded {len(entries_lang2_incorrect)} entries from datasets/{second_lang}_incorrect.jsonl")

        # Combine entries into pairs for preference methods
        # 4 pair combinations:
        # 1. lang1 correct, lang2 incorrect
        # 2. lang1 incorrect, lang2 correct
        # 3. both correct
        # 4. both incorrect
        pairs_lang1_correct_lang2_incorrect = combine_entries_to_pairs(
            entries_lang1_correct, entries_lang2_incorrect, first_lang, second_lang
        )
        pairs_lang1_incorrect_lang2_correct = combine_entries_to_pairs(
            entries_lang1_incorrect, entries_lang2_correct, first_lang, second_lang
        )
        pairs_both_correct = combine_entries_to_pairs(
            entries_lang1_correct, entries_lang2_correct, first_lang, second_lang
        )
        pairs_both_incorrect = combine_entries_to_pairs(
            entries_lang1_incorrect, entries_lang2_incorrect, first_lang, second_lang
        )

        print(f"Created {len(pairs_lang1_correct_lang2_incorrect)} pairs for {first_lang}_correct_{second_lang}_incorrect")
        print(f"Created {len(pairs_lang1_incorrect_lang2_correct)} pairs for {first_lang}_incorrect_{second_lang}_correct")
        print(f"Created {len(pairs_both_correct)} pairs for {first_lang}_correct_{second_lang}_correct")
        print(f"Created {len(pairs_both_incorrect)} pairs for {first_lang}_incorrect_{second_lang}_incorrect")

        # Get or create model and tokenizer with caching
        model_name = config.model.value
        match config.model:
            case Model.GRANITE_3_1_8B_INSTRUCT:
                display_model_name = "granite_3_1_8b"
            case Model.QWEN_2_5_7B_INSTRUCT:
                display_model_name = "qwen_2_5_7b"
            case Model.QWEN_2_5_14B_INSTRUCT:
                display_model_name = "qwen_2_5_14b"
            case Model.QWEN_2_5_32B_INSTRUCT:
                display_model_name = "qwen_2_5_32b"
            case Model.QWEN_2_5_72B_INSTRUCT:
                display_model_name = "qwen_2_5_72b"

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
                # Process pairs for preference_direct
                for pairs, dataset_suffix in [
                    (pairs_lang1_correct_lang2_incorrect, f"{first_lang}_correct_{second_lang}_incorrect"),
                    (pairs_lang1_incorrect_lang2_correct, f"{first_lang}_incorrect_{second_lang}_correct"),
                    (pairs_both_correct, f"{first_lang}_correct_{second_lang}_correct"),
                    (pairs_both_incorrect, f"{first_lang}_incorrect_{second_lang}_incorrect")
                ]:
                    output_dir = f"result/{display_model_name}/preferences_local_direct"
                    os.makedirs(output_dir, exist_ok=True)
                    collect_preference_local_direct(
                        pairs=pairs,
                        model=model,
                        tokenizer=tokenizer,
                        model_name=model_name,
                        model_interface=model_interface,
                        output_file=f"{output_dir}/{dataset_suffix}.jsonl",
                        device="cuda"
                    )

            # case ResultType.PREFERENCE_THINKING:
            #     # Process pairs for preference_thinking
            #     for pairs, dataset_suffix in [
            #         (pairs_lang1_correct_lang2_incorrect, f"{first_lang}_correct_{second_lang}_incorrect"),
            #         (pairs_lang1_incorrect_lang2_correct, f"{first_lang}_incorrect_{second_lang}_correct"),
            #         (pairs_both_correct, f"{first_lang}_correct_{second_lang}_correct"),
            #         (pairs_both_incorrect, f"{first_lang}_incorrect_{second_lang}_incorrect")
            #     ]:
            #         output_dir = f"result/{display_model_name}/preferences_local_thinking"
            #         os.makedirs(output_dir, exist_ok=True)
            #         collect_preference_local_thinking(
            #             pairs=pairs,
            #             model=model,
            #             tokenizer=tokenizer,
            #             model_name=model_name,
            #             model_interface=model_interface,
            #             output_file=f"{output_dir}/{dataset_suffix}.jsonl",
            #             device="cuda",
            #             batch_size=12
            #         )

            case ResultType.PERPLEXITY:
                # Process individual entries for perplexity
                for entries, entry_suffix in [
                    (entries_lang1_correct, f"{first_lang}_correct"),
                    (entries_lang1_incorrect, f"{first_lang}_incorrect"),
                    (entries_lang2_correct, f"{second_lang}_correct"),
                    (entries_lang2_incorrect, f"{second_lang}_incorrect")
                ]:
                    output_dir = f"result/{display_model_name}/perplexities_local"
                    os.makedirs(output_dir, exist_ok=True)
                    collect_perplexity_local(
                        entries=entries,
                        model=model,
                        tokenizer=tokenizer,
                        model_name=model_name,
                        model_interface=model_interface,
                        output_file=f"{output_dir}/{entry_suffix}.jsonl",
                        device="cuda"
                    )

        print("Collected results for configuration: ", config)
