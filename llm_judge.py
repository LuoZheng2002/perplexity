"""
LLM as Judge: Exploring the relationship between perplexity and preference
"""

import os
import sys
from datasets import load_dataset
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
import json

# Set UTF-8 encoding for console output (Windows fix)
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Load environment variables
load_dotenv()

def explore_dataset(lang="ar_xy"):
    """Load and explore the dataset structure"""
    print(f"Loading dataset with language: {lang}...")
    ds = load_dataset("willchow66/mmmlu-intersection-filtered", lang)

    print("\nDataset structure:")
    print(f"Keys: {ds.keys()}")

    # Check the first split
    for split_name in ds.keys():
        print(f"\n{split_name} split:")
        print(f"Number of samples: {len(ds[split_name])}")
        print(f"Features: {ds[split_name].features}")
        print(f"\nFirst 3 samples:")
        for i in range(min(3, len(ds[split_name]))):
            print(f"\n--- Sample {i} ---")
            sample = ds[split_name][i]
            for key, value in sample.items():
                print(f"{key}: {value}")
        break  # Only show first split for now

    return ds

def prepare_answer_pairs_bilingual(lang1="zh_cn", lang2="en", num_samples=100):
    """
    Load three language datasets and create answer pairs.
    Question is always in English, while answers are from lang1 and lang2.
    """
    # Always load English for questions
    print(f"\nLoading en dataset (for questions)...")
    ds_en = load_dataset("willchow66/mmmlu-intersection-filtered", "en")

    print(f"Loading {lang1} dataset (for answer 1)...")
    ds1 = load_dataset("willchow66/mmmlu-intersection-filtered", lang1)

    print(f"Loading {lang2} dataset (for answer 2)...")
    ds2 = load_dataset("willchow66/mmmlu-intersection-filtered", lang2)

    train_en = ds_en['train']
    train1 = ds1['train']
    train2 = ds2['train']

    # Make sure datasets align (same original_index)
    print(f"\nDataset sizes: en={len(train_en)}, {lang1}={len(train1)}, {lang2}={len(train2)}")

    pairs = []

    for i in range(min(num_samples, len(train_en), len(train1), len(train2))):
        sample_en = train_en[i]
        sample1 = train1[i]
        sample2 = train2[i]

        # Verify they're the same question (same original_index)
        if sample_en['original_index'] != sample1['original_index'] or \
           sample_en['original_index'] != sample2['original_index']:
            print(f"Warning: Misaligned samples at index {i}")
            continue

        # Get correct answers from both answer languages
        correct_letter1 = sample1['Answer']
        correct_answer1 = sample1[correct_letter1]

        correct_letter2 = sample2['Answer']
        correct_answer2 = sample2[correct_letter2]

        # Get question in English
        question_en = sample_en['Question']

        pairs.append({
            'question': question_en,  # Always English
            'answer_lang1': correct_answer1,
            'answer_lang2': correct_answer2,
            'lang1': lang1,
            'lang2': lang2,
            'subject': sample_en['Subject']
        })

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


def calculate_perplexity(pairs, api_key, output_file="perplexities.jsonl"):
    """
    Calculate perplexity for each answer using teacher forcing.
    For each answer, we prompt the LLM with the English question and measure the log probability
    of generating the given answer.
    Writes results incrementally to output_file in JSON Lines format.

    Note: Since OpenAI doesn't provide token-level probabilities in chat API,
    we'll use the completion API or estimate likelihood.
    """
    import re
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
                    results_dict[idx] = {
                        'perplexity_lang1': result.get('perplexity_lang1'),
                        'perplexity_lang2': result.get('perplexity_lang2')
                    }
        print(f"Found {len(processed_indices)} already processed samples")

    print("\nCalculating perplexities...")
    print("Note: Using likelihood scoring as proxy for perplexity")
    print(f"Results will be written to {output_file}")

    # Open file in append mode
    with open(output_file, 'a', encoding='utf-8') as f:
        for i, pair in enumerate(pairs):
            # Skip if already processed
            if i in processed_indices:
                continue

            try:
                # For answer in lang1: ask model to rate likelihood given English question
                prompt1 = f"""Given this question, rate how likely this answer is on a scale of 0-100.

Question: {pair['question']}
Answer: {pair['answer_lang1']}

Respond with only a number between 0 and 100."""

                response1 = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt1}],
                    temperature=0,
                    max_tokens=10
                )

                score1_text = response1.choices[0].message.content.strip()
                numbers = re.findall(r'\d+\.?\d*', score1_text)
                score1 = float(numbers[0]) if numbers else 50.0

                # Lower perplexity = better (inverse of score)
                perplexity1 = 100.0 / (score1 + 1)

                # For answer in lang2: ask model to rate likelihood given English question
                prompt2 = f"""Given this question, rate how likely this answer is on a scale of 0-100.

Question: {pair['question']}
Answer: {pair['answer_lang2']}

Respond with only a number between 0 and 100."""

                response2 = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt2}],
                    temperature=0,
                    max_tokens=10
                )

                score2_text = response2.choices[0].message.content.strip()
                numbers = re.findall(r'\d+\.?\d*', score2_text)
                score2 = float(numbers[0]) if numbers else 50.0

                perplexity2 = 100.0 / (score2 + 1)

                # Write result immediately
                result = {
                    'index': i,
                    'perplexity_lang1': perplexity1,
                    'perplexity_lang2': perplexity2,
                    'score_lang1': score1,
                    'score_lang2': score2,
                    'question': pair['question'],
                    'answer_lang1': pair['answer_lang1'],
                    'answer_lang2': pair['answer_lang2'],
                    'lang1': pair['lang1'],
                    'lang2': pair['lang2']
                }
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                f.flush()  # Flush to disk immediately

                results_dict[i] = {
                    'perplexity_lang1': perplexity1,
                    'perplexity_lang2': perplexity2
                }

                if (len(results_dict)) % 10 == 0:
                    print(f"  Processed {len(results_dict)}/{len(pairs)} samples")

            except Exception as e:
                print(f"Error on sample {i}: {e}")
                # Write error result
                result = {
                    'index': i,
                    'perplexity_lang1': None,
                    'perplexity_lang2': None,
                    'error': str(e)
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
    # Load API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in .env file")
        exit(1)

    # Explore datasets
    print("="*60)
    print("STEP 1: Exploring Datasets")
    print("="*60)
    print("\nEnglish (en) dataset (used for questions):")
    explore_dataset("en")

    # Prepare answer pairs (comparing zh_cn vs en)
    print("\n" + "="*60)
    print("STEP 2: Preparing Answer Pairs")
    print("="*60)
    pairs = prepare_answer_pairs_bilingual(lang1="zh_cn", lang2="en", num_samples=20)

    # Show example pairs
    if pairs:
        print("\nExample pair:")
        print(f"Question (en): {pairs[0]['question']}")
        print(f"Answer 1 ({pairs[0]['lang1']}): {pairs[0]['answer_lang1']}")
        print(f"Answer 2 ({pairs[0]['lang2']}): {pairs[0]['answer_lang2']}")

    # Collect preferences
    print("\n" + "="*60)
    print("STEP 3: Collecting Preferences")
    print("="*60)
    preferences = collect_preferences(pairs, api_key, output_file="preferences.jsonl")

    # Calculate perplexities
    print("\n" + "="*60)
    print("STEP 4: Calculating Perplexities")
    print("="*60)
    perplexities_lang1, perplexities_lang2 = calculate_perplexity(pairs, api_key, output_file="perplexities.jsonl")

    # Compare results
    print("\n" + "="*60)
    print("STEP 5: Analyzing Results")
    print("="*60)
    compare_results(preferences, perplexities_lang1, perplexities_lang2, "zh_cn", "en")
