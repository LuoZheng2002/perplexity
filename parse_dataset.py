"""
Parse and normalize dataset structure across different languages.
Handles inconsistent field naming conventions across language datasets.
"""

from datasets import load_dataset


def parse_dataset(language="en", num_samples=None):
    """
    Load and parse dataset for a given language, returning normalized structure.

    The raw datasets have inconsistent field naming:
    - English: 'question', 'choices' (list), 'answer' (int index), 'subject'
    - Chinese (zh_cn): 'Question', 'A', 'B', 'C', 'D' (individual choice fields),
                       'Answer' (letter), 'Subject'

    This function normalizes to a consistent structure:
    {
        'original_index': int,
        'question': str,
        'choices': dict with keys 'A'/'B'/'C'/'D' or '0'/'1'/'2'/'3',
        'answer': str (the correct choice text),
        'answer_letter': str (A/B/C/D or 0/1/2/3),
        'subject': str
    }

    Args:
        language: Language code (e.g., "en", "zh_cn", "ar_xy")
        num_samples: Maximum number of samples to load (None = all)
        subject: Filter by subject (e.g., "abstract_algebra"). If None, load all subjects.

    Returns:
        List of normalized sample dictionaries
    """
    print(f"Loading dataset for language: {language}...")

    ds = load_dataset("willchow66/mmmlu-intersection-filtered", language)

    # Get the training split
    train_split = ds['train']

    # Determine the number of samples to process
    total_samples = len(train_split)
    samples_to_process = total_samples if num_samples is None else min(num_samples, total_samples)

    print(f"Processing up to {samples_to_process}/{total_samples} samples...")

    normalized_data = []
    processed_count = 0

    for i in range(total_samples):
        if len(normalized_data) >= samples_to_process:
            break

        sample = train_split[i]

        normalized_sample = _normalize_sample(sample, language)
        normalized_data.append(normalized_sample)
        processed_count += 1

    print(f"Successfully loaded and normalized {len(normalized_data)} samples")
    return normalized_data


def _normalize_sample(sample, language):
    """
    Normalize a single sample based on the language format.

    Args:
        sample: Raw sample from the dataset
        language: Language code to determine format

    Returns:
        Normalized sample dictionary
    """
    if language == "en":
        return _normalize_english(sample)
    elif language == "zh_cn":
        return _normalize_chinese(sample)
    else:
        # For other languages, try to infer the format
        if "choices" in sample:
            return _normalize_english(sample)
        elif "A" in sample and "B" in sample and "C" in sample and "D" in sample:
            return _normalize_chinese(sample)
        else:
            raise ValueError(f"Unknown dataset format for language: {language}")


def _normalize_english(sample):
    """
    Normalize English format dataset.
    Format: 'question', 'choices' (list), 'answer' (int index), 'subject'
    """
    # answer is an index (0, 1, 2, 3)
    answer_idx = sample['answer']
    choices_list = sample['choices']

    # Create choices dict with letter keys (A, B, C, D)
    choices_dict = {chr(ord('A') + i): choice for i, choice in enumerate(choices_list)}

    answer_text = choices_list[answer_idx]
    answer_letter = chr(ord('A') + answer_idx)  # Convert index to letter (A, B, C, D)

    return {
        'original_index': sample['original_index'],
        'question': sample['question'],
        'choices': choices_dict,  # {'A': '...', 'B': '...', 'C': '...', 'D': '...'}
        'answer': answer_text,  # The actual answer text
        'answer_letter': answer_letter,  # A/B/C/D
        'answer_idx': answer_idx,  # 0/1/2/3
        'subject': sample['subject']
    }


def _normalize_chinese(sample):
    """
    Normalize Chinese format dataset.
    Format: 'Question', 'A', 'B', 'C', 'D' (individual fields),
            'Answer' (letter), 'Subject'
    """
    # Build choices dict from individual fields
    choices_dict = {
        'A': sample['A'],
        'B': sample['B'],
        'C': sample['C'],
        'D': sample['D']
    }

    # answer is a letter (A, B, C, D)
    answer_letter = sample['Answer']
    answer_idx = ord(answer_letter) - ord('A')  # Convert letter to index (0, 1, 2, 3)
    answer_text = choices_dict[answer_letter]

    return {
        'original_index': sample['original_index'],
        'question': sample['Question'],
        'choices': choices_dict,  # {'A': '...', 'B': '...', 'C': '...', 'D': '...'}
        'answer': answer_text,  # The actual answer text
        'answer_letter': answer_letter,  # A/B/C/D
        'answer_idx': answer_idx,  # 0/1/2/3
        'subject': sample['Subject']
    }


def verify_alignment(lang1="en", lang2="zh_cn", num_samples=20):
    """
    Verify that two language datasets are aligned (same questions, same answers).

    Args:
        lang1: First language code
        lang2: Second language code
        num_samples: Number of samples to check

    Returns:
        Boolean indicating if all checked samples are aligned
    """
    print(f"\nVerifying alignment between {lang1} and {lang2}...")

    data1 = parse_dataset(lang1, num_samples)
    data2 = parse_dataset(lang2, num_samples)

    all_aligned = True
    misaligned_count = 0

    for i in range(min(len(data1), len(data2))):
        sample1 = data1[i]
        sample2 = data2[i]

        # Check if original_index matches
        if sample1['original_index'] != sample2['original_index']:
            print(f"  Sample {i}: Misaligned indices - {sample1['original_index']} vs {sample2['original_index']}")
            misaligned_count += 1
            all_aligned = False
            continue

        # Check if answer indices match
        if sample1['answer_idx'] != sample2['answer_idx']:
            print(f"  Sample {i}: Misaligned answers - {sample1['answer_idx']} vs {sample2['answer_idx']}")
            misaligned_count += 1
            all_aligned = False

    if all_aligned:
        print(f"✓ All {min(len(data1), len(data2))} checked samples are aligned")
    else:
        print(f"✗ Found {misaligned_count} misaligned samples")

    return all_aligned


if __name__ == "__main__":
    # Example usage
    print("="*60)
    print("PARSING DATASET - Example Usage")
    print("="*60)

    # Parse English dataset
    print("\nParsing English dataset:")
    en_data = parse_dataset("en", num_samples=3)
    print(f"\nFirst English sample:")
    for key, value in en_data[0].items():
        if key != 'choices':
            print(f"  {key}: {value}")
        else:
            print(f"  {key}:")
            for choice_key, choice_val in value.items():
                print(f"    {choice_key}: {choice_val}")

    # Parse Chinese dataset
    print("\n" + "="*60)
    print("\nParsing Chinese dataset:")
    zh_data = parse_dataset("zh_cn", num_samples=3)
    print(f"\nFirst Chinese sample:")
    for key, value in zh_data[0].items():
        if key != 'choices':
            print(f"  {key}: {value}")
        else:
            print(f"  {key}:")
            for choice_key, choice_val in value.items():
                print(f"    {choice_key}: {choice_val}")

    # Parse dataset filtered by subject
    print("\n" + "="*60)
    print("\nParsing English dataset filtered by subject (abstract_algebra):")
    en_data_subject = parse_dataset("en", num_samples=3, subject="abstract_algebra")
    print(f"\nFirst sample from abstract_algebra:")
    for key, value in en_data_subject[0].items():
        if key != 'choices':
            print(f"  {key}: {value}")
        else:
            print(f"  {key}:")
            for choice_key, choice_val in value.items():
                print(f"    {choice_key}: {choice_val}")

    # Verify alignment
    print("\n" + "="*60)
    verify_alignment("en", "zh_cn", num_samples=10)


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
            'answer1': str (correct answer from lang1),
            'answer2': str (correct answer from lang2),
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
        answer1 = sample_lang1['answer']  # Already the correct answer text
        answer2 = sample_lang2['answer']  # Already the correct answer text

        pairs.append({
            'question': sample_en['question'],  # Always English
            'answer1': answer1,
            'answer2': answer2,
            'lang1': lang1,
            'lang2': lang2,
            'subject': sample_en['subject'],
            'original_index': sample_en['original_index']
        })

    if misaligned_count > 0:
        print(f"Warning: Skipped {misaligned_count} misaligned samples")

    print(f"Created {len(pairs)} answer pairs comparing {lang1} vs {lang2} (with English questions)")
    return pairs