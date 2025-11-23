

def generate_answer_datasets(lang1, lang2):
    '''
    Load datasets from two languages and create individual answer entries.
    If lang1 and lang2 are both not English, load the English dataset as well for questions.
    Question is always in English, while answers are extracted from lang1 and lang2 datasets.

    All input datasets have contents to be the same except for their languages.

    The input datasets are in the form of multiple choice questions with 4 options (A, B, C, D), where only 1 is correct.

    The correct answer index can be retrieved from the input datasets. Then the incorrect answer is the one with index to be (correct_index + 1) % 4.
    For example, if the correct answer is B (index 1), then the incorrect answer is C (index 2).

    For each call of this function, we generate four datasets (individual entries, not pairs):
        1. lang1 correct answers
        2. lang1 incorrect answers
        3. lang2 correct answers
        4. lang2 incorrect answers

    The output datasets are saved to files named:
        {lang}_correct.jsonl
        {lang}_incorrect.jsonl

    Args:
        lang1: First language code (e.g., "zh_cn")
        lang2: Second language code (e.g., "en")

    Content to be written to the corresponding dataset files:
        List of entries with structure:
        {
            'index': int,
            'question': str (English question),
            'answer': str,
            'lang': str (language code),
            'is_correct': bool,
            'subject': str,
        }
    '''
    import json
    import os
    from parse_dataset import parse_dataset

    print(f"\n{'='*60}")
    print(f"Generating answer datasets for {lang1} and {lang2}")
    print(f"{'='*60}\n")

    # Load datasets
    print("Loading datasets...")
    [lang1, lang2] = sorted([lang1, lang2])
    data_lang1 = parse_dataset(lang1, num_samples=None)
    data_lang2 = parse_dataset(lang2, num_samples=None)

    # Load English dataset for questions if neither language is English
    if lang1 != "en" and lang2 != "en":
        data_en = parse_dataset("en", num_samples=None)
    else:
        # Use the English dataset that's already loaded
        data_en = data_lang1 if lang1 == "en" else data_lang2

    # Verify dataset alignment
    min_length = min(len(data_lang1), len(data_lang2), len(data_en))
    print(f"Dataset sizes: {lang1}={len(data_lang1)}, {lang2}={len(data_lang2)}, en={len(data_en)}")
    print(f"Will process {min_length} samples\n")

    # Create four datasets (individual entries)
    dataset_lang1_correct = []
    dataset_lang1_incorrect = []
    dataset_lang2_correct = []
    dataset_lang2_incorrect = []

    misaligned_count = 0

    for i in range(min_length):
        sample_lang1 = data_lang1[i]
        sample_lang2 = data_lang2[i]
        sample_en = data_en[i]

        # Verify alignment
        if sample_lang1['original_index'] != sample_lang2['original_index'] or \
           sample_lang1['original_index'] != sample_en['original_index']:
            misaligned_count += 1
            print(f"Warning: Misaligned samples at index {i}, skipping...")
            continue

        # Get correct answer index
        correct_idx = sample_lang1['answer_idx']  # Should be same across all languages

        # Calculate incorrect answer index: (correct_idx + 1) % 4
        incorrect_idx = (correct_idx + 1) % 4

        # Convert index to letter
        incorrect_letter = chr(ord('A') + incorrect_idx)

        # Extract correct and incorrect answers from both languages
        correct_answer1 = sample_lang1['answer']
        incorrect_answer1 = sample_lang1['choices'][incorrect_letter]

        correct_answer2 = sample_lang2['answer']
        incorrect_answer2 = sample_lang2['choices'][incorrect_letter]

        # Create entry for lang1 correct
        entry_lang1_correct = {
            'index': sample_en['original_index'],
            'question': sample_en['question'],
            'answer': correct_answer1,
            'lang': lang1,
            'is_correct': True,
            'subject': sample_en['subject']
        }
        dataset_lang1_correct.append(entry_lang1_correct)

        # Create entry for lang1 incorrect
        entry_lang1_incorrect = {
            'index': sample_en['original_index'],
            'question': sample_en['question'],
            'answer': incorrect_answer1,
            'lang': lang1,
            'is_correct': False,
            'subject': sample_en['subject']
        }
        dataset_lang1_incorrect.append(entry_lang1_incorrect)

        # Create entry for lang2 correct
        entry_lang2_correct = {
            'index': sample_en['original_index'],
            'question': sample_en['question'],
            'answer': correct_answer2,
            'lang': lang2,
            'is_correct': True,
            'subject': sample_en['subject']
        }
        dataset_lang2_correct.append(entry_lang2_correct)

        # Create entry for lang2 incorrect
        entry_lang2_incorrect = {
            'index': sample_en['original_index'],
            'question': sample_en['question'],
            'answer': incorrect_answer2,
            'lang': lang2,
            'is_correct': False,
            'subject': sample_en['subject']
        }
        dataset_lang2_incorrect.append(entry_lang2_incorrect)

    if misaligned_count > 0:
        print(f"Warning: Skipped {misaligned_count} misaligned samples\n")

    # Create output directory if it doesn't exist
    os.makedirs("datasets", exist_ok=True)

    # Save datasets to JSONL files
    file_lang1_correct = f"datasets/{lang1}_correct.jsonl"
    file_lang1_incorrect = f"datasets/{lang1}_incorrect.jsonl"
    file_lang2_correct = f"datasets/{lang2}_correct.jsonl"
    file_lang2_incorrect = f"datasets/{lang2}_incorrect.jsonl"

    print(f"Saving datasets...")
    with open(file_lang1_correct, 'w', encoding='utf-8') as f:
        for entry in dataset_lang1_correct:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(f"  Saved {len(dataset_lang1_correct)} entries to {file_lang1_correct}")

    with open(file_lang1_incorrect, 'w', encoding='utf-8') as f:
        for entry in dataset_lang1_incorrect:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(f"  Saved {len(dataset_lang1_incorrect)} entries to {file_lang1_incorrect}")

    with open(file_lang2_correct, 'w', encoding='utf-8') as f:
        for entry in dataset_lang2_correct:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(f"  Saved {len(dataset_lang2_correct)} entries to {file_lang2_correct}")

    with open(file_lang2_incorrect, 'w', encoding='utf-8') as f:
        for entry in dataset_lang2_incorrect:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(f"  Saved {len(dataset_lang2_incorrect)} entries to {file_lang2_incorrect}")

    print(f"\n{'='*60}")
    print(f"Successfully generated {len(dataset_lang1_correct)} entries for each of 4 datasets")
    print(f"{'='*60}\n")

    return dataset_lang1_correct, dataset_lang1_incorrect, dataset_lang2_correct, dataset_lang2_incorrect


if __name__ == "__main__":
    # Example usage
    generate_answer_datasets("zh_cn", "en")
