

def generate_answer_pair_datasets(lang1, lang2):
    '''
    Load datasets from two languages and create bilingual answer pairs.
    If lang1 and lang2 are both not English, load the English dataset as well for questions.
    Question is always in English, while answers are extracted from lang1 and lang2 datasets.

    All input datasets have contents to be the same except for their languages.

    The input datasets are in the form of multiple choise questions with 4 options (A, B, C, D), where only 1 is correct.

    For each output dataset, we select answers based on correctness settings.

    The correct answer index can be retrieved from the input datasets. Then the incorrect answer is the one with index to be (correct_index + 1) % 4.
    For example, if the correct answer is B (index 1), then the incorrect answer is C (index 2).

    Answer 1 is always in a language whose abbreviation has a smaller lexicographical order. For example, for ("zh_cn", "en"), answer 1 is always in "en", and answer 2 is always in "zh_cn".

    For each call of this function, we generate four datasets:
        1. lang1 correct, lang2 incorrect
        2. lang1 incorrect, lang2 correct
        3. both correct
        4. both incorrect

    The output datasets are saved to files named:
        pair_{lang1}_correct_{lang2}_incorrect.jsonl
        pair_{lang1}_incorrect_{lang2}_correct.jsonl
        pair_{lang1}_correct_{lang2}_correct.jsonl
        pair_{lang1}_incorrect_{lang2}_incorrect.jsonl
    Args:
        lang1: First language code (e.g., "zh_cn")
        lang2: Second language code (e.g., "en")

    Content to be written to the corresponding dataset files:
        List of pairs with structure:
        {
            'index': int,
            'question': str (English question),
            'answer1': str (answer in lang1),
            'answer2': str (answer in lang2),
            'lang1': str (language code),
            'lang2': str (language code),
            'correct_answer': int, # 0 if both incorrect, 1 if lang1 correct, 2 if lang2 correct, 3 if both correct
            'subject': str,
        }
    '''
    import json
    import os
    from parse_dataset import parse_dataset

    print(f"\n{'='*60}")
    print(f"Generating answer pair datasets for {lang1} and {lang2}")
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

    # Create four datasets
    dataset1 = []  # lang1 correct, lang2 incorrect
    dataset2 = []  # lang1 incorrect, lang2 correct
    dataset3 = []  # both correct
    dataset4 = []  # both incorrect

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

        # Create entry for dataset1 (lang1 correct, lang2 incorrect)
        entry1 = {
            'index': sample_en['original_index'],
            'question': sample_en['question'],
            'answer1': correct_answer1,
            'answer2': incorrect_answer2,
            'lang1': lang1,
            'lang2': lang2,
            'correct_answer': 1,
            'subject': sample_en['subject']
        }
        dataset1.append(entry1)

        # Create entry for dataset2 (lang1 incorrect, lang2 correct)
        entry2 = {
            'index': sample_en['original_index'],
            'question': sample_en['question'],
            'answer1': incorrect_answer1,
            'answer2': correct_answer2,
            'lang1': lang1,
            'lang2': lang2,
            'correct_answer': 2,
            'subject': sample_en['subject']
        }
        dataset2.append(entry2)

        # Create entry for dataset3 (both correct)
        entry3 = {
            'index': sample_en['original_index'],
            'question': sample_en['question'],
            'answer1': correct_answer1,
            'answer2': correct_answer2,
            'lang1': lang1,
            'lang2': lang2,
            'correct_answer': 3,  # both correct
            'subject': sample_en['subject']
        }
        dataset3.append(entry3)

        # Create entry for dataset4 (both incorrect)
        entry4 = {
            'index': sample_en['original_index'],
            'question': sample_en['question'],
            'answer1': incorrect_answer1,
            'answer2': incorrect_answer2,
            'lang1': lang1,
            'lang2': lang2,
            'correct_answer': 0,  # both incorrect
            'subject': sample_en['subject']
        }
        dataset4.append(entry4)

    if misaligned_count > 0:
        print(f"Warning: Skipped {misaligned_count} misaligned samples\n")

    # Create output directory if it doesn't exist
    os.makedirs("datasets", exist_ok=True)

    # Save datasets to JSONL files
    file1 = f"datasets/pair_{lang1}_correct_{lang2}_incorrect.jsonl"
    file2 = f"datasets/pair_{lang1}_incorrect_{lang2}_correct.jsonl"
    file3 = f"datasets/pair_{lang1}_correct_{lang2}_correct.jsonl"
    file4 = f"datasets/pair_{lang1}_incorrect_{lang2}_incorrect.jsonl"

    print(f"Saving datasets...")
    with open(file1, 'w', encoding='utf-8') as f:
        for entry in dataset1:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(f"  Saved {len(dataset1)} entries to {file1}")

    with open(file2, 'w', encoding='utf-8') as f:
        for entry in dataset2:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(f"  Saved {len(dataset2)} entries to {file2}")

    with open(file3, 'w', encoding='utf-8') as f:
        for entry in dataset3:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(f"  Saved {len(dataset3)} entries to {file3}")

    with open(file4, 'w', encoding='utf-8') as f:
        for entry in dataset4:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(f"  Saved {len(dataset4)} entries to {file4}")

    print(f"\n{'='*60}")
    print(f"Successfully generated {len(dataset1)} answer pairs for each of 4 datasets")
    print(f"{'='*60}\n")

    return dataset1, dataset2, dataset3, dataset4


if __name__ == "__main__":
    # Example usage
    generate_answer_pair_datasets("zh_cn", "en")