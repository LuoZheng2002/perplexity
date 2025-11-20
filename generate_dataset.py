

def generate_answer_pair_datasets(lang1, lang2):
    '''
    Load datasets from two languages and create bilingual answer pairs.
    If lang1 and lang2 are both not English, load the English dataset as well for questions.
    Question is always in English, while answers are extracted from lang1 and lang2 datasets.

    All input datasets have contents to be the same except for their languages.

    The input datasets are in the form of multiple choise questions with 4 options (A, B, C, D), where only 1 is correct.

    For each output dataset, we select answers in one language to be always correct, and answers in the other language to be always incorrect.

    The correct answer index can be retrieved from the input datasets. Then the incorrect answer is the one with index to be (correct_index + 1) % 4.
    For example, if the correct answer is B (index 1), then the incorrect answer is C (index 2).

    Answer 1 is always in a language whose abbreviation has a smaller lexicographical order. For example, for ("zh_cn", "en"), answer 1 is always in "en", and answer 2 is always in "zh_cn".

    For each call of this function, we generate two datasets where two languages take turns to be the correct answer language.

    The output datasets are saved to files named:
        pair_{lang1}_correct_{lang2}_incorrect.jsonl
        pair_{lang1}_incorrect_{lang2}_correct.jsonl
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
            'correct_answer': int, # 1 if lang1 is correct, 2 if lang2 is correct
            'subject': str,
        }
    '''

    