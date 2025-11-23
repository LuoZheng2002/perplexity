import json
from pathlib import Path
from typing import Callable, Optional

from calculate_accuracy import calculate_accuracy
from calculate_correlation import calculate_pearson_correlation
from calculate_bias_binary import calculate_bias_binary
from calculate_bias_continuous import calculate_bias_continuous


def load_jsonl(file_path, subject_filter: Optional[Callable[[str], bool]] = None):
    """
    Load JSONL file and return a dictionary indexed by 'index' field.

    Args:
        file_path: Path to the JSONL file
        subject_filter: Optional function that takes a subject name and returns True to keep the sample

    Returns:
        Dict mapping index to sample dict
    """
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                # Filter by subject if filter is specified
                if subject_filter is not None and not subject_filter(item.get('subject', '')):
                    continue
                data[item['index']] = item
    return data


def load_jsonl_as_list(file_path, subject_filter: Optional[Callable[[str], bool]] = None):
    """
    Load JSONL file and return a list of samples.

    Args:
        file_path: Path to the JSONL file
        subject_filter: Optional function that takes a subject name and returns True to keep the sample

    Returns:
        List of sample dicts
    """
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                # Filter by subject if filter is specified
                if subject_filter is not None and not subject_filter(item.get('subject', '')):
                    continue
                samples.append(item)
    return samples


def find_model_dirs(result_dir):
    """Find all model directories in the result directory."""
    model_dirs = []
    for item in result_dir.iterdir():
        if item.is_dir() and item.name not in ['accuracy', 'correlation', 'bias', 'bias_continuous', 'metrics']:
            model_dirs.append(item)
    return model_dirs


def find_result_files(model_dir):
    """
    Find all result files for a model.
    Returns a dict with keys: perplexities_local, preferences_local_direct, preferences_local_thinking
    Each value is a dict mapping lang_pair to file path.
    """
    result_files = {
        'perplexities_local': {},
        'preferences_local_direct': {},
        'preferences_local_thinking': {}
    }

    for result_type in result_files.keys():
        type_dir = model_dir / result_type
        if type_dir.exists():
            for file in type_dir.glob("*.jsonl"):
                lang_pair = file.stem  # e.g., "en_correct_zh_cn_incorrect"
                result_files[result_type][lang_pair] = file

    return result_files


def get_correct_preference(lang_pair):
    """
    Determine which preference is correct based on lang_pair.

    For 'en_correct_zh_cn_incorrect': correct when preference == 1
    For 'en_incorrect_zh_cn_correct': correct when preference == 2
    """
    if "en_correct_zh_cn_incorrect" in lang_pair:
        return 1
    elif "en_incorrect_zh_cn_correct" in lang_pair:
        return 2
    else:
        raise ValueError(f"Cannot determine correct preference from lang_pair: {lang_pair}")


def collect_metric(subject_filter_name: str, subject_filter: Callable[[str], bool]):
    """
    Collect all metrics for samples matching the subject filter.
    Output to result/metrics/[model_name]_[subject_filter_name].json

    Args:
        subject_filter_name: Name of the subject filter (e.g., "stem", "humanities")
        subject_filter: Function that takes a subject name and returns True to keep the sample
    """
    result_dir = Path("result")
    metrics_dir = result_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    model_dirs = find_model_dirs(result_dir)

    if len(model_dirs) == 0:
        print("No model directories found.")
        return

    for model_dir in model_dirs:
        model_name = model_dir.name
        print(f"\nProcessing model: {model_name}")

        result_files = find_result_files(model_dir)

        metrics = {
            "model": model_name,
            "subject_filter": subject_filter_name,
            "accuracy": {},
            "correlation": {},
            "bias_binary": {},
            "bias_continuous": {}
        }

        # Calculate accuracy for each result type and lang_pair
        for result_type, files in result_files.items():
            metrics["accuracy"][result_type] = {}
            for lang_pair, file_path in files.items():
                try:
                    samples = load_jsonl_as_list(file_path, subject_filter)
                    correct_preference = get_correct_preference(lang_pair)
                    accuracy, correct, total = calculate_accuracy(samples, correct_preference)
                    metrics["accuracy"][result_type][lang_pair] = {
                        "accuracy": accuracy,
                        "correct": correct,
                        "total": total
                    }
                    print(f"  Accuracy ({result_type}/{lang_pair}): {accuracy:.4f} ({correct}/{total})")
                except Exception as e:
                    print(f"  Error calculating accuracy for {result_type}/{lang_pair}: {e}")

        # Calculate correlation between perplexity and preference_direct for each lang_pair
        perplexity_files = result_files['perplexities_local']
        preference_direct_files = result_files['preferences_local_direct']

        for lang_pair in perplexity_files.keys():
            if lang_pair in preference_direct_files:
                perplexity_file = perplexity_files[lang_pair]
                preference_file = preference_direct_files[lang_pair]

                # Load and filter data
                perplexity_data = load_jsonl(perplexity_file, subject_filter)
                preference_data = load_jsonl(preference_file, subject_filter)

                try:
                    correlation, matches, total = calculate_pearson_correlation(
                        perplexity_data, preference_data
                    )
                    metrics["correlation"][lang_pair] = {
                        "pearson_correlation": correlation,
                        "matches": matches,
                        "total": total
                    }
                    print(f"  Correlation ({lang_pair}): {correlation:.4f} ({matches}/{total} matches)")
                except Exception as e:
                    print(f"  Error calculating correlation for {lang_pair}: {e}")

                # Calculate bias_binary
                try:
                    bias, differences, total = calculate_bias_binary(
                        perplexity_data, preference_data
                    )
                    metrics["bias_binary"][lang_pair] = {
                        "bias": bias,
                        "differences": differences,
                        "total": total
                    }
                    print(f"  Bias Binary ({lang_pair}): {bias:.4f} ({differences}/{total} differences)")
                except Exception as e:
                    print(f"  Error calculating bias_binary for {lang_pair}: {e}")

        # Calculate bias_continuous between two preference files (need log_prob fields)
        preference_thinking_files = result_files['preferences_local_thinking']

        for lang_pair in preference_direct_files.keys():
            if lang_pair in preference_thinking_files:
                file1 = preference_direct_files[lang_pair]
                file2 = preference_thinking_files[lang_pair]

                # Load and filter data
                data1 = load_jsonl(file1, subject_filter)
                data2 = load_jsonl(file2, subject_filter)

                try:
                    avg_bias, sum_abs_diff, total = calculate_bias_continuous(
                        data1, data2
                    )
                    metrics["bias_continuous"][lang_pair] = {
                        "avg_bias": avg_bias,
                        "sum_abs_diff": sum_abs_diff,
                        "total": total
                    }
                    print(f"  Bias Continuous ({lang_pair}): {avg_bias:.6f} (n={total})")
                except Exception as e:
                    print(f"  Error calculating bias_continuous for {lang_pair}: {e}")

        # Write metrics to JSON file
        output_file = metrics_dir / f"{model_name}_{subject_filter_name}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)

        print(f"  -> Metrics written to {output_file}")


stem_subjects = {
    # Mathematics
        'abstract_algebra', 'college_mathematics', 'elementary_mathematics',
        'high_school_mathematics', 'high_school_statistics', 'formal_logic'
    # Physics & Chemistry
        'astronomy', 'college_physics', 'high_school_physics',
        'college_chemistry', 'high_school_chemistry', 'conceptual_physics',
    # Biology & Medicine
        'anatomy', 'college_biology', 'high_school_biology',
        'clinical_knowledge', 'college_medicine', 'professional_medicine',
        'medical_genetics', 'nutrition', 'virology', 'human_aging', 'human_sexuality',
    # Computer Science
        'college_computer_science', 'high_school_computer_science',
        'computer_security', 'machine_learning',
    # Engineering
        'electrical_engineering'
    }

# Example subject filters
def is_stem(subject: str) -> bool:
    """Filter for STEM subjects."""
    
    return subject.lower() in stem_subjects

def is_non_stem(subject: str) -> bool:
    """Filter for non-STEM subjects."""
    return subject.lower() not in stem_subjects


def is_all(subject: str) -> bool:
    """Filter that keeps all subjects."""
    return True


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python collect_metric.py <subject_filter_name>")
        print("Example: python collect_metric.py stem")
        print("Available filters: stem, humanities, all")
        sys.exit(1)

    filter_name = sys.argv[1]

    # Map filter names to filter functions
    filters = {
        "stem": is_stem,
        "non_stem": is_non_stem,
        "all": is_all,
    }

    if filter_name not in filters:
        print(f"Unknown filter: {filter_name}")
        print(f"Available filters: {', '.join(filters.keys())}")
        sys.exit(1)

    collect_metric(filter_name, filters[filter_name])
