import json
import os
from pathlib import Path

def load_jsonl(file_path):
    """Load JSONL file and return a dictionary indexed by 'index' field."""
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                data[item['index']] = item
    return data

def calculate_correlation(perplexity_file, preference_file):
    """
    Compare preferences sample by sample between perplexity and preference files.
    Returns the correlation (ratio of matching preferences).
    """
    # Load both files
    perplexity_data = load_jsonl(perplexity_file)
    preference_data = load_jsonl(preference_file)

    # Get common indices
    common_indices = set(perplexity_data.keys()) & set(preference_data.keys())

    if len(common_indices) == 0:
        return 0.0, 0, 0

    # Count matches
    matches = 0
    total = len(common_indices)

    for idx in common_indices:
        if perplexity_data[idx]['preference'] == preference_data[idx]['preference']:
            matches += 1

    correlation = matches / total if total > 0 else 0.0

    return correlation, matches, total

def find_matching_pairs(result_dir):
    """
    Find matching pairs of perplexity and preference files.
    Returns a list of tuples (perplexity_file, preference_file, prefix).
    """
    pairs = []

    # Find all perplexity files
    perplexity_files = list(result_dir.glob("*_perplexities_local.jsonl"))

    for perplexity_file in perplexity_files:
        # Extract prefix by removing the suffix
        prefix = perplexity_file.stem.replace("_perplexities_local", "")

        # Look for corresponding preference file
        preference_file = result_dir / f"{prefix}_preferences_local_direct.jsonl"

        if preference_file.exists():
            pairs.append((perplexity_file, preference_file, prefix))

    return pairs

def main():
    result_dir = Path("result")
    correlation_dir = result_dir / "correlation"

    # Create correlation directory if it doesn't exist
    correlation_dir.mkdir(exist_ok=True)

    # Find matching pairs
    pairs = find_matching_pairs(result_dir)

    if len(pairs) == 0:
        print("No matching pairs found.")
        return

    # Process each pair
    for perplexity_file, preference_file, prefix in pairs:
        # Calculate correlation
        correlation, matches, total = calculate_correlation(perplexity_file, preference_file)

        # Create output filename
        output_file = correlation_dir / f"{prefix}_correlation.json"

        # Write correlation to JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "correlation": correlation,
                "matches": matches,
                "total": total
            }, f)

        print(f"Processed pair: {prefix}")
        print(f"  Perplexity file: {perplexity_file.name}")
        print(f"  Preference file: {preference_file.name}")
        print(f"  Correlation: {correlation:.4f} ({matches}/{total} matches)")
        print(f"  -> Written to {output_file.relative_to(result_dir)}")
        print()

if __name__ == "__main__":
    main()
