import json
import os
from pathlib import Path

def calculate_accuracy(result_file):
    """
    Calculate accuracy based on preference and filename.

    For files with 'en_correct_zh_cn_incorrect': correct when preference == 1
    For files with 'en_incorrect_zh_cn_correct': correct when preference == 2
    """
    filename = os.path.basename(result_file)

    # Determine which preference is correct based on filename
    if "en_correct_zh_cn_incorrect" in filename:
        correct_preference = 1
    elif "en_incorrect_zh_cn_correct" in filename:
        correct_preference = 2
    else:
        raise ValueError(f"Cannot determine correct preference from filename: {filename}")

    # Read the JSONL file and count correct predictions
    total = 0
    correct = 0

    with open(result_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                data = json.loads(line)
                total += 1
                if data['preference'] == correct_preference:
                    correct += 1

    # Calculate accuracy
    if total == 0:
        accuracy = 0.0
    else:
        accuracy = correct / total

    return accuracy

def main():
    result_dir = Path("result")
    accuracy_dir = result_dir / "accuracy"

    # Create accuracy directory if it doesn't exist
    accuracy_dir.mkdir(exist_ok=True)

    # Process each .jsonl file in the result directory
    for result_file in result_dir.glob("*.jsonl"):
        # Skip if it's already an accuracy file
        if "_accuracy" in result_file.name:
            continue

        # Calculate accuracy
        accuracy = calculate_accuracy(result_file)

        # Create output filename in the accuracy folder
        output_file = accuracy_dir / (result_file.stem + "_accuracy.json")

        # Write accuracy to JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({"accuracy": accuracy}, f)

        print(f"Processed {result_file.name}: accuracy = {accuracy:.4f}")
        print(f"  -> Written to {output_file.relative_to(result_dir)}")

if __name__ == "__main__":
    main()
