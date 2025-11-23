def calculate_accuracy(samples, correct_preference):
    """
    Calculate accuracy based on preference values.

    Args:
        samples: List of sample dicts, each containing a 'preference' field
        correct_preference: The preference value (1 or 2) that is considered correct

    Returns:
        Tuple of (accuracy, correct_count, total_count)
    """
    total = len(samples)
    correct = sum(1 for sample in samples if sample['preference'] == correct_preference)

    if total == 0:
        return 0.0, 0, 0

    accuracy = correct / total
    return accuracy, correct, total
