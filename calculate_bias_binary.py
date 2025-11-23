def calculate_bias_binary(data1, data2):
    """
    Compare preferences sample by sample between two data dicts.
    Returns the bias (proportion of different preferences), number of differences, and total samples.
    Bias = number of samples with different preference / total samples

    Args:
        data1: Dict mapping index to sample dict (each with 'preference' field)
        data2: Dict mapping index to sample dict (each with 'preference' field)

    Returns:
        Tuple of (bias, differences, total)
    """
    # Find common indices
    common_indices = set(data1.keys()) & set(data2.keys())

    # Count differences
    differences = 0

    for idx in sorted(common_indices):
        pref1 = data1[idx]['preference']
        pref2 = data2[idx]['preference']

        if pref1 != pref2:
            differences += 1

    total = len(common_indices)

    # Calculate bias
    if total == 0:
        return 0.0, 0, 0

    bias = differences / total

    return bias, differences, total
