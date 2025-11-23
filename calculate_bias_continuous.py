import math


def calculate_bias_continuous(data1, data2):
    """
    Compare probability differences sample by sample between two data dicts.

    Steps:
    1. For each sample in each dict: signed_diff = exp(log_prob_1) - exp(log_prob_2)
    2. For each common sample: abs_diff = abs(signed_diff_1 - signed_diff_2)
    3. Average all abs_diff values

    Args:
        data1: Dict mapping index to sample dict (each with 'log_prob_1' and 'log_prob_2' fields)
        data2: Dict mapping index to sample dict (each with 'log_prob_1' and 'log_prob_2' fields)

    Returns:
        Tuple of (avg_bias, sum_abs_diff, total)
    """
    # Find common indices
    common_indices = set(data1.keys()) & set(data2.keys())

    # Calculate absolute differences
    abs_diffs = []

    for idx in sorted(common_indices):
        # Calculate signed difference for data1
        log_prob_1_data1 = data1[idx]['log_prob_1']
        log_prob_2_data1 = data1[idx]['log_prob_2']
        signed_diff_1 = math.exp(log_prob_1_data1) - math.exp(log_prob_2_data1)

        # Calculate signed difference for data2
        log_prob_1_data2 = data2[idx]['log_prob_1']
        log_prob_2_data2 = data2[idx]['log_prob_2']
        signed_diff_2 = math.exp(log_prob_1_data2) - math.exp(log_prob_2_data2)

        # Calculate absolute difference between the two signed differences
        abs_diff = abs(signed_diff_1 - signed_diff_2)
        abs_diffs.append(abs_diff)

    total = len(common_indices)

    # Calculate average bias
    if total == 0:
        return 0.0, 0.0, 0

    sum_abs_diff = sum(abs_diffs)
    avg_bias = sum_abs_diff / total

    return avg_bias, sum_abs_diff, total
