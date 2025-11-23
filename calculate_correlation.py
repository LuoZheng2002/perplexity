def calculate_pearson_correlation(data1, data2):
    """
    Compare preferences sample by sample between two data dicts.
    Returns the Pearson correlation coefficient, number of matches, and total samples.

    Args:
        data1: Dict mapping index to sample dict (each with 'preference' field)
        data2: Dict mapping index to sample dict (each with 'preference' field)

    Returns:
        Tuple of (correlation, matches, total)
    """
    from scipy.stats import pearsonr

    # Find common indices
    common_indices = set(data1.keys()) & set(data2.keys())

    # Extract preference values for common indices
    prefs1 = []
    prefs2 = []
    matches = 0

    for idx in sorted(common_indices):
        pref1 = data1[idx]['preference']
        pref2 = data2[idx]['preference']

        prefs1.append(pref1)
        prefs2.append(pref2)

        if pref1 == pref2:
            matches += 1

    total = len(common_indices)

    # Calculate Pearson correlation
    if total < 2:
        return 0.0, matches, total

    # Check for variance - correlation requires variance in both variables
    if len(set(prefs1)) == 1 or len(set(prefs2)) == 1:
        return 0.0, matches, total

    correlation, _ = pearsonr(prefs1, prefs2)

    return correlation, matches, total
