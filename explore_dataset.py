from datasets import load_dataset


def explore_dataset(lang="ar_xy"):
    """Load and explore the dataset structure"""
    print(f"Loading dataset with language: {lang}...")
    ds = load_dataset("willchow66/mmmlu-intersection-filtered", lang)

    print("\nDataset structure:")
    print(f"Keys: {ds.keys()}")

    # Check the first split
    for split_name in ds.keys():
        print(f"\n{split_name} split:")
        print(f"Number of samples: {len(ds[split_name])}")
        print(f"Features: {ds[split_name].features}")
        print(f"\nFirst 5 samples:")
        for i in range(min(5, len(ds[split_name]))):
            print(f"\n--- Sample {i} ---")
            sample = ds[split_name][i]
            for key, value in sample.items():
                print(f"{key}: {value}")
        break  # Only show first split for now

    return ds

def print_all_subjects(lang="en"):
    ds = load_dataset("willchow66/mmmlu-intersection-filtered", lang)
    subjects = set()
    for split_name in ds.keys():
        for sample in ds[split_name]:
            subjects.add(sample['subject'])
    print(f"All subjects in {lang} dataset:")
    for subject in sorted(subjects):
        print(subject)


if __name__ == "__main__":
    # Explore datasets
    print("="*60)
    print("STEP 1: Exploring Datasets")
    print("="*60)
    print("\nEnglish (en) dataset (used for questions):")
    # explore_dataset("en")

    # explore_dataset("zh_cn")

    print_all_subjects()