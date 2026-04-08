import os
import json
import argparse
from datasets import load_dataset, get_dataset_config_names
from PIL import Image

DATASET_NAME = "OpenDataArena/MMFineReason-Full-2.3M-Qwen3-VL-235B-Thinking"
SAMPLES_PER_SUBSET = 10


def get_script_root():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(script_dir, "../../"))


def save_image(image, path):
    if image is None:
        return False
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(path, format="JPEG")
    return True


def download_subset(subset_name, image_dir, data_dir, n=SAMPLES_PER_SUBSET):
    print(f"\n[{subset_name}] Attempting to load {n} samples...")
    subset_image_dir = os.path.join(image_dir, subset_name)
    os.makedirs(subset_image_dir, exist_ok=True)

    try:
        ds = load_dataset(DATASET_NAME, name=subset_name, split="train", streaming=True)
        samples_raw = list(ds.take(n))
    except Exception as e:
        print(f"  [SKIP] Failed to load subset '{subset_name}': {e}")
        return []

    if not samples_raw:
        print(f"  [SKIP] No samples found in subset '{subset_name}'.")
        return []

    saved_samples = []
    for idx, sample in enumerate(samples_raw):
        image = sample.get("image") or sample.get("images")
        if isinstance(image, list):
            image = image[0] if image else None

        image_filename = f"{subset_name}_{idx:04d}.jpg"
        image_path = os.path.join(subset_image_dir, image_filename)

        image_saved = False
        if image is not None:
            image_saved = save_image(image, image_path)

        # Collect all non-image fields for metadata
        meta = {}
        for k, v in sample.items():
            if k in ("image", "images"):
                continue
            if isinstance(v, (str, int, float, bool, list, dict)) or v is None:
                meta[k] = v
            else:
                meta[k] = str(v)

        meta["_image_path"] = image_path if image_saved else None
        meta["_subset"] = subset_name
        meta["_index"] = idx
        saved_samples.append(meta)

    metadata_path = os.path.join(data_dir, f"{subset_name}_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(saved_samples, f, ensure_ascii=False, indent=2)

    print(f"  [OK] Saved {len(saved_samples)} samples → {metadata_path}")
    return saved_samples


def main():
    parser = argparse.ArgumentParser(
        description="MMFineReason Subset Downloader — downloads 10 samples per subset"
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Root output directory (default: mmfinereason dataset root)",
    )
    parser.add_argument(
        "--subsets",
        nargs="*",
        default=None,
        help="Specific subset names to download (default: all available)",
    )
    args = parser.parse_args()

    dataset_root = args.output_root if args.output_root else get_script_root()
    image_dir = os.path.join(dataset_root, "mmfinereason_images")
    data_dir = os.path.join(dataset_root, "data", "metadata")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # Discover subsets
    if args.subsets:
        subsets = args.subsets
        print(f"Using user-specified subsets: {subsets}")
    else:
        print(f"Fetching available configs for '{DATASET_NAME}'...")
        try:
            subsets = get_dataset_config_names(DATASET_NAME)
            print(f"Found {len(subsets)} subsets: {subsets}")
        except Exception as e:
            print(f"Could not retrieve config names: {e}")
            print("Falling back to default split (no subset).")
            subsets = [None]

    summary = {}
    for subset in subsets:
        if subset is None:
            # Dataset without named configs
            result = download_subset("default", image_dir, data_dir)
        else:
            result = download_subset(subset, image_dir, data_dir)
        summary[subset or "default"] = len(result)

    # Save overall summary
    summary_path = os.path.join(dataset_root, "data", "metadata", "download_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("Download complete. Summary:")
    for name, count in summary.items():
        status = "OK" if count > 0 else "SKIPPED"
        print(f"  [{status}] {name}: {count} samples")
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
