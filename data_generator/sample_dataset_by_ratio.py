import argparse
import random
import shutil
from pathlib import Path


DEFAULT_SOURCE_ROOT = Path(r"F:\Research__dir\dl_lidar\datasets\Dataset_Wave_Lidar_10000samples_non_uniform")
DEFAULT_OUTPUT_ROOT = Path(r"F:\Research__dir\dl_lidar\datasets")
SPLITS = ("train", "val", "test")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Randomly copy train/val/test subsets from an existing dataset root using one or more ratios.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT, help="Source dataset root.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Parent directory used to store the sampled dataset folders.",
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        required=True,
        help="One or more sampling ratios, for example 0.6 0.8",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--dataset-name-prefix",
        type=str,
        default="Dataset_Wave_Lidar_v3_10000samples_subset",
        help="Prefix used when naming output dataset folders.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow copying into an existing output folder. Existing files are kept and may be overwritten.",
    )
    return parser.parse_args()


def validate_ratios(ratios):
    for ratio in ratios:
        if not (0.0 < ratio <= 1.0):
            raise ValueError(f"Each ratio must be in (0, 1], got {ratio}")


def format_ratio_tag(ratio):
    percent = int(round(ratio * 100))
    return f"ratio{percent:02d}"


def collect_split_files(source_root, split):
    split_dir = source_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    files = sorted(split_dir.glob("*.mat"))
    if not files:
        raise FileNotFoundError(f"No .mat files found in split directory: {split_dir}")
    return files


def ensure_output_dir(path, overwrite):
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory already exists and is not empty: {path}\n"
            "Use --overwrite to allow writing into it."
        )
    path.mkdir(parents=True, exist_ok=True)


def copy_subset_for_ratio(source_root, output_root, ratio, seed, dataset_name_prefix, overwrite):
    ratio_tag = format_ratio_tag(ratio)
    dataset_dir = output_root / f"{dataset_name_prefix}_{ratio_tag}"

    print(f"\n=== Sampling ratio {ratio:.2f} -> {dataset_dir} ===")

    rng = random.Random(seed + int(round(ratio * 1000)))

    for split in SPLITS:
        files = collect_split_files(source_root, split)
        split_output_dir = dataset_dir / split
        ensure_output_dir(split_output_dir, overwrite)

        sample_count = int(round(len(files) * ratio))
        sample_count = max(1, min(len(files), sample_count))
        selected = sorted(rng.sample(files, sample_count))

        for src in selected:
            dst = split_output_dir / src.name
            shutil.copy2(src, dst)

        print(f"[{split}] copied {sample_count} / {len(files)} files")


def main():
    args = parse_args()
    validate_ratios(args.ratios)

    if not args.source_root.exists():
        raise FileNotFoundError(f"Source dataset root not found: {args.source_root}")

    for ratio in args.ratios:
        copy_subset_for_ratio(
            source_root=args.source_root,
            output_root=args.output_root,
            ratio=ratio,
            seed=args.seed,
            dataset_name_prefix=args.dataset_name_prefix,
            overwrite=args.overwrite,
        )

    print("\nAll requested subset datasets have been generated.")


if __name__ == "__main__":
    main()
