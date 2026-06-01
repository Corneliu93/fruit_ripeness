import os
import argparse
from pathlib import Path

BASE_DIR = "dataset"
CLASS_MAP = {
    "apple":  ["unripe", "ripe", "overripe"],
    "banana": ["unripe", "ripe", "overripe"],
    "pear":   ["unripe", "ripe", "overripe"],
}
VALID_EXT = {".jpg", ".jpeg", ".png"}


def rename_folder(folder: str, fruit: str, stage: str) -> int:
    folder_path = Path(folder)
    if not folder_path.exists():
        print(f"  ⚠️  Nonexistent folder: {folder}")
        return 0

    files = sorted([
        f for f in folder_path.iterdir()
        if f.suffix.lower() in VALID_EXT and f.is_file()
    ])

    if not files:
        print(f"  ℹ️  {fruit}/{stage}: no images")
        return 0

    print(f"\n  📁 {fruit}/{stage}  ({len(files)} files)")

    # Step 1: → temp
    temp_files = []
    for i, f in enumerate(files):
        temp = folder_path / f"__tmp_{i:04d}{f.suffix.lower()}"
        f.rename(temp)
        temp_files.append(temp)

    # Step 2: temp → final
    for idx, temp_f in enumerate(sorted(temp_files), start=1):
        ext = ".jpg" if temp_f.suffix.lower() == ".jpeg" else temp_f.suffix.lower()
        final = folder_path / f"{fruit}_{stage}_{idx:03d}{ext}"
        temp_f.rename(final)
        print(f"    → {final.name}")

    return len(temp_files)


def main(target_folder=None):
    total = 0
    if target_folder:
        parts = Path(target_folder).parts
        stage, fruit = parts[-1], parts[-2]
        total += rename_folder(target_folder, fruit, stage)
    else:
        for fruit, stages in CLASS_MAP.items():
            for stage in stages:
                total += rename_folder(os.path.join(BASE_DIR, fruit, stage), fruit, stage)
    print(f"\n  ✅ Total renamed: {total} files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default=None,
        help="Optional: only one folder (e.g., dataset/apple/unripe)")
    args = parser.parse_args()
    main(args.folder)