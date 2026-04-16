import argparse
import shutil
import tarfile
from pathlib import Path


def to_path(path_like):
    return Path(path_like).resolve()


def parse_mapping(file_path, split_char=" "):
    mapping = {}
    with file_path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            key, value = raw.split(split_char, 1)
            mapping[int(key)] = value.strip()
    return mapping


def ensure_exists(path):
    if not path.exists():
        raise FileNotFoundError(f"Required CUB file not found: {path}")


def extract_archive(archive_path, extract_root):
    archive_path = to_path(archive_path)
    extract_root = to_path(extract_root)
    ensure_exists(archive_path)
    extract_root.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=extract_root)

    candidate = extract_root / "CUB_200_2011"
    ensure_exists(candidate)
    return candidate


def prepare_cub200_dataset(raw_root, output_root, copy_mode="copy"):
    raw_root = to_path(raw_root)
    output_root = to_path(output_root)

    images_root = raw_root / "images"
    images_txt = raw_root / "images.txt"
    split_txt = raw_root / "train_test_split.txt"

    for needed in (images_root, images_txt, split_txt):
        ensure_exists(needed)

    id_to_relpath = parse_mapping(images_txt)
    id_to_split = parse_mapping(split_txt)

    train_root = output_root / "train"
    test_root = output_root / "test"
    train_root.mkdir(parents=True, exist_ok=True)
    test_root.mkdir(parents=True, exist_ok=True)

    copied = 0
    for image_id, rel_path in id_to_relpath.items():
        if image_id not in id_to_split:
            raise RuntimeError(f"Missing split flag for image id {image_id}")

        src = images_root / rel_path
        ensure_exists(src)

        class_name = rel_path.split("/", 1)[0]
        split_flag = int(id_to_split[image_id])
        split_root = train_root if split_flag == 1 else test_root
        dst_dir = split_root / class_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name

        if dst.exists():
            continue

        if copy_mode != "copy":
            raise ValueError(f"Unsupported copy mode: {copy_mode}")
        shutil.copy2(src, dst)
        copied += 1

    return copied


def main():
    parser = argparse.ArgumentParser(
        description="Prepare CUB-200-2011 into ImageFolder-style train/test layout."
    )
    parser.add_argument(
        "--raw-root",
        default=None,
        help="Path to the extracted CUB_200_2011 directory.",
    )
    parser.add_argument(
        "--archive-path",
        default="datasets/CUB_200_2011.tgz",
        help="Path to CUB_200_2011.tgz archive.",
    )
    parser.add_argument(
        "--extract-root",
        default="datasets/_downloads/cub200/extracted",
        help="Temporary extraction directory for the raw archive.",
    )
    parser.add_argument(
        "--output-root",
        default="datasets/cub200",
        help="Output folder for processed CUB200 dataset.",
    )
    parser.add_argument("--copy-mode", choices=("copy",), default="copy")
    args = parser.parse_args()

    raw_root = args.raw_root
    if raw_root is None:
        raw_root = extract_archive(
            archive_path=args.archive_path,
            extract_root=args.extract_root,
        )

    copied = prepare_cub200_dataset(
        raw_root=raw_root,
        output_root=args.output_root,
        copy_mode=args.copy_mode,
    )
    print(f"Prepared CUB200 dataset. Copied files: {copied}")


if __name__ == "__main__":
    main()
