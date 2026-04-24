import argparse
import csv
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".ppm", ".bmp"}


def to_path(path_like):
    return Path(path_like).resolve()


def ensure_exists(path):
    if not path.exists():
        raise FileNotFoundError(f"Required GTSRB path not found: {path}")


def class_dir_name(class_id):
    return f"{int(class_id):05d}"


def iter_images(root):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def find_case_insensitive(root, name):
    wanted = name.lower()
    for path in root.iterdir():
        if path.name.lower() == wanted:
            return path
    return None


def find_first_existing(root, candidates):
    for candidate in candidates:
        path = root / candidate
        if path.exists():
            return path
    for candidate in candidates:
        found = find_case_insensitive(root, candidate)
        if found is not None:
            return found
    return None


def read_test_labels(csv_path):
    ensure_exists(csv_path)
    labels = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        if "Path" not in reader.fieldnames and "Filename" not in reader.fieldnames:
            f.seek(0)
            reader = csv.DictReader(f)

        for row in reader:
            class_value = row.get("ClassId") or row.get("ClassID") or row.get("class_id")
            image_value = row.get("Path") or row.get("Filename") or row.get("file")
            if class_value is None or image_value is None:
                continue
            labels[Path(image_value).name] = int(class_value)
    return labels


def copy_image(src, split_root, class_id):
    dst_dir = split_root / class_dir_name(class_id)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if not dst.exists():
        shutil.copy2(src, dst)


def prepare_kaggle_layout(raw_root, output_root):
    train_root = find_first_existing(raw_root, ["Train", "train"])
    test_root = find_first_existing(raw_root, ["Test", "test"])
    test_csv = find_first_existing(raw_root, ["Test.csv", "test.csv"])
    if train_root is None or test_root is None or test_csv is None:
        return False

    output_train = output_root / "train"
    output_test = output_root / "test"

    for class_root in sorted(train_root.iterdir()):
        if not class_root.is_dir():
            continue
        class_id = int(class_root.name)
        for image_path in iter_images(class_root):
            copy_image(image_path, output_train, class_id)

    labels = read_test_labels(test_csv)
    for image_path in iter_images(test_root):
        class_id = labels.get(image_path.name)
        if class_id is not None:
            copy_image(image_path, output_test, class_id)

    return True


def prepare_official_layout(raw_root, output_root):
    train_root = raw_root / "GTSRB" / "Final_Training" / "Images"
    test_root = raw_root / "GTSRB" / "Final_Test" / "Images"
    test_csv = raw_root / "GT-final_test.csv"
    if not train_root.exists():
        train_root = raw_root / "Final_Training" / "Images"
    if not test_root.exists():
        test_root = raw_root / "Final_Test" / "Images"
    if not test_csv.exists():
        test_csv = raw_root / "Final_Test" / "Images" / "GT-final_test.csv"
    if not train_root.exists() or not test_root.exists() or not test_csv.exists():
        return False

    output_train = output_root / "train"
    output_test = output_root / "test"

    for class_root in sorted(train_root.iterdir()):
        if not class_root.is_dir():
            continue
        class_id = int(class_root.name)
        for image_path in iter_images(class_root):
            copy_image(image_path, output_train, class_id)

    labels = read_test_labels(test_csv)
    for image_path in iter_images(test_root):
        class_id = labels.get(image_path.name)
        if class_id is not None:
            copy_image(image_path, output_test, class_id)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Prepare GTSRB into ImageFolder-style train/test layout."
    )
    parser.add_argument(
        "--raw-root",
        required=True,
        help="Path to an extracted GTSRB folder, such as a Kaggle dataset folder.",
    )
    parser.add_argument(
        "--output-root",
        default="datasets/gtsrb",
        help="Output folder for processed GTSRB dataset.",
    )
    args = parser.parse_args()

    raw_root = to_path(args.raw_root)
    output_root = to_path(args.output_root)
    ensure_exists(raw_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if not prepare_kaggle_layout(raw_root, output_root):
        if not prepare_official_layout(raw_root, output_root):
            raise RuntimeError(
                "Could not recognize GTSRB layout. Expected Kaggle-style "
                "Train/Test/Test.csv or official Final_Training/Final_Test layout."
            )

    train_count = sum(1 for _ in iter_images(output_root / "train"))
    test_count = sum(1 for _ in iter_images(output_root / "test"))
    print(f"Prepared GTSRB dataset: train={train_count}, test={test_count}")


if __name__ == "__main__":
    main()
