import argparse
import random
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def ensure_split_dirs(dataset_dir: Path, split: str):
    (dataset_dir / split / "images").mkdir(parents=True, exist_ok=True)
    (dataset_dir / split / "labels").mkdir(parents=True, exist_ok=True)

def list_images(neg_dir: Path):
    imgs = sorted([p for p in neg_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])
    if not imgs:
        raise FileNotFoundError(f"No images found in negative crops folder: {neg_dir}")
    return imgs

def copy_img_and_make_empty_label(img: Path, out_img_dir: Path, out_lbl_dir: Path, prefix: str, overwrite: bool):
    new_img_name = f"{prefix}{img.name}"
    dst_img = out_img_dir / new_img_name
    stem = Path(new_img_name).stem
    dst_lbl = out_lbl_dir / f"{stem}.txt"

    if dst_img.exists() and not overwrite:
        raise FileExistsError(f"Destination image exists: {dst_img} (use --overwrite)")
    if dst_lbl.exists() and not overwrite:
        raise FileExistsError(f"Destination label exists: {dst_lbl} (use --overwrite)")

    shutil.copy2(img, dst_img)
    dst_lbl.write_text("")  # empty = no objects

def main():
    ap = argparse.ArgumentParser(description="Add negative crops to a Roboflow 3-way split YOLO dataset (empty labels in destination only).")
    ap.add_argument("--dataset_dir", required=True, help="Path to Microplastics-V3-ValidSplit folder")
    ap.add_argument("--neg_dir", required=True, help="Path to negative_crops folder containing ONLY images")
    ap.add_argument("--n_train", type=int, default=-1, help="How many negatives to put into train (default: all remaining after val/test)")
    ap.add_argument("--n_val", type=int, default=0, help="How many negatives to put into valid (default: 0)")
    ap.add_argument("--n_test", type=int, default=0, help="How many negatives to put into test (default: 0)")
    ap.add_argument("--seed", type=int, default=354, help="Random seed")
    ap.add_argument("--prefix", default="neg_", help="Prefix for negative filenames to avoid collisions")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite if files already exist")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    neg_dir = Path(args.neg_dir).resolve()

    if not dataset_dir.exists():
        raise FileNotFoundError(f"dataset_dir not found: {dataset_dir}")
    if not neg_dir.exists():
        raise FileNotFoundError(f"neg_dir not found: {neg_dir}")

    for split in ["train", "valid", "test"]:
        ensure_split_dirs(dataset_dir, split)

    imgs = list_images(neg_dir)
    random.seed(args.seed)
    random.shuffle(imgs)

    n_total = len(imgs)
    n_val = max(0, args.n_val)
    n_test = max(0, args.n_test)

    if n_val + n_test > n_total:
        raise ValueError(f"Requested n_val+n_test={n_val+n_test} but only have {n_total} negatives")

    remaining = imgs[n_val + n_test:]

    # If n_train=-1, use all remaining
    if args.n_train < 0:
        n_train = len(remaining)
    else:
        n_train = min(args.n_train, len(remaining))

    val_imgs = imgs[:n_val]
    test_imgs = imgs[n_val:n_val + n_test]
    train_imgs = remaining[:n_train]

    print(f"Negatives found: {n_total}")
    print(f"Placing negatives -> train: {len(train_imgs)}, valid: {len(val_imgs)}, test: {len(test_imgs)}")
    print(f"Dataset: {dataset_dir}")
    print(f"Neg folder: {neg_dir}")

    def place(split: str, img_list):
        out_img_dir = dataset_dir / split / "images"
        out_lbl_dir = dataset_dir / split / "labels"
        for img in img_list:
            copy_img_and_make_empty_label(img, out_img_dir, out_lbl_dir, args.prefix, args.overwrite)

    place("valid", val_imgs)
    place("test", test_imgs)
    place("train", train_imgs)

    print("✅ Done. Negative images copied and empty labels created in destination dataset only.")
    print(f"Note: negative filenames were prefixed with '{args.prefix}'.")

if __name__ == "__main__":
    main()
