#!/usr/bin/env python3
import os
import shutil
from pathlib import Path

SPLITS = ["train", "valid", "test"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def poly_to_bbox_line(line: str):
    """
    Converts one YOLO-seg polygon label line:
      cls x1 y1 x2 y2 x3 y3 ...
    into YOLO bbox:
      cls xc yc w h

    All coords are normalized [0,1].
    Returns None if the line is malformed.
    """
    line = line.strip()
    if not line:
        return None

    parts = line.split()
    if len(parts) < 7:  # cls + at least 3 points (6 nums)
        return None

    cls = parts[0]
    nums = parts[1:]

    if len(nums) % 2 != 0:
        return None

    try:
        xs = [float(nums[i]) for i in range(0, len(nums), 2)]
        ys = [float(nums[i]) for i in range(1, len(nums), 2)]
    except ValueError:
        return None

    x_min = max(0.0, min(1.0, min(xs)))
    x_max = max(0.0, min(1.0, max(xs)))
    y_min = max(0.0, min(1.0, min(ys)))
    y_max = max(0.0, min(1.0, max(ys)))

    w = x_max - x_min
    h = y_max - y_min
    if w <= 0.0 or h <= 0.0:
        return None

    xc = x_min + w / 2.0
    yc = y_min + h / 2.0

    return f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"

def copy_images(src_images: Path, dst_images: Path):
    dst_images.mkdir(parents=True, exist_ok=True)
    for p in src_images.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMG_EXTS:
            continue
        rel = p.relative_to(src_images)
        out = dst_images / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)

def convert_labels(src_labels: Path, dst_labels: Path):
    dst_labels.mkdir(parents=True, exist_ok=True)

    label_files = sorted(src_labels.glob("*.txt"))
    total_instances_in = 0
    total_instances_out = 0
    bad_lines = 0
    missing = 0

    for lf in label_files:
        out_lines = []
        with lf.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                total_instances_in += 1
                bb = poly_to_bbox_line(s)
                if bb is None:
                    bad_lines += 1
                    continue
                out_lines.append(bb)
                total_instances_out += 1

        out_path = dst_labels / lf.name
        with out_path.open("w", encoding="utf-8") as w:
            w.write("\n".join(out_lines) + ("\n" if out_lines else ""))

    # If there are images without labels YOLO can handle it, so we don't error.
    return total_instances_in, total_instances_out, bad_lines, missing

def copy_root_files(seg_root: Path, out_root: Path):
    # Copy common files if present
    for name in ["data.yaml", "dataset.yaml", "README.dataset.txt", "README.roboflow.txt"]:
        src = seg_root / name
        if src.exists() and src.is_file():
            shutil.copy2(src, out_root / name)

def main():
    # Run this from URS_PROJECT/ (recommended)
    seg_root = Path("./datasets/Microplastics-V3-ValidSplit").resolve()
    out_root = Path("./datasets/Microplastics-Bounding-Box").resolve()

    print(f"SEG ROOT: {seg_root}")
    print(f"OUT ROOT: {out_root}")

    if not seg_root.exists():
        raise SystemExit(f"❌ Seg dataset not found: {seg_root}")

    if out_root.exists():
        raise SystemExit(
            f"❌ Output folder already exists: {out_root}\n"
            f"Rename/delete it first so we don't overwrite anything."
        )

    out_root.mkdir(parents=True, exist_ok=False)
    copy_root_files(seg_root, out_root)

    grand_in = grand_out = grand_bad = 0

    for split in SPLITS:
        src_images = seg_root / split / "images"
        src_labels = seg_root / split / "labels"

        if not src_images.is_dir():
            raise SystemExit(f"❌ Missing folder: {src_images}")
        if not src_labels.is_dir():
            raise SystemExit(f"❌ Missing folder: {src_labels}")

        dst_images = out_root / split / "images"
        dst_labels = out_root / split / "labels"

        print(f"\n--- Converting split: {split} ---")
        copy_images(src_images, dst_images)

        total_in, total_out, bad, _ = convert_labels(src_labels, dst_labels)
        print(f"{split}: instances_in={total_in}, instances_out={total_out}, bad_lines_skipped={bad}")

        grand_in += total_in
        grand_out += total_out
        grand_bad += bad

    print("\n✅ DONE")
    print(f"TOTAL instances_in={grand_in}, instances_out={grand_out}, bad_lines_skipped={grand_bad}")
    print(f"Output dataset created at:\n  {out_root}")

if __name__ == "__main__":
    main()
