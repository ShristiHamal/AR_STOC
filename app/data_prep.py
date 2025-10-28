# data_prep.py
import os
import csv
import json
from pathlib import Path
from typing import Optional

def validate_openpose_json(json_path: Path) -> bool:
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        # Check structure
        if "people" not in data or len(data["people"]) == 0:
            return False
        first = data["people"][0]
        return "pose_keypoints_2d" in first and len(first["pose_keypoints_2d"]) > 0
    except Exception:
        return False

def make_csv(root_dir: Path, split: str) -> Path:
    split_dir = Path(root_dir) / split
    pair_file = split_dir / f"{split}_pairs.txt"
    required_dirs = {
        "image": split_dir / "image",
        "cloth": split_dir / "cloth",
        "mask": split_dir / "cloth-mask",
        "pose": split_dir / "openpose_json"
    }

    for name, path in required_dirs.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name} directory: {path}")

    with pair_file.open('r') as f:
        pairs = [tuple(line.strip().split()[:2]) for line in f if line.strip()]

    csv_out = split_dir / f"{split}_pairs.csv"
    with csv_out.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['person_image', 'cloth_image', 'mask_image', 'pose_json'])

        for person, cloth in pairs:
            person_path = required_dirs["image"] / person
            cloth_path = required_dirs["cloth"] / cloth
            mask_path = required_dirs["mask"] / cloth  # assumes same filename for mask
            pose_path = required_dirs["pose"] / f"{Path(person).stem}_keypoints.json"

            if not (person_path.exists() and cloth_path.exists() and mask_path.exists() and pose_path.exists()):
                # skip bad pairs
                continue
            if not validate_openpose_json(pose_path):
                continue

            writer.writerow([str(person_path), str(cloth_path), str(mask_path), str(pose_path)])

    print(f"CSV written: {csv_out}")
    return csv_out
