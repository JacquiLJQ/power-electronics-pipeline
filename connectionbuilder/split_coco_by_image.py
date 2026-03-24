import json
import os
from pathlib import Path
from collections import defaultdict


def safe_stem(filename: str) -> str:
    """
    Turn 'xxx.jpg' into 'xxx'. Keeps everything else.
    Also removes problematic path separators just in case.
    """
    name = Path(filename).name
    return Path(name).stem


def split_coco(coco_path: str, out_dir: str):
    coco_path = Path(coco_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with coco_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    # COCO fields (roboflow usually has these)
    info = coco.get("info", {})
    licenses = coco.get("licenses", [])
    categories = coco.get("categories", [])
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])  # assumed present

    # Group annotations by image_id
    ann_by_img = defaultdict(list)
    for ann in annotations:
        ann_by_img[ann["image_id"]].append(ann)

    # Build per-image coco json
    created = 0
    missing = 0

    for img in images:
        img_id = img["id"]
        file_name = img.get("file_name", f"image_{img_id}")
        stem = safe_stem(file_name)

        sub = {
            "info": info,
            "licenses": licenses,
            "categories": categories,
            "images": [img],
            "annotations": ann_by_img.get(img_id, []),
        }

        # If you want to skip images with no annotations, uncomment:
        # if len(sub["annotations"]) == 0:
        #     continue

        out_path = out_dir / f"{stem}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(sub, f, ensure_ascii=False, indent=2)

        created += 1
        if len(sub["annotations"]) == 0:
            missing += 1

    print(f"Done. Wrote {created} files to: {out_dir}")
    if missing:
        print(f"Note: {missing} images had 0 annotations (still exported).")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coco", required=True, help="Path to COCO JSON exported from Roboflow"
    )
    parser.add_argument(
        "--out", required=True, help="Output folder for per-image JSONs"
    )
    args = parser.parse_args()

    split_coco(args.coco, args.out)
