#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path

import cv2


def parse_yolo_line(line: str):
    # supports: "cls xc yc w h" (with extra spaces)
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cls = int(float(parts[0]))
    xc = float(parts[1])
    yc = float(parts[2])
    w = float(parts[3])
    h = float(parts[4])
    return cls, xc, yc, w, h


def yolo_to_xyxy(xc, yc, w, h, img_w, img_h):
    # YOLO normalized center format -> pixel xyxy
    x1 = (xc - w / 2.0) * img_w
    y1 = (yc - h / 2.0) * img_h
    x2 = (xc + w / 2.0) * img_w
    y2 = (yc + h / 2.0) * img_h
    return x1, y1, x2, y2


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True, help="path to circuit.png")
    ap.add_argument("--labels", required=True, help="path to labels.txt (YOLO format)")
    ap.add_argument(
        "--out", default="", help="output image path (default: <img>_bbox.png)"
    )
    ap.add_argument("--thickness", type=int, default=2, help="bbox line thickness")
    ap.add_argument("--font_scale", type=float, default=0.6, help="label font scale")
    ap.add_argument("--no_text", action="store_true", help="do not draw class id text")
    args = ap.parse_args()

    img_path = Path(args.img)
    lab_path = Path(args.labels)

    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")

    H, W = img.shape[:2]

    if not lab_path.exists():
        raise RuntimeError(f"labels file not found: {lab_path}")

    with open(lab_path, "r", encoding="utf-8") as f:
        lines = [ln for ln in (x.strip() for x in f.readlines()) if ln]

    drawn = 0
    for ln in lines:
        parsed = parse_yolo_line(ln)
        if parsed is None:
            continue
        cls, xc, yc, w, h = parsed
        x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, w, h, W, H)

        # clamp to image bounds
        x1 = int(round(clamp(x1, 0, W - 1)))
        y1 = int(round(clamp(y1, 0, H - 1)))
        x2 = int(round(clamp(x2, 0, W - 1)))
        y2 = int(round(clamp(y2, 0, H - 1)))

        # ignore degenerate boxes
        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), args.thickness)

        if not args.no_text:
            text = str(cls)
            # draw a small filled background for readability
            (tw, th), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, args.font_scale, 1
            )
            tx, ty = x1, max(0, y1 - 4)
            bx1, by1 = tx, max(0, ty - th - 4)
            bx2, by2 = tx + tw + 6, ty + 2
            cv2.rectangle(img, (bx1, by1), (bx2, by2), (0, 255, 0), -1)
            cv2.putText(
                img,
                text,
                (tx + 3, ty - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                args.font_scale,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        drawn += 1

    out_path = (
        Path(args.out) if args.out else img_path.with_name(img_path.stem + "_bbox.png")
    )
    cv2.imwrite(str(out_path), img)
    print(f"[OK] drew {drawn} boxes -> {out_path}")


if __name__ == "__main__":
    main()
