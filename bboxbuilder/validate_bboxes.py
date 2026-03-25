# validate_bboxes.py

from __future__ import annotations

import json
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np


# =========================
# Basic data structures
# =========================


@dataclass
class Det:
    cls_id: int
    conf: float
    x1: int
    y1: int
    x2: int
    y2: int

    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    def to_xyxy(self) -> List[int]:
        return [self.x1, self.y1, self.x2, self.y2]

    def to_xywh_abs(self) -> List[float]:
        cx = (self.x1 + self.x2) / 2.0
        cy = (self.y1 + self.y2) / 2.0
        w = self.x2 - self.x1
        h = self.y2 - self.y1
        return [cx, cy, w, h]

    def to_yolo_line(self, img_w: int, img_h: int) -> str:
        cx = (self.x1 + self.x2) / 2.0 / img_w
        cy = (self.y1 + self.y2) / 2.0 / img_h
        w = (self.x2 - self.x1) / img_w
        h = (self.y2 - self.y1) / img_h
        return f"{self.cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {self.conf:.6f}"


# =========================
# Helper functions
# =========================


def clamp_box(
    x1: int, y1: int, x2: int, y2: int, w: int, h: int
) -> Tuple[int, int, int, int]:
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    return x1, y1, x2, y2


def find_images(folder: Path, image_exts: List[str]) -> List[Path]:
    ext_set = {e.lower() for e in image_exts}
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in ext_set])


# def read_binary_image(image_path: Path) -> np.ndarray:
#     """
#     Read already-binarized image as grayscale.
#     Expected pixel values: 0 or 255.
#     """
#     img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
#     if img is None:
#         raise RuntimeError(f"Failed to read image: {image_path}")
#     return img
def read_binary_image(image_path: Path) -> np.ndarray:
    """
    Read already-binarized image as grayscale.
    Expected final shape: (H, W)
    Expected pixel values: 0 or 255
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = np.squeeze(img, axis=2)

    return img


def load_yolo_txt(txt_path: Path, img_w: int, img_h: int) -> List[Det]:
    dets: List[Det] = []

    if not txt_path.exists():
        return dets

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            cls_id = int(float(parts[0]))
            cx = float(parts[1]) * img_w
            cy = float(parts[2]) * img_h
            bw = float(parts[3]) * img_w
            bh = float(parts[4]) * img_h
            conf = float(parts[5]) if len(parts) >= 6 else 1.0

            x1 = int(round(cx - bw / 2.0))
            y1 = int(round(cy - bh / 2.0))
            x2 = int(round(cx + bw / 2.0))
            y2 = int(round(cy + bh / 2.0))
            x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, img_w, img_h)

            if x2 <= x1 or y2 <= y1:
                continue

            dets.append(
                Det(
                    cls_id=cls_id,
                    conf=conf,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )

    return dets


def intersection_area(a: Det, b: Det) -> int:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    return max(0, x2 - x1) * max(0, y2 - y1)


def overlap_ratio_min_area(a: Det, b: Det) -> float:
    """
    overlap ratio = intersection / min(area_a, area_b)
    Better than IoU for near-duplicate detections.
    """
    inter = intersection_area(a, b)
    if inter <= 0:
        return 0.0
    denom = max(1, min(a.area(), b.area()))
    return inter / denom


class DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int):
        pa, pb = self.find(a), self.find(b)
        if pa != pb:
            self.parent[pb] = pa


def extract_ring_mask(img_h: int, img_w: int, det: Det, expand_px: int) -> np.ndarray:
    ox1, oy1, ox2, oy2 = det.x1, det.y1, det.x2, det.y2
    ex1, ey1, ex2, ey2 = clamp_box(
        ox1 - expand_px,
        oy1 - expand_px,
        ox2 + expand_px,
        oy2 + expand_px,
        img_w,
        img_h,
    )

    outer = np.zeros((img_h, img_w), dtype=np.uint8)
    inner = np.zeros((img_h, img_w), dtype=np.uint8)

    outer[ey1:ey2, ex1:ex2] = 255
    inner[oy1:oy2, ox1:ox2] = 255

    ring = cv2.subtract(outer, inner)
    return ring


def has_wire_connection(
    neededdebug: bool,
    binary_img: np.ndarray,
    det: Det,
    expand_min: int = 1,
    expand_max: int = 5,
    min_black_pixels: int = 3,
) -> Tuple[bool, Dict]:
    """
    On binary image:
      black pixel = 0
      white pixel = 255

    In the outer ring around bbox, if enough black pixels exist,
    treat component as having wire connection.
    """

    img_h, img_w = binary_img.shape[:2]
    per_expand_info = []

    for px in range(expand_min, expand_max + 1):
        ring = extract_ring_mask(img_h, img_w, det, px)
        black_in_ring = int(np.sum((ring > 0) & (binary_img == 0)))

        per_expand_info.append(
            {
                "expand_px": px,
                "black_pixels_in_ring": black_in_ring,
            }
        )

        if black_in_ring >= min_black_pixels:
            return True, {
                "checked": True,
                "connected": True,
                "connected_at_expand_px": px,
                "checks": per_expand_info,
            }

    return False, {
        "checked": True,
        "connected": False,
        "connected_at_expand_px": None,
        "checks": per_expand_info,
    }


def det_to_component_dict(
    det: Det,
    component_id: str,
    source: str,
    merged_from: Optional[List[str]] = None,
    wire_connection: Optional[Dict] = None,
    flagged_cross_class_overlap: bool = False,
    flagged_disconnected: bool = False,
    class_name: Optional[str] = None,
) -> Dict:
    return {
        "component_id": component_id,
        "class_id": det.cls_id,
        "class_name": class_name,
        "confidence": det.conf,
        "bbox_xyxy": det.to_xyxy(),
        "bbox_xywh_abs": det.to_xywh_abs(),
        "source": source,
        "merged_from": merged_from or [],
        "flags": {
            "cross_class_overlap": flagged_cross_class_overlap,
            "disconnected": flagged_disconnected,
        },
        "wire_connection": (
            wire_connection
            if wire_connection is not None
            else {
                "checked": False,
                "connected": None,
                "connected_at_expand_px": None,
                "checks": [],
            }
        ),
        "manual_review": {
            "reviewed": False,
            "edited": False,
            "deleted": False,
            "notes": "",
        },
    }


# =========================
# Merge logic
# =========================


def build_raw_components(
    dets: List[Det],
    class_id_to_name: Optional[Dict[int, str]] = None,
) -> List[Dict]:
    raw_components = []
    for i, det in enumerate(dets):
        comp_id = f"raw_{i}"
        raw_components.append(
            det_to_component_dict(
                det=det,
                component_id=comp_id,
                source="raw_yolo",
                merged_from=[],
                wire_connection=None,
                flagged_cross_class_overlap=False,
                flagged_disconnected=False,
                class_name=(
                    class_id_to_name.get(det.cls_id) if class_id_to_name else None
                ),
            )
        )
    return raw_components


def merge_same_class_dets_json_first(
    dets: List[Det],
    raw_components: List[Dict],
    same_class_overlap_thr: float = 0.6,
    class_id_to_name: Optional[Dict[int, str]] = None,
) -> Tuple[List[Det], List[Dict], int]:
    """
    Merge highly overlapping boxes of the same class.
    Return:
      - merged_dets
      - current_components
      - merged_same_class_count
    """
    if not dets:
        return [], [], 0

    merged_dets: List[Det] = []
    current_components: List[Dict] = []
    merged_same_class_count = 0

    class_to_indices: Dict[int, List[int]] = {}
    for i, d in enumerate(dets):
        class_to_indices.setdefault(d.cls_id, []).append(i)

    comp_counter = 0

    for cls_id, idxs in class_to_indices.items():
        group_dets = [dets[i] for i in idxs]
        group_raw_components = [raw_components[i] for i in idxs]

        n = len(group_dets)
        dsu = DSU(n)

        for i in range(n):
            for j in range(i + 1, n):
                r = overlap_ratio_min_area(group_dets[i], group_dets[j])
                if r >= same_class_overlap_thr:
                    dsu.union(i, j)

        clusters: Dict[int, List[int]] = {}
        for i in range(n):
            root = dsu.find(i)
            clusters.setdefault(root, []).append(i)

        for cluster_indices in clusters.values():
            if len(cluster_indices) == 1:
                idx0 = cluster_indices[0]
                det = group_dets[idx0]
                raw_comp = group_raw_components[idx0]

                comp_id = f"comp_{comp_counter}"
                comp_counter += 1

                merged_dets.append(det)
                current_components.append(
                    det_to_component_dict(
                        det=det,
                        component_id=comp_id,
                        source="raw_yolo",
                        merged_from=[raw_comp["component_id"]],
                        wire_connection=None,
                        flagged_cross_class_overlap=False,
                        flagged_disconnected=False,
                        class_name=(
                            class_id_to_name.get(det.cls_id)
                            if class_id_to_name
                            else None
                        ),
                    )
                )
            else:
                merged_same_class_count += len(cluster_indices) - 1

                cluster_dets = [group_dets[k] for k in cluster_indices]
                cluster_raw_ids = [
                    group_raw_components[k]["component_id"] for k in cluster_indices
                ]

                x1 = min(d.x1 for d in cluster_dets)
                y1 = min(d.y1 for d in cluster_dets)
                x2 = max(d.x2 for d in cluster_dets)
                y2 = max(d.y2 for d in cluster_dets)
                conf = max(d.conf for d in cluster_dets)

                merged_det = Det(
                    cls_id=cls_id,
                    conf=conf,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )

                comp_id = f"comp_{comp_counter}"
                comp_counter += 1

                merged_dets.append(merged_det)
                current_components.append(
                    det_to_component_dict(
                        det=merged_det,
                        component_id=comp_id,
                        source="merged_same_class",
                        merged_from=cluster_raw_ids,
                        wire_connection=None,
                        flagged_cross_class_overlap=False,
                        flagged_disconnected=False,
                        class_name=(
                            class_id_to_name.get(merged_det.cls_id)
                            if class_id_to_name
                            else None
                        ),
                    )
                )

    return merged_dets, current_components, merged_same_class_count


# =========================
# Validation logic
# =========================


def detect_cross_class_overlaps(
    dets: List[Det],
    current_components: List[Dict],
    diff_class_overlap_thr: float = 0.35,
) -> List[Dict]:
    flagged_pairs = []
    n = len(dets)

    for i in range(n):
        for j in range(i + 1, n):
            if dets[i].cls_id == dets[j].cls_id:
                continue

            r = overlap_ratio_min_area(dets[i], dets[j])
            if r >= diff_class_overlap_thr:
                current_components[i]["flags"]["cross_class_overlap"] = True
                current_components[j]["flags"]["cross_class_overlap"] = True

                flagged_pairs.append(
                    {
                        "component_id_a": current_components[i]["component_id"],
                        "component_id_b": current_components[j]["component_id"],
                        "idx_a": i,
                        "idx_b": j,
                        "cls_a": dets[i].cls_id,
                        "cls_b": dets[j].cls_id,
                        "overlap_ratio": round(r, 4),
                        "box_a": dets[i].to_xyxy(),
                        "box_b": dets[j].to_xyxy(),
                    }
                )

    return flagged_pairs


# def run_wire_checks(
#     neededdebug: bool,
#     binary_img: np.ndarray,
#     dets: List[Det],
#     current_components: List[Dict],
#     expand_min: int,
#     expand_max: int,
#     min_black_pixels: int,
#     skip_wire_check_classes: set[int],
# ) -> List[Dict]:
#     disconnected_boxes = []

#     for i, det in enumerate(dets):
#         comp = current_components[i]

#         if det.cls_id in skip_wire_check_classes:
#             comp["wire_connection"] = {
#                 "checked": False,
#                 "connected": None,
#                 "connected_at_expand_px": None,
#                 "checks": [],
#                 "skipped": True,
#                 "skip_reason": "class_in_skip_wire_check_classes",
#             }
#             continue

#         connected, info = has_wire_connection(
#             neededdebug,
#             binary_img=binary_img,
#             det=det,
#             expand_min=expand_min,
#             expand_max=expand_max,
#             min_black_pixels=min_black_pixels,
#         )

#         info["skipped"] = False
#         comp["wire_connection"] = info

#         if not connected:
#             comp["flags"]["disconnected"] = True
#             disconnected_boxes.append(
#                 {
#                     "component_id": comp["component_id"],
#                     "idx": i,
#                     "cls_id": det.cls_id,
#                     "box": det.to_xyxy(),
#                     "details": info,
#                 }
#             )


#     return disconnected_boxes
def run_wire_checks(
    neededdebug: bool,
    binary_img: np.ndarray,
    dets: List[Det],
    current_components: List[Dict],
    expand_min: int,
    expand_max: int,
    min_black_pixels: int,
    skip_wire_check_classes: set[int],
) -> Tuple[List[Det], List[Dict], List[Dict]]:
    """
    Run wire checks on current components.

    Returns:
      - kept_dets: components that passed / were kept
      - kept_components: JSON component entries that remain in components_current
      - disconnected_boxes: removed components recorded for audit/debug
    """
    kept_dets: List[Det] = []
    kept_components: List[Dict] = []
    disconnected_boxes = 0

    for i, det in enumerate(dets):
        comp = current_components[i]

        if det.cls_id in skip_wire_check_classes:
            comp["wire_connection"] = {
                "checked": False,
                "connected": None,
                "connected_at_expand_px": None,
                "checks": [],
                "skipped": True,
                "skip_reason": "class_in_skip_wire_check_classes",
            }
            kept_dets.append(det)
            kept_components.append(comp)
            continue

        connected, info = has_wire_connection(
            neededdebug,
            binary_img=binary_img,
            det=det,
            expand_min=expand_min,
            expand_max=expand_max,
            min_black_pixels=min_black_pixels,
        )

        info["skipped"] = False
        comp["wire_connection"] = info

        if connected:
            kept_dets.append(det)
            kept_components.append(comp)
        else:
            comp["flags"]["disconnected"] = True  # has been removed
            comp["manual_review"]["deleted"] = True
            comp["manual_review"]["notes"] = "Auto-removed by wire connection check"

            disconnected_boxes += 1
            # .append(
            #     {
            #         "component_id": comp["component_id"],
            #         "idx": i,
            #         "cls_id": det.cls_id,
            #         "box": det.to_xyxy(),
            #         "details": info,
            #         "removed_from_current": True,
            #     }
            # )

    return kept_dets, kept_components, disconnected_boxes


# =========================
# JSON / export / visualization
# =========================


def build_image_json(
    image_path: Path,
    image_width: int,
    image_height: int,
    raw_components: List[Dict],
    current_components: List[Dict],
    merged_same_class_count: int,
    cross_class_overlap_pairs: List[Dict],
    disconnected_boxes: List[Dict],
) -> Dict:
    cross_class_overlap_flag = len(cross_class_overlap_pairs) > 0
    # disconnected_component_flag = len(disconnected_boxes) > 0
    final_flag = cross_class_overlap_flag  # or disconnected_component_flag
    removed_disconnected_components_flag = disconnected_boxes > 0

    return {
        "image_name": image_path.name,
        "image_path": str(image_path),
        "image_width": image_width,
        "image_height": image_height,
        "flags": {
            "cross_class_overlap_flag": cross_class_overlap_flag,
            "disconnected_component_flag": removed_disconnected_components_flag,
            "final_flag": final_flag,
        },
        "summary": {
            "raw_component_count": len(raw_components),
            "merged_component_count": len(current_components),
            "merged_same_class_count": merged_same_class_count,
            "cross_class_overlap_pair_count": len(cross_class_overlap_pairs),
            "auto_removed_disconnected_count": disconnected_boxes,
        },
        "cross_class_overlap_pairs": cross_class_overlap_pairs,
        # "disconnected_components": disconnected_boxes,
        "components_raw": raw_components,
        "components_current": current_components,
    }


def save_image_json(data: Dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_yolo_txt_from_components(
    components_current: List[Dict],
    out_txt: Path,
    img_w: int,
    img_h: int,
):
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    with open(out_txt, "w", encoding="utf-8") as f:
        for comp in components_current:
            if comp["manual_review"]["deleted"]:
                continue

            cls_id = comp["class_id"]
            conf = comp["confidence"]
            x1, y1, x2, y2 = comp["bbox_xyxy"]

            det = Det(
                cls_id=cls_id,
                conf=conf,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            )
            f.write(det.to_yolo_line(img_w, img_h) + "\n")


# def draw_visualization(
#     img_gray: np.ndarray,
#     components_current: List[Dict],
#     cross_pairs: List[Dict],
#     disconnected_boxes: List[Dict],
#     out_path: Path,
# ):
#     vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

#     component_id_to_box = {
#         comp["component_id"]: comp["bbox_xyxy"] for comp in components_current
#     }

#     for comp in components_current:
#         x1, y1, x2, y2 = comp["bbox_xyxy"]
#         cls_id = comp["class_id"]
#         comp_id = comp["component_id"]

#         cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(
#             vis,
#             f"{comp_id}:cls{cls_id}",
#             (x1, max(0, y1 - 5)),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.45,
#             (0, 180, 0),
#             1,
#             cv2.LINE_AA,
#         )

#     for pair in cross_pairs:
#         a = pair["box_a"]
#         b = pair["box_b"]
#         cv2.rectangle(vis, (a[0], a[1]), (a[2], a[3]), (0, 0, 255), 2)
#         cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)

#     for item in disconnected_boxes:
#         x1, y1, x2, y2 = item["box"]
#         cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
#         cv2.putText(
#             vis,
#             "DISCONNECTED",
#             (x1, min(vis.shape[0] - 5, y2 + 15)),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.5,
#             (255, 0, 0),
#             2,
#             cv2.LINE_AA,
#         )

#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     cv2.imwrite(str(out_path), vis)


def save_index_json(index_list: List[Dict], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(index_list, f, indent=2, ensure_ascii=False)


# =========================
# Main image-level validation
# =========================


def validate_one_image(
    image_path: Path,
    label_path: Path,
    json_out_path: Path,
    vis_out_path: Path,
    export_txt_out_path: Optional[Path],
    same_class_overlap_thr: float,
    diff_class_overlap_thr: float,
    expand_min: int,
    expand_max: int,
    min_black_pixels: int,
    skip_wire_check_classes: set[int],
    class_id_to_name: Optional[Dict[int, str]] = None,
    export_current_labels_txt: bool = True,
) -> Dict:
    binary_img = read_binary_image(image_path)
    img_h, img_w = binary_img.shape[:2]

    # 1) load raw YOLO detections
    raw_dets = load_yolo_txt(label_path, img_w, img_h)
    raw_components = build_raw_components(
        dets=raw_dets,
        class_id_to_name=class_id_to_name,
    )

    # 2) merge same-class overlapping detections
    merged_dets, current_components, merged_same_class_count = (
        merge_same_class_dets_json_first(
            dets=raw_dets,
            raw_components=raw_components,
            same_class_overlap_thr=same_class_overlap_thr,
            class_id_to_name=class_id_to_name,
        )
    )

    # 3) detect cross-class overlaps
    cross_class_overlap_pairs = detect_cross_class_overlaps(
        dets=merged_dets,
        current_components=current_components,
        diff_class_overlap_thr=diff_class_overlap_thr,
    )

    # 4) only if no cross-class overlap, run wire checks
    needdebug = False
    if "100" in str(image_path):
        print(str(image_path))
        needdebug = True
    disconnected_boxes = 0
    if len(cross_class_overlap_pairs) == 0:
        merged_dets, current_components, disconnected_boxes = run_wire_checks(
            needdebug,
            binary_img=binary_img,
            dets=merged_dets,
            current_components=current_components,
            expand_min=expand_min,
            expand_max=expand_max,
            min_black_pixels=min_black_pixels,
            skip_wire_check_classes=skip_wire_check_classes,
        )
        if needdebug == True:
            needdebug = False
    else:
        for comp in current_components:
            comp["wire_connection"] = {
                "checked": False,
                "connected": None,
                "connected_at_expand_px": None,
                "checks": [],
                "skipped": True,
                "skip_reason": "cross_class_overlap_flagged_first",
            }

    # 5) build JSON object
    image_json = build_image_json(
        image_path=image_path,
        image_width=img_w,
        image_height=img_h,
        raw_components=raw_components,
        current_components=current_components,
        merged_same_class_count=merged_same_class_count,
        cross_class_overlap_pairs=cross_class_overlap_pairs,
        disconnected_boxes=disconnected_boxes,
    )

    # 6) save JSON
    save_image_json(image_json, json_out_path)

    # 7) save visualization
    # draw_visualization(
    #     img_gray=binary_img,
    #     components_current=current_components,
    #     cross_pairs=cross_class_overlap_pairs,
    #     disconnected_boxes=disconnected_boxes,
    #     out_path=vis_out_path,
    # )

    # 8) optional: export current labels txt
    if export_current_labels_txt and export_txt_out_path is not None:
        save_yolo_txt_from_components(
            components_current=current_components,
            out_txt=export_txt_out_path,
            img_w=img_w,
            img_h=img_h,
        )

    return image_json


# =========================
# Folder-level validation
# =========================


def validate_folder(cfg: dict):
    pred_root = Path(cfg["output_dir"]) / "yolo_pred"
    output_dir = Path(cfg["output_dir"]) / "postcheck"

    labels_dir = pred_root / "labels"
    images_dir = Path(cfg["preprocessed_imgs"])

    json_dir = output_dir / "json"
    vis_dir = output_dir / "validation_vis"
    export_label_dir = output_dir / "export_labels"
    index_json_path = output_dir / "index.json"

    same_class_overlap_thr = cfg.get("same_class_overlap_thr", 0.6)
    diff_class_overlap_thr = cfg.get("diff_class_overlap_thr", 0.35)
    expand_min = cfg.get("expand_min", 1)
    expand_max = cfg.get("expand_max", 5)
    min_black_pixels = cfg.get("min_black_pixels", 3)
    skip_wire_check_classes = set(cfg.get("skip_wire_check_classes", []))
    image_exts = cfg.get(
        "image_exts", [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]
    )
    export_current_labels_txt = cfg.get("export_current_labels_txt", True)

    class_id_to_name_raw = cfg.get("class_id_to_name", None)
    if class_id_to_name_raw is not None:
        class_id_to_name = {int(k): v for k, v in class_id_to_name_raw.items()}
    else:
        class_id_to_name = None

    image_paths = find_images(images_dir, image_exts)
    index_list = []

    for image_path in image_paths:
        label_path = labels_dir / f"{image_path.stem}.txt"
        json_out_path = json_dir / f"{image_path.stem}.json"
        vis_out_path = vis_dir / image_path.name
        export_txt_out_path = export_label_dir / f"{image_path.stem}.txt"

        image_json = validate_one_image(
            image_path=image_path,
            label_path=label_path,
            json_out_path=json_out_path,
            vis_out_path=vis_out_path,
            export_txt_out_path=export_txt_out_path,
            same_class_overlap_thr=same_class_overlap_thr,
            diff_class_overlap_thr=diff_class_overlap_thr,
            expand_min=expand_min,
            expand_max=expand_max,
            min_black_pixels=min_black_pixels,
            skip_wire_check_classes=skip_wire_check_classes,
            class_id_to_name=class_id_to_name,
            export_current_labels_txt=export_current_labels_txt,
        )

        final_flag = image_json["flags"]["final_flag"]
        raw_count = image_json["summary"]["raw_component_count"]
        merged_count = image_json["summary"]["merged_component_count"]

        index_list.append(
            {
                "image_name": image_json["image_name"],
                "image_path": image_json["image_path"],
                "json_path": str(json_out_path),
                "visualization_path": str(vis_out_path),
                "export_label_path": (
                    str(export_txt_out_path) if export_current_labels_txt else None
                ),
                "final_flag": final_flag,
                "cross_class_overlap_flag": image_json["flags"][
                    "cross_class_overlap_flag"
                ],
                "disconnected_component_flag": image_json["flags"][
                    "disconnected_component_flag"
                ],
                "raw_component_count": raw_count,
                "merged_component_count": merged_count,
            }
        )

        print(
            f"[CHECKED] {image_json['image_name']} | "
            f"raw={raw_count} | "
            f"current={merged_count} | "
            f"cross_flag={image_json['flags']['cross_class_overlap_flag']} | "
            f"disconnect_flag={image_json['flags']['disconnected_component_flag']} | "
            f"final_flag={final_flag}"
        )

    save_index_json(index_list, index_json_path)
    print(f"[DONE] Per-image JSON saved to: {json_dir}")
    print(f"[DONE] Index JSON saved to: {index_json_path}")
