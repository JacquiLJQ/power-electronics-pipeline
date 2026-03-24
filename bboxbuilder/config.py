# config.py

CONFIG = {
    # ---------- paths ----------
    "weights": r"weights/best.pt",
    "source": r"data/raw/circuitnet",
    "output_dir": r"runs/pipeline_test",
    "preprocessed_imgs": r"data/preprocessed/circuitnet",
    # ---------- yolo inference ----------
    "imgsz": 640,
    # "conf": 0.25,
    # "iou": 0.7,
    # "device": "0",
    # ---------- image preprocess ----------
    "use_otsu_binarize": True,
    "binary_threshold": 127,  # only used if use_otsu_binarize = False
    "binary_invert": False,
    # ---------- bbox validation ----------
    # same-class boxes with heavy overlap -> merge first
    "same_class_overlap_thr": 0.35,
    # different-class boxes with heavy overlap -> raise flag
    "diff_class_overlap_thr": 0.35,
    # expand bbox outward by 1~5 pixels and check whether wire exists
    "expand_min": 1,
    "expand_max": 5,
    # for white background + black wire/component schematics
    # "black_thresh": 120,
    "min_black_pixels": 3,
    # classes to skip wire-connection check
    # e.g. text, labels, arrows, etc.
    "skip_wire_check_classes": [],
    # optional: image suffixes
    "image_exts": [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"],
}
