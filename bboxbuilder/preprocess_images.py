# preprocess_images.py

from pathlib import Path
import cv2


def binarize_image(img_bgr, use_otsu=True, binary_threshold=127, binary_invert=False):
    """
    Convert image to pure black/white.
    Output pixel values are only 0 or 255.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if use_otsu:
        threshold_type = cv2.THRESH_BINARY_INV if binary_invert else cv2.THRESH_BINARY
        _, binary = cv2.threshold(gray, 0, 255, threshold_type | cv2.THRESH_OTSU)
    else:
        threshold_type = cv2.THRESH_BINARY_INV if binary_invert else cv2.THRESH_BINARY
        _, binary = cv2.threshold(gray, binary_threshold, 255, threshold_type)

    return binary


def preprocess_input_images(cfg: dict):
    source_dir = Path(cfg["source"])
    output_dir = Path(cfg["preprocessed_imgs"])
    output_dir.mkdir(parents=True, exist_ok=True)

    image_exts = {e.lower() for e in cfg.get("image_exts", [".jpg", ".jpeg", ".png"])}
    use_otsu = cfg.get("use_otsu_binarize", True)
    binary_threshold = cfg.get("binary_threshold", 127)
    binary_invert = cfg.get("binary_invert", False)

    image_paths = sorted(
        [p for p in source_dir.iterdir() if p.suffix.lower() in image_exts]
    )

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Failed to read image: {img_path}")
            continue

        binary = binarize_image(
            img_bgr=img,
            use_otsu=use_otsu,
            binary_threshold=binary_threshold,
            binary_invert=binary_invert,
        )

        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), binary)

    print(f"[DONE] Preprocessed binary images saved to: {output_dir}")
    return output_dir


def main():
    print("=== Running Preprocess (Binarization) ===")

    try:
        from config import CONFIG

        cfg = CONFIG
        print("[INFO] Loaded config.py")
    except Exception:
        print("[WARN] config.py not found")
        return

    if not Path(cfg["source"]).exists():
        raise RuntimeError(f"Input image folder not found: {cfg['source']}")

    preprocess_input_images(cfg)


if __name__ == "__main__":
    main()
