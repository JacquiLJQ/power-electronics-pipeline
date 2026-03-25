# run_yolo_infer.py

from pathlib import Path
from ultralytics import YOLO


def run_inference(cfg: dict):
    """
    Run YOLO inference on a folder of images and save:
      - annotated images
      - YOLO txt labels
      - confidence values in txt
    """
    weights = cfg["weights"]
    source = str(Path(cfg["preprocessed_imgs"]))
    output_dir = str((Path(cfg["output_dir"])).resolve())
    imgsz = cfg.get("imgsz", 640)
    # conf = cfg.get("conf", 0.25)
    # iou = cfg.get("iou", 0.7)
    device = cfg.get("device", "0")

    model = YOLO(weights)

    model.predict(
        source=source,
        imgsz=imgsz,
        save=True,
        save_txt=True,
        device=device,
        save_conf=True,
        project=output_dir,
        name="yolo_pred",
        exist_ok=True,
        verbose=True,
    )

    print(
        f"[DONE] YOLO inference finished. Results saved to: {Path(output_dir) / 'yolo_pred'}"
    )


def main():
    print("=== Running YOLO Inference Standalone ===")

    try:
        from config import CONFIG

        cfg = CONFIG
        print("[INFO] Loaded config.py")
    except Exception:
        print("[WARN] config.py not found, quiting")
        return

    # sanity check
    if not Path(cfg["preprocessed_imgs"]).exists():
        raise RuntimeError(
            f"Preprocessed image folder not found: {cfg['preprocessed_imgs']}\n"
            f"Run preprocess_images.py first."
        )

    run_inference(cfg)


if __name__ == "__main__":
    main()
