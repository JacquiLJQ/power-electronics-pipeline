# main_pipeline.py

from config import CONFIG
from preprocess_images import preprocess_input_images
from run_yolo_infer import run_inference
from validate_bboxes import validate_folder


def main():
    cfg = CONFIG

    print("========== PIPELINE CONFIG ==========")
    for k, v in cfg.items():
        print(f"{k}: {v}")
    print("=====================================")

    # Step 0: preprocess input images into pure black/white
    # preprocess_input_images(cfg)

    # Step 1: YOLO inference on preprocessed images
    # run_inference(cfg)

    # Step 2: bbox validation on preprocessed prediction results
    validate_folder(cfg)

    print("\nPipeline finished.")
    # print(f"Preprocessed input: {cfg['output_dir']}/preprocessed_inputs")
    # print(f"YOLO output:        {cfg['output_dir']}/yolo_pred")
    # print(f"Postcheck output:   {cfg['output_dir']}/postcheck")


if __name__ == "__main__":
    main()
