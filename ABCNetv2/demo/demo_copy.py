# demo.py
# ABCNetv2 Detection + SFX Cropping
import argparse
import glob
import multiprocessing as mp
import os
import time
import json
import cv2
import tqdm

from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from predictor import VisualizationDemo
from adet.config import get_cfg
from detectron2.config import CfgNode

WINDOW_NAME = "COCO detections"

def add_custom_configs(cfg: CfgNode):
    """Add custom configs for ABCNetv2 if needed."""
    _C = cfg
    _C.SOLVER.BEST_CHECKPOINTER = CfgNode({"ENABLED": False})
    _C.SOLVER.BEST_CHECKPOINTER.METRIC = "bbox/AP50"
    _C.SOLVER.BEST_CHECKPOINTER.MODE = "max"
    _C.TRAIN_LOG_PERIOD = 200

def setup_cfg(args):
    """Load config and set thresholds"""
    cfg = get_cfg()
    add_custom_configs(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # Set score thresholds for multiple heads
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.MEInst.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold

    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="ABCNetv2 Demo with SFX Cropping")
    parser.add_argument(
        "--config-file",
        default="configs/eval.yaml",
        metavar="FILE",
        help="Path to config file",
    )
    parser.add_argument(
        "--input", nargs="+", help="A list of space separated input images or folders"
    )
    parser.add_argument(
        "--output",
        help="Directory to save outputs and crops. Required.",
        required=True
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)

    # Expand input paths
    input_images = []
    for path in args.input:
        if os.path.isdir(path):
            for fname in os.listdir(path):
                if fname.lower().endswith((".jpg", ".png")):
                    input_images.append(os.path.join(path, fname))
        elif os.path.isfile(path):
            input_images.append(path)
        else:
            input_images.extend(glob.glob(os.path.expanduser(path)))

    if not input_images:
        raise ValueError("No valid input images found.")

    # Ensure output folder exists
    os.makedirs(args.output, exist_ok=True)

    # Store coordinates for all images
    all_coords = {}

    for path in tqdm.tqdm(input_images, desc="Processing images"):
        # Read image
        img = read_image(path, format="BGR")
        start_time = time.time()

        # Run detection
        predictions, visualized_output = demo.run_on_image(img)
        num_instances = len(predictions["instances"]) if "instances" in predictions else 0
        logger.info(f"{path}: detected {num_instances} instances in {time.time() - start_time:.2f}s")

        # Prepare crop folder for this image
        filename_base = os.path.splitext(os.path.basename(path))[0]
        crop_dir = os.path.join(args.output, "crops", filename_base)
        os.makedirs(crop_dir, exist_ok=True)
        all_coords[filename_base] = []

        # Crop each SFX instance
        if "instances" in predictions:
            instances = predictions["instances"].to("cpu")
            if instances.has("pred_boxes"):
                boxes = instances.pred_boxes.tensor.numpy()
                for idx, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box)
                    h, w = img.shape[:2]
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

                    if x2 > x1 and y2 > y1:
                        crop = img[y1:y2, x1:x2]
                        crop_path = os.path.join(crop_dir, f"sfx_{idx}.png")
                        cv2.imwrite(crop_path, crop)
                        all_coords[filename_base].append({
                            "file": f"sfx_{idx}.png",
                            "box": [x1, y1, x2, y2]
                        })

        # Save visualized output
        out_file = os.path.join(args.output, os.path.basename(path))
        try:
            visualized_output.save(out_file)
        except Exception as e:
            print(f"⚠ Could not save visualized output for {path}: {e}")

    # Save coordinates JSON
    coords_file = os.path.join(args.output, "coordinates.json")
    with open(coords_file, "w") as f:
        json.dump(all_coords, f, indent=2)
    print(f"✓ All crops and coordinates saved in {args.output}/")
