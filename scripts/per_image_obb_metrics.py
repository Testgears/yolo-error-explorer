# scripts/per_image_obb_metrics.py

import argparse
from pathlib import Path

import pandas as pd
import torch
from ultralytics.utils.metrics import batch_probiou
from ultralytics.utils.ops import xyxyxyxy2xywhr


def read_obb_txt(path: Path, has_conf: bool = False, conf_thres: float = 0.25):
    if not path.exists() or not path.read_text().strip():
        return torch.empty((0, 6 if has_conf else 5))

    rows = []

    for line in path.read_text().strip().splitlines():
        parts = line.split()

        if has_conf and len(parts) < 10:
            continue
        if not has_conf and len(parts) < 9:
            continue

        cls = float(parts[0])
        coords = list(map(float, parts[1:9]))

        # Convert 4-point OBB format to xywhr format used by Ultralytics OBB IoU
        pts = torch.tensor(coords, dtype=torch.float32).view(1, 4, 2)
        xywhr = xyxyxyxy2xywhr(pts).squeeze(0)

        if has_conf:
            conf = float(parts[9])
            if conf < conf_thres:
                continue
            rows.append([cls, *xywhr.tolist(), conf])
        else:
            rows.append([cls, *xywhr.tolist()])

    return torch.tensor(rows, dtype=torch.float32)


def evaluate_image(gt, pred, iou_thres):
    """
    gt shape:   [N, 5] -> class, x, y, w, h, r
    pred shape: [M, 6] -> class, x, y, w, h, r, conf
    """

    # No GT and no predictions
    if len(gt) == 0 and len(pred) == 0:
        return 0, 0, 0, 0.0, 0.0, 0.0, []

    # No GT, but there are predictions
    if len(gt) == 0:
        return 0, len(pred), 0, 0.0, 0.0, 0.0, []

    # GT exists, but there are no predictions
    if len(pred) == 0:
        missed_gt_indices = list(range(len(gt)))
        return 0, 0, len(gt), 0.0, 0.0, 0.0, missed_gt_indices

    gt_cls = gt[:, 0]
    gt_boxes = gt[:, 1:6]

    pred_cls = pred[:, 0]
    pred_boxes = pred[:, 1:6]
    pred_conf = pred[:, 6]

    # Sort predictions by confidence, same general matching idea used in detection eval
    order = torch.argsort(pred_conf, descending=True)
    pred_cls = pred_cls[order]
    pred_boxes = pred_boxes[order]

    ious = batch_probiou(gt_boxes, pred_boxes)

    matched_gt = set()
    matched_pred = set()

    for pred_idx in range(len(pred_boxes)):
        best_iou = 0.0
        best_gt_idx = None

        for gt_idx in range(len(gt_boxes)):
            if gt_idx in matched_gt:
                continue

            if int(pred_cls[pred_idx].item()) != int(gt_cls[gt_idx].item()):
                continue

            iou = float(ious[gt_idx, pred_idx].item())

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx is not None and best_iou >= iou_thres:
            matched_gt.add(best_gt_idx)
            matched_pred.add(pred_idx)

    tp = len(matched_pred)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    missed_gt_indices = [
        i for i in range(len(gt_boxes))
        if i not in matched_gt
        ]

    return tp, fp, fn, precision, recall, f1, missed_gt_indices


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO OBB predictions and generate per-image metrics."
    )

    parser.add_argument(
        "--gt",
        required=True,
        help="Path to the ground-truth YOLO OBB labels folder"
    )
    parser.add_argument(
        "--pred",
        required=True,
        help="Path to the YOLO OBB prediction labels folder"
    )
    parser.add_argument(
        "--output",
        default="per_image_metrics.csv",
        help="Path for the generated metrics CSV"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IoU threshold used for matching predictions to ground truth (default: 0.5)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Minimum prediction confidence threshold (default: 0.25)"
    )

    args = parser.parse_args()

    gt_dir = Path(args.gt)
    pred_dir = Path(args.pred)

    # Validate input directories
    if not gt_dir.exists():
        raise SystemExit(
            f"Error: Ground-truth directory does not exist: {gt_dir}"
        )

    if not pred_dir.exists():
        raise SystemExit(
            f"Error: Prediction directory does not exist: {pred_dir}"
        )

    if not gt_dir.is_dir():
        raise SystemExit(
            f"Error: Ground-truth path is not a directory: {gt_dir}"
        )

    if not pred_dir.is_dir():
        raise SystemExit(
            f"Error: Prediction path is not a directory: {pred_dir}"
        )

    # Find all label files present in either directory
    all_names = sorted(
        {p.name for p in gt_dir.glob("*.txt")}
        | {p.name for p in pred_dir.glob("*.txt")}
    )

    if not all_names:
        raise SystemExit(
            "Error: No .txt label files found. "
            "Check the paths provided to --gt and --pred."
        )

    rows = []

    # Evaluate each image
    for name in all_names:
        gt_file = gt_dir / name
        pred_file = pred_dir / name

        gt = read_obb_txt(
            gt_file,
            has_conf=False
        )

        pred = read_obb_txt(
            pred_file,
            has_conf=True,
            conf_thres=args.conf
        )

        tp, fp, fn, precision, recall, f1, missed_gt_indices = evaluate_image(
            gt,
            pred,
            args.iou
        )

        rows.append({
            "image": Path(name).stem,
            "gt": len(gt),
            "pred": len(pred),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "missed_gt_indices": ";".join(map(str, missed_gt_indices)),
        })

    # Create and sort results
    df = pd.DataFrame(rows)

    df = df.sort_values(
        ["fn", "recall", "fp"],
        ascending=[False, True, False]
    )

    # Save CSV
    df.to_csv(args.output, index=False)

    print(f"Saved: {args.output}")
    print(f"Images evaluated: {len(df)}")

    print("\nTop images with most false negatives:")
    print(df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()