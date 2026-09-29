YOLO Error Explorer is an interactive browser-based tool for visualizing and analyzing errors in YOLO oriented bounding box (OBB) detection results.

It combines per-image evaluation metrics with visual inspection of ground-truth and predicted bounding boxes, making it easier to identify false negatives, false positives, and recurring model failure patterns.

Features

- Visualize YOLO OBB predictions alongside ground-truth annotations

- Highlight missed ground-truth objects (false negatives)

- Display per-image precision, recall, and F1 score

- Filter images by false-negative count

- Sort images by false negatives, recall, F1 score, or filename

- Toggle ground-truth and prediction overlays

- Generate dashboard-compatible metrics directly from YOLO OBB label files

- Process images and labels locally in the browser

## Project Structure

yolo-error-explorer/
├── index.html
├── README.md
├── requirements.txt
├── scripts/
│   └── per_image_obb_metrics.py
└── examples/
    └── per_image_metrics.example.csv

## Installation
- Python 3.10 or later is recommended.
- Create and activate a virtual environment:
`python3 -m venv .venv`
`source .venv/bin/activate`

- Install the required Python dependencies:
`pip install -r requirements.txt`

## Generating the Metrics CSV
The included Python evaluator converts YOLO OBB ground-truth and prediction labels into the metrics CSV used by the dashboard.

Run:
```
python scripts/per_image_obb_metrics.py \
  --gt path/to/ground_truth/labels \
  --pred path/to/prediction/labels \
  --output per_image_metrics.csv
```

Optional arguments:
```
--iou     IoU threshold used for matching predictions to ground truth
          Default: 0.5

--conf    Minimum prediction confidence threshold
          Default: 0.25
```

Example:

```
python scripts/per_image_obb_metrics.py \
  --gt data/ground_truth_labels \
  --pred data/prediction_labels \
  --output per_image_metrics.csv \
  --iou 0.5 \
  --conf 0.25
```

The evaluator uses Ultralytics ProbIoU for OBB matching. It also records the indices of unmatched ground-truth boxes so that the dashboard uses the same evaluation results when highlighting missed detections.


## Input files

YOLO Error Explorer requires a metrics CSV and the corresponding
validation images.

### Required

- `per_image_metrics.csv` — per-image evaluation metrics
- Validation images — the images used during model evaluation

### Optional

- Ground-truth label files — used to visualize ground-truth boxes
- Prediction label files — used to visualize model predictions

Providing both label folders enables full visual error analysis, including highlighting missed ground-truth objects.

### Metrics CSV format

The CSV must contain the following columns:

| Column | Description |
|---|---|
| image | Image filename/stem |
| gt | Number of ground-truth objects |
| pred | Number of predicted objects |
| tp | True positives |
| fp | False positives |
| fn | False negatives |
| precision | Precision for the image |
| recall | Recall for the image |
| f1 | F1 score for the image |
| missed_gt_indices | Indices of unmatched ground-truth boxes used for visualization |

Example:

```csv
image,gt,pred,tp,fp,fn,precision,recall,f1,,missed_gt_indices
image_001,5,4,4,0,1,1.0,0.8,0.889,2
image_002,3,3,3,0,0,1.0,1.0,1.0,
image_003,5,4,3,1,2,0.75,0.6,0.6667,1;4
```

## YOLO OBB Label Format

Ground-truth labels are expected in YOLO OBB format:

`class x1 y1 x2 y2 x3 y3 x4 y4`

Prediction labels must additionally contain a confidence score:

`class x1 y1 x2 y2 x3 y3 x4 y4 confidence`

Coordinates are expected to be normalized relative to the image dimensions.

The label filename should correspond to the image filename.

For example:

```
image_001.jpg
image_001.txt
```

Using the Dashboard

Open index.html in a browser.

Then select:

1. The generated metrics CSV

2. The corresponding validation images

3. Ground-truth labels, if available

4. Prediction labels, if available

After loading the files, select an image from the sidebar to inspect its metrics and detection results.

Ground-truth boxes are displayed in green, missed ground-truth objects are highlighted in red, and model predictions are displayed in blue.

## Evaluation Design
Model evaluation is performed by the Python preprocessing script rather than independently in the browser.

The evaluator performs class-aware matching between predictions and ground-truth OBBs using Ultralytics ProbIoU. The resulting false-negative indices are written to the metrics CSV and consumed directly by the dashboard.

This keeps metric computation and visual error highlighting consistent and provides a single source of truth for evaluation results.

## Privacy
Images and label files selected through the dashboard are processed locally in the browser. The static dashboard does not require the dataset to be uploaded to a server.