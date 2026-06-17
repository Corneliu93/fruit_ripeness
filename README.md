# Image-Based Fruit Ripeness Classification using CNNs and Transfer Learning

A deep learning project for automated fruit ripeness detection using Convolutional Neural Networks
and MobileNetV2 transfer learning. Classifies **9 classes** across 3 fruit types (apple, banana, pear)
× 3 ripeness stages (unripe, ripe, overripe).

---

## Project Structure

```
fruit_ripeness/
├── dataset/                          (original captured images; git-ignored)
│   ├── apple/        {unripe, ripe, overripe}
│   ├── banana/       {unripe, ripe, overripe}
│   ├── pear/         {unripe, ripe, overripe}
│   ├── metadata.csv
│   └── capture_instructions.md
├── dataset_splits/                   (derived from results/data_split.csv by
│   │                                  src.data_pipeline.build_dataset_splits_from_csv; git-ignored)
│   ├── train/  (9 class folders: apple_unripe ... pear_overripe — 315 images)
│   ├── val/    (the same 9 class folders — 63 images)
│   └── test/   (the same 9 class folders — 72 images)
├── src/
│   ├── utils.py            (utilities, class definitions and ordering, seed management)
│   ├── models.py           (Baseline CNN and MobileNetV2 builders)
│   ├── data_pipeline.py    (tf.data pipelines: baseline and MobileNetV2)
│   ├── train.py            (training script with CLI arguments)
│   └── evaluate.py         (metrics, confusion matrix, ROC curves, Grad-CAM)
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_train_baseline.ipynb
│   ├── 03_evaluation_baseline.ipynb
│   ├── 04_train_mobilenetv2_phase1.ipynb
│   ├── 05_train_mobilenetv2_phase2.ipynb
│   ├── 06_evaluation_mobilenetv2.ipynb
│   └── 07_gradcam_mobilenetv2.ipynb
├── saved_models/                     (trained checkpoints; git-ignored)
│   ├── baseline_best.h5
│   ├── mobilenet_phase1_best.h5
│   └── mobilenet_phase2_best.h5
├── results/                          (generated artefacts are git-ignored)
│   ├── data_split.csv
│   ├── results.csv
│   ├── class_names.json
│   ├── test_evaluation_comparison.csv
│   ├── history_baseline.csv
│   ├── history_mobilenet_phase1.csv
│   ├── history_mobilenet_phase2.csv
│   ├── classification_report_baseline.txt
│   ├── classification_report_mobilenet_phase1.txt
│   ├── classification_report_mobilenet_phase2.txt
│   ├── confusion_matrix_baseline.png
│   ├── confusion_matrix_mobilenet_phase1.png
│   ├── confusion_matrix_mobilenet_phase2.png
│   ├── roc_curves_baseline.png
│   ├── roc_curves_mobilenet_phase1.png
│   ├── roc_curves_mobilenet_phase2.png
│   ├── training_curves_baseline.png
│   ├── mobilenet_phase1_training_curves.png
│   ├── mobilenet_phase2_training_curves.png
│   ├── gradcam_mobilenet_phase2.png
│   ├── raw_samples.png
│   ├── preprocessing_demo.png
│   ├── augmentation_demo.png
│   └── training_batch_sample.png
├── web_app/
│   ├── app.py
│   ├── templates/index.html
│   ├── static/uploads/
│   └── model/                        (deployed copy of mobilenet_phase2_best.h5; added in Week 8)
├── .gitignore
├── check_split.py
├── dataset_check.py
├── manifest.txt
├── README.md
├── rename_images.py
└── requirements.txt
```

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell
# source .venv/bin/activate         # macOS / Linux
pip install -r requirements.txt
```

---

## Dataset

Target: **50 images per class**, 9 classes = 450 images total. Each class combines self-captured
photographs with a small proportion of public-source images (see `dataset/metadata.csv` for the
breakdown). The original images live under `dataset/<fruit>/<stage>/`.

See `dataset/capture_instructions.txt` for the photo capture protocol, and run `dataset_check.py`
to validate class counts.

> `src/data_pipeline.py` also includes an optional synthetic-image generator
> (`generate_synthetic_dataset`) as a documented fallback. It was not required for this project.

---

## Class ordering (important)

Two class orderings are used and reconciled in code:

- **Canonical** (`src.utils.CLASS_NAMES`): grouped by fruit then ripeness
  (`apple_unripe, apple_ripe, apple_overripe, ...`). Used by the baseline pipeline and in the report.
- **Alphabetical**: the order produced by `tf.keras.utils.image_dataset_from_directory`, used while
  training the MobileNetV2 models.

At evaluation time, MobileNetV2 predictions are permuted from alphabetical to canonical order
(`src.utils.alpha_probs_to_canonical`) so that every figure and table uses one consistent ordering.

---

## Training

### Baseline CNN (CLI)
```bash
python src/train.py --model baseline --data_dir ./dataset --batch_size 32 --epochs 50 --seed 42
```

### MobileNetV2 (notebooks)
MobileNetV2 uses the `dataset_splits/` pipeline with MobileNetV2 pre-processing, so it is trained
through the notebooks rather than the baseline CLI:

1. `notebooks/04_train_mobilenetv2_phase1.ipynb` — Phase 1, frozen base, head-only training.
2. `notebooks/05_train_mobilenetv2_phase2.ipynb` — Phase 2, fine-tuning from layer 100 (BatchNorm
   kept frozen) at a reduced learning rate.

Running notebook 04 also builds `dataset_splits/` from `results/data_split.csv` on first execution.

---

## Evaluation

Full test-set evaluation (per-class metrics, confusion matrix, ROC curves) and Grad-CAM are produced
by the notebooks, which call `src/evaluate.py`:

- `notebooks/06_evaluation_mobilenetv2.ipynb` — test-set metrics for Phase 1 and Phase 2.
- `notebooks/07_gradcam_mobilenetv2.ipynb` — Grad-CAM overlays for the final model.

The final model can also be evaluated from the CLI:
```bash
python src/evaluate.py --model_kind mobilenet --model_path saved_models/mobilenet_phase2_best.h5 --splits_dir dataset_splits --output results/
```

The baseline is evaluated with `--model_kind baseline --model_path saved_models/baseline_best.h5 --data_dir ./dataset`.

---

## Results (held-out test set, 72 images)

| Model | Test accuracy | Macro F1 | Macro AUC |
|---|---|---|---|
| Baseline CNN | 9.72% | 0.0511 | 0.7212 |
| MobileNetV2 Phase 1 (frozen base) | 81.94% | 0.8182 | 0.9870 |
| MobileNetV2 Phase 2 (fine-tuned) | **83.33%** | **0.8322** | **0.9909** |

Validation accuracy reached 88.89% (Phase 1) and 90.48% (Phase 2). The Baseline CNN trained from
scratch fails on this small dataset, which motivates the transfer-learning approach; fine-tuning the
upper base layers adds a further gain over feature extraction alone.

---

## Web Demo (Flask) — Week 8

```bash
python web_app/app.py
# Open a browser at http://127.0.0.1:5000
```

The demo exposes `mobilenet_phase2_best.h5` through a browser-based upload-and-predict interface.

---

## Reproducibility

All experiments use fixed seeds via `src.utils.set_seeds(42)`:
- `os.environ["PYTHONHASHSEED"] = "42"`
- `random.seed(42)`, `np.random.seed(42)`, `tf.random.set_seed(42)`

The train/val/test split is recorded in `results/data_split.csv` and is the single source of truth
for `dataset_splits/`.

---

## Weekly Progress

| Week | Milestone | Status |
|------|-----------|--------|
| 1 | Repository skeleton and structure | ✅ Done |
| 2 | Dataset and data pipeline         | ✅ Done |
| 3 | Baseline CNN trained              | ✅ Done |
| 4 | Baseline evaluation               | ✅ Done |
| 5 | MobileNetV2 Phase 1               | ✅ Done |
| 6 | MobileNetV2 fine-tuned            | ✅ Done |
| 7 | Full evaluation + Grad-CAM        | ✅ Done |
| 8 | Flask demo                        | ✅ Done |
| 9 | Report                            | ✅ Done |
| 10 | Final submission                 | ✅ Done |

---

## Author

- Student: Corneliu Rosca (STU141830)
- Project: Image-Based Fruit Ripeness Classification using CNNs and Transfer Learning