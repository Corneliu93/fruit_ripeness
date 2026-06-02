# Image-Based Fruit Ripeness Classification using CNNs and Transfer Learning

A deep learning project for automated fruit ripeness detection using Convolutional Neural Networks
and MobileNetV2 transfer learning. Classifies **9 classes** across 3 fruit types (apple, banana, pear)
× 3 ripeness stages (unripe, ripe, overripe).

---

## Project Structure

```
fruit_ripeness/
├── dataset/
│   ├── apple/        {unripe, ripe, overripe}
│   ├── banana/       {unripe, ripe, overripe}
│   ├── pear/         {unripe, ripe, overripe}
│   ├── metadata.csv
│   └── capture_instructions.md
├── dataset_splits/
│   ├── train/  → 9 clase, 315 imagini
│   ├── val/    →  9 clase,  63 imagini
│   └── test/   →  9 clase,  72 imagini
├── src/
│   ├── data_pipeline.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── notebooks/
│   ├── results/data_split.csv
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_train_baseline.ipynb
│   ├── 03_evaluation_baseline.ipynb
│   └── 04_train_mobilenetv2_phase1.ipynb
├── saved_models/
│   ├── baseline_best.h5
│   ├── mobilenet_phase1_best.h5
│   └── mobilenet_finetuned.h5
├── results/
│   ├── results.csv
│   ├── data_split.csv
│   ├── history_baseline.csv
│   ├── augmentation_demo.png
│   ├── class_names.json
│   ├── classification_report_baseline.txt
│   ├── confusion_matrix_baseline.png
│   ├── mobilenet_phase1_history.json
│   ├── mobilenet_phase1_training_curves.png
│   ├── preprocessing_demo.png
│   ├── raw_samples.png
│   ├── roc_curves_baseline.png
│   ├── training_batch_sample.png
│   ├── training_curves_baseline.png
│   ├── 
│   └── gradcam_{example}.png
├── web_app/
│   ├── app.py
│   ├── templates/index.html
│   ├── static/uploads/
│   └── model/mobilenet_finetuned.h5
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
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Dataset

Target: **50 images per class** (minimum 25), 9 classes = 450 images total.

See `dataset/capture_instructions.md` for photo capture protocol.

---

## Training

### Baseline CNN
```bash
python src/train.py --model baseline --data_dir ./dataset --batch_size 32 --epochs 50 --seed 42
```

### MobileNetV2 Phase 1 (frozen base)
```bash
python src/train.py --model mobilenet --phase phase1 --data_dir ./dataset --batch_size 32 --epochs 20
```

### MobileNetV2 Fine-tuning (unfreeze top 20%)
```bash
python src/train.py --model mobilenet --phase finetune --data_dir ./dataset --batch_size 32 --epochs 20
```

---

## Evaluation
```bash
python src/evaluate.py --model_path saved_models/mobilenet_finetuned.h5 --data_dir ./dataset/test --output results/
```

---

## Web Demo (Flask)
```bash
python web_app/app.py
# Open browser at http://127.0.0.1:5000
```

---

## Dataset Validation
```bash
python dataset_check.py
```

---

## Reproducibility

All experiments use fixed seeds:
- `np.random.seed(42)`
- `tf.random.set_seed(42)`

---

## ⚠️ Dataset Limitation (Fallback)

If real images cannot be captured, the pipeline trains on:
1. Synthetic augmented images (generated via `synthesize_augment.py`)
2. Public subset (≤10 images of a public fruit dataset)


---

## Weekly Progress

| Week | Milestone | Status |
|------|-----------|--------|
| 1 | Repo skeleton, structure | ✅ Done |
| 2 | Dataset + data pipeline | ✅ Done |
| 3 | Baseline CNN trained | ✅ Done |
| 4 | Baseline evaluation | ✅ Done |
| 5 | MobileNetV2 Phase 1 | ✅ Done |
| 6 | MobileNetV2 Fine-tuned | ⏳ Pending |
| 7 | Full evaluation + Grad-CAM | ⏳ Pending |
| 8 | Flask demo | ⏳ Pending |
| 9 | Report + slides | ⏳ Pending |
| 10 | Final submission | ⏳ Pending |

---

## Git Commit Convention
```
init project
add dataset structure
add data pipeline
baseline model
train baseline
add mobilenet transfer learning
mobilenet phase1 trained
mobilenet finetuned
add evaluation and gradcam
add web demo
final report and slides
```

---

## Author
- Student: Corneliu Rosca
- Project: Image-based fruit ripeness classification