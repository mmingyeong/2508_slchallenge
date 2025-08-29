

# Strong Lens Data Challenge – ML Pipeline

This repository implements a **binary classification pipeline** for the [Strong Lens Data Challenge](https://slchallenge.cbpf.br/), using **ConvNeXt V2** models to classify strong gravitational lenses from real and simulated LSST-like data.

## Features

* **Data loading**: Safe FITS reader with corrupted-file handling.
* **Training**: Mixed precision, early stopping, checkpointing.
* **Prediction**: Inference on train/val/test splits with CSV and NPY outputs.
* **Evaluation**: Accuracy, AUC, calibration, and diagnostic plots.

## Quick Start

```bash
# Train
python src/train.py --epochs 30 --save_dir ./checkpoints

# Predict
python src/predict.py --model_path ./checkpoints/best.pt --which test

# Evaluate
python src/evaluate.py --from_npy \
  --labels ./pred_outputs/labels_test.npy \
  --probs ./pred_outputs/probs_test.npy \
  --preds ./pred_outputs/preds_test.npy \
  --out_dir ./eval_outputs
```

## Next Steps

* Train with the **full dataset** and more epochs for better performance.
* Experiment with other models or hyperparameters for improved accuracy.

