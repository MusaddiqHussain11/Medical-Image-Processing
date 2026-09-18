[README.md](https://github.com/user-attachments/files/32379388/README.md)
# ASAM-CNN: A Lightweight Multi-Component Adaptive Self-Attention Network for Tympanic Membrane Disease Classification from Otoscopic Images

Official implementation of **ASAM-CNN**, rebuilt from the `Zenodo.rar` export
(24 files: `Complete Model.txt` + 23 individual visualization scripts) into a
clean, modular, reproducible repository. Every function is a faithful,
line-for-line port of the corresponding original script — nothing about the
model, training, or plotting logic was changed, only reorganized into
reusable functions.

## Dataset

880 otoscopic images, 4 classes (Chronic Otitis Media, Earwax Plug,
Myringosclerosis, Normal Tympanic Membrane), in `Training-validation/` +
`Testing/` subfolders: https://doi.org/10.6084/m9.figshare.11886630

## Results Summary

| Metric | Value |
|---|---|
| Mean 5-fold CV accuracy | 99.31% ± 0.44% |
| Ensemble test accuracy | 97.50% |
| Macro F1-score | 0.9751 |
| Macro-averaged AUC | 0.9973 |
| Parameters | 2.5M |

## Setup

```bash
pip install -r requirements.txt
```

## Repository Structure

```
├── src/
│   ├── model.py             # ASAM-CNN architecture (AdaptiveScaleLayer,
│   │                         # DynamicTemperatureLayer, adaptive_self_attention,
│   │                         # channel_attention, build_model)
│   ├── preprocess.py        # circular_crop, apply_clahe_rgb, preprocess
│   ├── train.py             # dataset loading + 5-fold stratified CV
│   ├── evaluate.py          # F1-weighted ensemble + TTA evaluation
│   ├── visualize_core.py    # dataset/training/eval plots (11 visualizations)
│   ├── interpretability.py  # Grad-CAM, attention maps, ASAM-vs-Grad-CAM, t-SNE (6)
│   ├── asam_internals.py    # ASAM-internal-mechanism plots (5) -- see note below
│   └── utils.py             # seeding, GPU config, dataset auto-discovery, figure autosave
├── configs/config.yaml
├── main.py                  # runs the full pipeline, all 24 visualizations
├── requirements.txt
├── CITATION.cff
└── .zenodo.json
```

## Reproducing Everything

```bash
python main.py --config configs/config.yaml
```

Or run stages individually:
```bash
python -m src.train --config configs/config.yaml      # training only
python -m src.evaluate --config configs/config.yaml    # train + evaluate
```

`configs/config.yaml` defaults to a Kaggle-style layout (`/kaggle/input/...`,
`/kaggle/working/...`). Edit `paths.input_root` / `paths.checkpoint_dir` /
`paths.figures_dir` to run elsewhere.

## All 24 Visualizations, Mapped to Source Files

| # | Original Zenodo.rar file | Function |
|---|---|---|
| 1 | SAMPLE TYMPANIC MEMBRANE IMAGES | `visualize_core.plot_sample_images` |
| 2 | PREPROCESSING PIPELINE VISUALIZATION | `visualize_core.plot_preprocessing_pipeline` |
| 3 | DATA AUGMENTATION EFFECT | `visualize_core.plot_augmentation_effects` |
| 4 | DATASET CLASS DISTRIBUTION | `visualize_core.plot_class_distribution` |
| 5 | 5-FOLD CROSS VALIDATION ACCURACY | `visualize_core.plot_cv_bar` |
| 6 | 5-FOLD CROSS VALIDATION F1 SCORE | `visualize_core.plot_cv_bar` |
| 7 | LEARNING CURVES (per fold) | `visualize_core.plot_learning_curves` |
| 8 | CONFUSION MATRIX | `visualize_core.plot_confusion_matrix` |
| 9 | PRECISION RECALL F1 PER CLASS | `visualize_core.plot_precision_recall_f1` |
| 10 | ROC CURVES (MULTI-CLASS) | `visualize_core.plot_roc_curves` |
| 11 | MISCLASSIFIED TYMPANIC MEMBRANE IMAGES | `visualize_core.plot_misclassified_images` |
| 12 | MODEL ACCURACY SUMMARY | `visualize_core.plot_model_accuracy_summary` |
| 13 | MODEL PARAMETERS OVERALL SUMMARY | `visualize_core.plot_model_parameters_detailed` |
| 14 | GRAD-CAM LESION LOCALIZATION | `interpretability.plot_gradcam_lesion_localization` |
| 15 | ATTENTION MAP VISUALIZATION | `interpretability.plot_attention_map_visualization` |
| 16 | ATTENTION HEATMAP COMPARISON | `interpretability.plot_attention_heatmap_comparison` |
| 17 | CLASS ACTIVATION COMPARISON | `interpretability.plot_class_activation_comparison` |
| 18 | ASAM vs Grad-CAM Agreement Summary | `interpretability.plot_asam_vs_gradcam_agreement` |
| 19 | T-SNE FEATURE EMBEDDING | `interpretability.plot_tsne` |
| 20 | ADAPTIVE SCALE LAYER — Heatmap per class | `asam_internals.plot_adaptive_scale_heatmap` |
| 21 | DYNAMIC TEMPERATURE LAYER — Value per class | `asam_internals.plot_dynamic_temperature_value` |
| 22 | DUAL-PATH ALPHA BLENDING — per class | `asam_internals.plot_dual_path_alpha_blending` |
| 23 | ASAM OUTPUT ATTENTION MAP per class | `asam_internals.plot_asam_output_attention_map` |
| 24 | LOCAL PATH vs GLOBAL PATH COMPARISON | `asam_internals.plot_local_vs_global_path` |

## ⚠️ Two Reconstructed Pieces — Please Verify

Everything above is a faithful port. Two small pieces referenced by your
original files were **not present** in any of the 24 exported `.txt` files,
so they were reconstructed by inference rather than copied:

1. **`make_gradcam_heatmap_fixed`** (used by `CLASS ACTIVATION COMPARISON`).
   Reconstructed in `interpretability.py` using the exact technique the
   script's own inline comments describe ("Absolute gradients — flat heatmap
   fix", "Robust normalization" via 5th/95th percentile clipping).

2. **The setup cell for the 5 ASAM-internals scripts** (`ADAPTIVE SCALE
   LAYER`, `DYNAMIC TEMPERATURE LAYER`, `DUAL-PATH ALPHA BLENDING`, `ASAM
   OUTPUT ATTENTION MAP`, `LOCAL PATH vs GLOBAL PATH COMPARISON`). These all
   use `model`, `class_images`, `VIZ_DIR`, `CLASS_COLORS`, and pre-discovered
   layer-name lists (`adaptive_scale_layers`, `dynamic_temp_layers`,
   `dense_sigmoid_layers`, `add_layers`, `depthwise_layers`) that are never
   defined in the archive — meaning the notebook that generated it had one
   more cell that wasn't exported. Reconstructed in
   `asam_internals.setup_asam_internals()`, inferred from how each variable
   is used and cross-checked against `model.py`'s actual layer graph. See
   the docstring at the top of `asam_internals.py` for the full reasoning.

**If you still have the original notebook**, please check these two pieces
against it and let me know if anything needs correcting — everything else
in this repository is copied faithfully from your `Zenodo.rar`.

## Citation

See `CITATION.cff`.

## License

MIT License (update to match your chosen license).
