"""
Training script for the SME3002 final project.

This file is intentionally written as the training-side description of the
submitted machine-learning algorithm.

Important submission note:
- main.py is the hidden-test inference entry point.
- train.py is the reproducibility / novelty / consistency entry point.
- The final model used by main.py is saved as model.pkl.xz.
- The core implementation is shared with main.py so that training and inference
  use exactly the same feature construction, model definitions, and prediction
  logic.

Algorithm summary:
1. Load DH_FR1.mat.
   The training data contain:
   - p:       ground-truth user positions, shape (2, N)
   - d_hat:   RTT observations from 18 base stations, shape (18, N)
   - p_bs or BS_positions: base-station coordinates, shape (2, 18)

2. Interpret the 18 RTT values as a fingerprint vector rather than as exact
   geometric distances. This is important because RTT measurements can include
   anchor-dependent bias, non-line-of-sight effects, multipath distortion, and
   nonlinear scale differences.

3. Build multiple feature views:
   - raw_log_sort:
       raw RTT, robustly normalized RTT, log-transformed RTT, inverse RTT,
       sorted RTT, sorted normalized RTT, sorted inverse RTT, RTT-weighted
       centroid features, nearest/farthest anchor coordinate features, and
       row-wise statistical features.
   - diff:
       all raw_log_sort information plus pairwise differences between normalized
       RTT values. This emphasizes relative anchor-to-anchor measurement
       structure.
   - diff_ratio:
       all diff information plus pairwise RTT ratios. This emphasizes scale-
       relative fingerprint structure.

4. Train multiple base regressors on different feature views:
   - HistGradientBoostingRegressor variants
   - ExtraTreesRegressor variants
   - RandomForestRegressor
   - KNN fingerprint baselines
   - SVR baseline

5. Evaluate base models using 5-fold out-of-fold validation.
   Each training sample is predicted only by a model that was not fitted using
   that sample. This is used to reduce train-set overfitting and to approximate
   hidden-test generalization.

6. Build a stacking representation from OOF base predictions.
   The meta input for one sample is the concatenation of all base-model OOF
   coordinate predictions.

7. Train a RidgeCV stacking model on OOF predictions.
   Ridge regularization is used so that the final meta-model does not over-rely
   on one base model or memorize the training set.

8. Select the top six base models by OOF mean position error and compute their
   average prediction.

9. Final OOF prediction is a conservative blend:
       final = 0.5 * RidgeCV_stack + 0.5 * top6_average

10. After OOF validation, fit every base model on all 700 labeled users and save
    a single compressed package to model.pkl.xz. main.py loads this package and
    performs only inference on the hidden test set.

The script does not add any package outside the standard environment. It uses
numpy, scipy, and scikit-learn, which are allowed in the project environment.
"""

import argparse
import lzma
import pickle
import time
from pathlib import Path

import numpy as np

from main import (
    MODEL_PATH,
    STACK_BLEND_ALPHA,
    build_features,
    error_metrics,
    fit_feature_state,
    fit_package,
    load_data,
    make_model_specs,
    predict_with_package,
)


FEATURE_VIEWS = {
    "raw_log_sort": [
        "raw RTT",
        "robust normalized RTT",
        "log(1 + RTT)",
        "inverse RTT",
        "sorted RTT",
        "sorted normalized RTT",
        "sorted inverse RTT",
        "RTT-weighted centroid",
        "nearest/farthest anchor coordinate features",
        "row-wise statistics",
    ],
    "diff": [
        "raw_log_sort feature view",
        "pairwise normalized RTT differences",
    ],
    "diff_ratio": [
        "diff feature view",
        "pairwise RTT ratios",
    ],
}


def print_section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def print_metrics(title, metrics):
    print_section(title)
    for key in ["mean", "rmse", "median", "p70", "p90", "p95", "max"]:
        print(f"{key:>8s}: {metrics[key]:.6f}")


def validate_dataset_shapes(p, d_hat, p_bs):
    """Check that the training data follow the expected project convention."""
    if p is None:
        raise RuntimeError("Training requires p labels, but p was not found in DH_FR1.mat")

    if p.shape[0] != 2:
        raise ValueError(f"p must have shape (2, N), but got {p.shape}")

    if p_bs.shape[0] != 2:
        raise ValueError(f"p_bs / BS_positions must have shape (2, 18), but got {p_bs.shape}")

    if d_hat.shape[0] != p_bs.shape[1]:
        raise ValueError(
            "d_hat first dimension must match the number of base stations: "
            f"d_hat={d_hat.shape}, p_bs={p_bs.shape}"
        )

    if d_hat.shape[1] != p.shape[1]:
        raise ValueError(
            "d_hat and p must contain the same number of users: "
            f"d_hat={d_hat.shape}, p={p.shape}"
        )


def describe_feature_views(d_hat, p_bs):
    """Print the exact feature-view dimensions used by the training pipeline."""
    feature_state = fit_feature_state(d_hat)

    print_section("FEATURE VIEWS")
    for mode, explanation in FEATURE_VIEWS.items():
        X = build_features(d_hat, p_bs, mode, feature_state)
        print(f"{mode:14s}: shape={X.shape}")
        for item in explanation:
            print(f"  - {item}")


def describe_model_zoo():
    """Print the base regressors and feature modes before training."""
    specs = make_model_specs()

    print_section("BASE MODEL ZOO")
    print(f"number of base models: {len(specs)}")
    for idx, (name, mode, model) in enumerate(specs, start=1):
        print(f"{idx:02d}. {name:24s} | feature={mode:12s} | model={model.__class__.__name__}")

    return specs


def describe_training_strategy(n_users):
    print_section("TRAINING / VALIDATION STRATEGY")
    print(f"number of labeled users: {n_users}")
    print("validation method      : 5-fold out-of-fold validation")
    print("base prediction target : 2D coordinate regression, [x, y]")
    print("meta model             : RidgeCV on OOF base predictions")
    print(f"final blend            : {STACK_BLEND_ALPHA:.2f} * stack + {1.0 - STACK_BLEND_ALPHA:.2f} * top6 average")
    print("model selection basis  : OOF mean error with RMSE / median / P90 / P95 / max checked together")
    print("saved model file       :", MODEL_PATH)


def save_package(package, output_path):
    output_path = Path(output_path)
    with lzma.open(output_path, "wb", preset=6) as f:
        pickle.dump(package, f, protocol=pickle.HIGHEST_PROTOCOL)
    return output_path.stat().st_size


def train_and_save(mat_path, output_path):
    t0 = time.time()

    p, d_hat, p_bs = load_data(mat_path)
    validate_dataset_shapes(p, d_hat, p_bs)

    print_section("DATA")
    print("p shape     :", p.shape)
    print("d_hat shape :", d_hat.shape)
    print("p_bs shape  :", p_bs.shape)

    describe_feature_views(d_hat, p_bs)
    describe_model_zoo()
    describe_training_strategy(n_users=d_hat.shape[1])

    package, diagnostics = fit_package(d_hat, p, p_bs, n_splits=5, random_state=42)

    saved_bytes = save_package(package, output_path)

    pred_train = predict_with_package(package, d_hat, p_bs)
    train_metrics = error_metrics(pred_train, p.T)

    print_section("BASE MODEL TOP 10 OOF")
    for row in diagnostics["base_oof_metrics"][:10]:
        print(
            f"{row['name']:24s} | {row['mode']:12s} | "
            f"mean={row['mean']:.6f} rmse={row['rmse']:.6f} "
            f"median={row['median']:.6f} p90={row['p90']:.6f} "
            f"p95={row['p95']:.6f} max={row['max']:.6f}"
        )

    print_metrics("STACK OOF METRICS", diagnostics["stack_oof_metrics"])
    print_metrics("AVG TOP6 OOF METRICS", diagnostics["avg_top6_oof_metrics"])
    print_metrics("FINAL BLENDED OOF METRICS", diagnostics["final_oof_metrics"])
    print_metrics("FINAL FIT TRAIN METRICS - reference only, not used for model selection", train_metrics)

    print_section("FINAL PACKAGE")
    print("top6 models :", ", ".join(package["top6_names"]))
    print("saved path  :", output_path)
    print("saved size  :", f"{saved_bytes:,} bytes")
    print("elapsed sec :", round(time.time() - t0, 2))

    return package, diagnostics


def main():
    parser = argparse.ArgumentParser(
        description="Train the RTT fingerprint OOF stacking model and save model.pkl.xz."
    )
    parser.add_argument(
        "--mat",
        default="DH_FR1.mat",
        help="Training MAT file. Default: DH_FR1.mat",
    )
    parser.add_argument(
        "--output",
        default=MODEL_PATH,
        help=f"Output model package path. Default: {MODEL_PATH}",
    )
    args = parser.parse_args()

    train_and_save(args.mat, args.output)


if __name__ == "__main__":
    main()
