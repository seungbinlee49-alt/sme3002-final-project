import os
import pickle
import lzma
import warnings

import numpy as np
import scipy.io as sio

from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

EPS = 1e-9
MODEL_PATH = "model.pkl.xz"
STACK_BLEND_ALPHA = 0.50


def load_data(mat_path=None):
    if mat_path is None:
        if os.path.exists("DH_FR1.mat"):
            mat_path = "DH_FR1.mat"
        elif os.path.exists("InF_DH_FR1.mat"):
            mat_path = "InF_DH_FR1.mat"
        else:
            raise FileNotFoundError("DH_FR1.mat or InF_DH_FR1.mat not found")

    data = sio.loadmat(mat_path, squeeze_me=False)

    if "p_bs" in data:
        p_bs = np.asarray(data["p_bs"], dtype=float)
    elif "BS_positions" in data:
        p_bs = np.asarray(data["BS_positions"], dtype=float)
    else:
        raise KeyError("Anchor position variable not found: expected p_bs or BS_positions")

    d_hat = np.asarray(data["d_hat"], dtype=float)

    p = None
    if "p" in data:
        p = np.asarray(data["p"], dtype=float)

    if p_bs.shape[0] != 2:
        p_bs = p_bs.T
    if d_hat.shape[0] != p_bs.shape[1]:
        d_hat = d_hat.T
    if p is not None and p.shape[0] != 2:
        p = p.T

    return p, d_hat, p_bs


def pairwise_diff(x):
    out = []
    m = x.shape[1]
    for i in range(m):
        for j in range(i + 1, m):
            out.append((x[:, i] - x[:, j])[:, None])
    return np.hstack(out)


def pairwise_ratio(x):
    out = []
    m = x.shape[1]
    for i in range(m):
        for j in range(i + 1, m):
            out.append((x[:, i] / (x[:, j] + EPS))[:, None])
    return np.hstack(out)


def weighted_centroid_features(d_hat, p_bs):
    x = np.maximum(d_hat.T.astype(float), 1e-6)
    anchors = p_bs.T.astype(float)

    w1 = 1.0 / x
    w2 = 1.0 / (x * x)
    row_median = np.maximum(np.median(x, axis=1, keepdims=True), 1e-6)
    w3 = np.exp(-x / row_median)

    w1 = w1 / np.sum(w1, axis=1, keepdims=True)
    w2 = w2 / np.sum(w2, axis=1, keepdims=True)
    w3 = w3 / np.sum(w3, axis=1, keepdims=True)

    return np.hstack([w1 @ anchors, w2 @ anchors, w3 @ anchors])


def fit_feature_state(d_hat):
    x = d_hat.T.astype(float)
    med = np.median(x, axis=0)
    q25 = np.percentile(x, 25, axis=0)
    q75 = np.percentile(x, 75, axis=0)
    iqr = np.where((q75 - q25) > EPS, q75 - q25, 1.0)
    return {"med": med, "iqr": iqr}


def build_features(d_hat, p_bs, mode, feature_state):
    x = d_hat.T.astype(float)

    med = feature_state["med"]
    iqr = feature_state["iqr"]

    z = (x - med[None, :]) / iqr[None, :]
    logx = np.log1p(np.maximum(x, 0.0))
    invx = 1.0 / np.maximum(x, 1e-6)

    sorted_x = np.sort(x, axis=1)
    sorted_z = np.sort(z, axis=1)
    sorted_inv = np.sort(invx, axis=1)

    rank_idx = np.argsort(x, axis=1)
    nearest = rank_idx[:, :8]
    farthest = rank_idx[:, -4:]
    anchors = p_bs.T.astype(float)

    near_features = []
    for row in range(x.shape[0]):
        vals = []
        for idx in nearest[row]:
            vals.extend([anchors[idx, 0], anchors[idx, 1], x[row, idx], z[row, idx]])
        for idx in farthest[row]:
            vals.extend([anchors[idx, 0], anchors[idx, 1], x[row, idx], z[row, idx]])
        near_features.append(vals)
    near_features = np.asarray(near_features, dtype=float)

    centroid = weighted_centroid_features(d_hat, p_bs)

    stat_features = np.hstack([
        np.mean(x, axis=1, keepdims=True),
        np.std(x, axis=1, keepdims=True),
        np.min(x, axis=1, keepdims=True),
        np.max(x, axis=1, keepdims=True),
        np.median(x, axis=1, keepdims=True),
        np.percentile(x, 25, axis=1, keepdims=True),
        np.percentile(x, 75, axis=1, keepdims=True),
    ])

    if mode == "raw_log_sort":
        feat = np.hstack([
            x, z, logx, invx,
            sorted_x, sorted_z, sorted_inv,
            centroid, near_features, stat_features
        ])
    elif mode == "diff":
        feat = np.hstack([
            x, z, logx, invx,
            sorted_x, sorted_z, sorted_inv,
            pairwise_diff(z),
            centroid, near_features, stat_features
        ])
    elif mode == "diff_ratio":
        feat = np.hstack([
            x, z, logx, invx,
            sorted_x, sorted_z, sorted_inv,
            pairwise_diff(z),
            pairwise_ratio(np.maximum(x, 1e-6)),
            centroid, near_features, stat_features
        ])
    else:
        raise ValueError("Unknown feature mode: " + str(mode))

    return np.where(np.isfinite(feat), feat, 0.0)


def make_model_specs():
    specs = []

    for mode in ["raw_log_sort", "diff", "diff_ratio"]:
        specs.append((f"{mode}_HistGBR_a", mode, MultiOutputRegressor(
            HistGradientBoostingRegressor(
                max_iter=450,
                learning_rate=0.030,
                max_leaf_nodes=15,
                l2_regularization=0.05,
                random_state=101,
            )
        )))

        specs.append((f"{mode}_HistGBR_b", mode, MultiOutputRegressor(
            HistGradientBoostingRegressor(
                max_iter=320,
                learning_rate=0.045,
                max_leaf_nodes=11,
                l2_regularization=0.15,
                random_state=102,
            )
        )))

    specs.append(("rawlog_ET_leaf1", "raw_log_sort", ExtraTreesRegressor(
        n_estimators=1000,
        min_samples_leaf=1,
        max_features=0.90,
        random_state=201,
        n_jobs=-1,
    )))

    specs.append(("rawlog_ET_leaf2", "raw_log_sort", ExtraTreesRegressor(
        n_estimators=1000,
        min_samples_leaf=2,
        max_features=0.80,
        random_state=202,
        n_jobs=-1,
    )))

    specs.append(("diff_ET_leaf1", "diff", ExtraTreesRegressor(
        n_estimators=1000,
        min_samples_leaf=1,
        max_features=0.85,
        random_state=203,
        n_jobs=-1,
    )))

    specs.append(("diff_ratio_ET_leaf1", "diff_ratio", ExtraTreesRegressor(
        n_estimators=1000,
        min_samples_leaf=1,
        max_features=0.80,
        random_state=204,
        n_jobs=-1,
    )))

    specs.append(("rawlog_RF_leaf1", "raw_log_sort", RandomForestRegressor(
        n_estimators=700,
        min_samples_leaf=1,
        max_features=0.80,
        random_state=301,
        n_jobs=-1,
    )))

    specs.append(("rawlog_KNN_9", "raw_log_sort", make_pipeline(
        RobustScaler(),
        KNeighborsRegressor(n_neighbors=9, weights="distance")
    )))

    specs.append(("diff_KNN_11", "diff", make_pipeline(
        RobustScaler(),
        KNeighborsRegressor(n_neighbors=11, weights="distance")
    )))

    specs.append(("rawlog_SVR", "raw_log_sort", make_pipeline(
        StandardScaler(),
        MultiOutputRegressor(SVR(C=60.0, gamma="scale", epsilon=0.05))
    )))

    return specs


def error_metrics(pred, y):
    e = np.linalg.norm(pred - y, axis=1)
    return {
        "mean": float(np.mean(e)),
        "rmse": float(np.sqrt(np.mean(e * e))),
        "median": float(np.median(e)),
        "p70": float(np.percentile(e, 70)),
        "p90": float(np.percentile(e, 90)),
        "p95": float(np.percentile(e, 95)),
        "max": float(np.max(e)),
    }


def fit_package(d_hat, p, p_bs, n_splits=5, random_state=42):
    y = p.T.astype(float)
    feature_state = fit_feature_state(d_hat)

    specs = make_model_specs()
    feature_modes = sorted(set(mode for _, mode, _ in specs))
    feature_cache = {
        mode: build_features(d_hat, p_bs, mode, feature_state)
        for mode in feature_modes
    }

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    oof = np.zeros((len(y), 2 * len(specs)), dtype=float)
    base_oof_metrics = []

    for mi, (name, mode, model_template) in enumerate(specs):
        print("OOF base:", name, "|", mode)
        X = feature_cache[mode]
        pred = np.zeros_like(y, dtype=float)

        for fold, (tr, va) in enumerate(kf.split(X), start=1):
            model = clone(model_template)
            model.fit(X[tr], y[tr])
            pred[va] = model.predict(X[va])
            print(" fold", fold, "done")

        oof[:, 2 * mi:2 * mi + 2] = pred
        met = error_metrics(pred, y)
        base_oof_metrics.append({"name": name, "mode": mode, **met})
        print(f" {name} mean={met['mean']:.6f} median={met['median']:.6f} p90={met['p90']:.6f}")

    base_oof_metrics_sorted = sorted(base_oof_metrics, key=lambda row: row["mean"])
    top6_names = [row["name"] for row in base_oof_metrics_sorted[:6]]
    name_to_index = {name: i for i, (name, _, _) in enumerate(specs)}
    top6_indices = [name_to_index[name] for name in top6_names]

    avg_top6_oof = np.mean([oof[:, 2*i:2*i+2] for i in top6_indices], axis=0)

    meta_model = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=np.logspace(-4, 4, 25))
    )

    stack_oof = np.zeros_like(y, dtype=float)
    for tr, va in kf.split(oof):
        meta = clone(meta_model)
        meta.fit(oof[tr], y[tr])
        stack_oof[va] = meta.predict(oof[va])

    final_oof = STACK_BLEND_ALPHA * stack_oof + (1.0 - STACK_BLEND_ALPHA) * avg_top6_oof

    print("Fit final base models on all training data")
    final_models = []
    for name, mode, model_template in specs:
        model = clone(model_template)
        model.fit(feature_cache[mode], y)
        final_models.append((name, mode, model))
        print(" fitted:", name)

    meta_model.fit(oof, y)

    package = {
        "version": "rtt_fingerprint_oof_stacking_v0_3",
        "feature_state": feature_state,
        "model_specs_names_modes": [(name, mode) for name, mode, _ in specs],
        "models": final_models,
        "meta_model": meta_model,
        "top6_names": top6_names,
        "stack_blend_alpha": STACK_BLEND_ALPHA,
        "base_oof_metrics": base_oof_metrics_sorted,
        "final_oof_metrics": error_metrics(final_oof, y),
        "p_min": np.min(y, axis=0),
        "p_max": np.max(y, axis=0),
    }

    diagnostics = {
        "base_oof_metrics": base_oof_metrics_sorted,
        "avg_top6_oof_metrics": error_metrics(avg_top6_oof, y),
        "stack_oof_metrics": error_metrics(stack_oof, y),
        "final_oof_metrics": error_metrics(final_oof, y),
    }

    return package, diagnostics


def predict_with_package(package, d_hat, p_bs):
    feature_state = package["feature_state"]

    feature_cache = {}
    for _, mode, _ in package["models"]:
        if mode not in feature_cache:
            feature_cache[mode] = build_features(d_hat, p_bs, mode, feature_state)

    base_preds = []
    name_to_pred = {}

    for name, mode, model in package["models"]:
        pred = model.predict(feature_cache[mode])
        pred = np.asarray(pred, dtype=float)
        base_preds.append(pred)
        name_to_pred[name] = pred

    base_mat = np.hstack(base_preds)
    stack_pred = package["meta_model"].predict(base_mat)

    top_preds = [name_to_pred[name] for name in package["top6_names"]]
    avg_top6 = np.mean(top_preds, axis=0)

    alpha = float(package.get("stack_blend_alpha", 0.5))
    pred = alpha * stack_pred + (1.0 - alpha) * avg_top6

    p_min = package.get("p_min", None)
    p_max = package.get("p_max", None)
    if p_min is not None and p_max is not None:
        margin = 5.0
        pred = np.minimum(np.maximum(pred, p_min[None, :] - margin), p_max[None, :] + margin)

    return pred


def fallback_predict(d_hat, p_bs):
    x = np.maximum(d_hat.T.astype(float), 1e-6)
    anchors = p_bs.T.astype(float)
    w = 1.0 / (x * x)
    w = w / np.sum(w, axis=1, keepdims=True)
    return w @ anchors


_MODEL_CACHE = None
_PRED_CACHE = None
_CALL_INDEX = 0


def get_model_package():
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        if os.path.exists(MODEL_PATH):
            with lzma.open(MODEL_PATH, "rb") as f:
                _MODEL_CACHE = pickle.load(f)
        else:
            _MODEL_CACHE = False
    return _MODEL_CACHE


def your_algorithm(d_hat_u, p_bs):
    """
    Single-user prediction function following the assignment template.
    For speed, full predictions are computed once and cached during main().
    """
    global _PRED_CACHE, _CALL_INDEX

    if _PRED_CACHE is not None:
        idx = _CALL_INDEX
        _CALL_INDEX += 1
        return np.asarray(_PRED_CACHE[:, idx], dtype=float)

    d_hat_u = np.asarray(d_hat_u, dtype=float).reshape(-1, 1)
    p_bs = np.asarray(p_bs, dtype=float)

    if p_bs.shape[0] != 2:
        p_bs = p_bs.T

    package = get_model_package()

    if package is not False:
        pred = predict_with_package(package, d_hat_u, p_bs)[0]
    else:
        pred = fallback_predict(d_hat_u, p_bs)[0]

    return np.asarray(pred, dtype=float)


def main():
    # 1) 입력 데이터 로드 — 채점기가 같은 폴더에 DH_FR1.mat 파일 자동 배치
    mat_path = "DH_FR1.mat"
    data = sio.loadmat(mat_path, squeeze_me=False)

    # 과제 설명에는 p_bs라고 되어 있으나, 제공 파일에는 BS_positions로 저장되어 있어 둘 다 지원
    if "p_bs" in data:
        p_bs = np.asarray(data["p_bs"], dtype=float)
    elif "BS_positions" in data:
        p_bs = np.asarray(data["BS_positions"], dtype=float)
    else:
        raise KeyError("p_bs or BS_positions not found in DH_FR1.mat")

    d_hat = np.asarray(data["d_hat"], dtype=float)

    # p는 채점용 GT이며, main.py 예측에는 사용하지 않음
    if "p" in data:
        p = np.asarray(data["p"], dtype=float)
    else:
        p = None

    if p_bs.shape[0] != 2:
        p_bs = p_bs.T
    if d_hat.shape[0] != p_bs.shape[1]:
        d_hat = d_hat.T

    # 2) 본인 알고리즘 — 사용자 수는 입력에서 동적으로 받기
    num_user = d_hat.shape[1]
    p_hat = np.zeros((2, num_user), dtype=float)

    # 속도 최적화: your_algorithm 호출 전에 전체 예측을 한 번만 계산해 cache에 저장
    global _PRED_CACHE, _CALL_INDEX
    _CALL_INDEX = 0

    package = get_model_package()
    if package is not False:
        _PRED_CACHE = predict_with_package(package, d_hat, p_bs).T
    else:
        _PRED_CACHE = fallback_predict(d_hat, p_bs).T

    for u in range(num_user):
        p_hat[:, u] = your_algorithm(d_hat[:, u], p_bs)

    _PRED_CACHE = None
    _CALL_INDEX = 0

    # 3) 결과 반환 — numpy 배열, 모양 (2, num_user)
    return p_hat


if __name__ == "__main__":
    p_hat = main()
    print("p_hat shape:", p_hat.shape)
    print(p_hat[:, :5])
