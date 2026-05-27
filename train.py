import pickle
import lzma
import time

from main import load_data, fit_package, predict_with_package, error_metrics, MODEL_PATH


def print_metrics(title, metrics):
    print("\n" + title)
    for key in ["mean", "rmse", "median", "p70", "p90", "p95", "max"]:
        print(f"{key:>8s}: {metrics[key]:.6f}")


def main():
    t0 = time.time()

    p, d_hat, p_bs = load_data("DH_FR1.mat")
    if p is None:
        raise RuntimeError("Training requires p labels, but p was not found in DH_FR1.mat")

    print("===== DATA =====")
    print("p:", p.shape)
    print("d_hat:", d_hat.shape)
    print("p_bs:", p_bs.shape)

    package, diagnostics = fit_package(d_hat, p, p_bs, n_splits=5, random_state=42)

    with lzma.open(MODEL_PATH, "wb", preset=6) as f:
        pickle.dump(package, f, protocol=pickle.HIGHEST_PROTOCOL)

    pred_train = predict_with_package(package, d_hat, p_bs)
    train_metrics = error_metrics(pred_train, p.T)

    print("\n===== BASE MODEL TOP 10 OOF =====")
    for row in diagnostics["base_oof_metrics"][:10]:
        print(
            f"{row['name']:24s} | {row['mode']:12s} | "
            f"mean={row['mean']:.6f} median={row['median']:.6f} "
            f"p90={row['p90']:.6f} max={row['max']:.6f}"
        )

    print_metrics("===== STACK OOF METRICS =====", diagnostics["stack_oof_metrics"])
    print_metrics("===== AVG TOP6 OOF METRICS =====", diagnostics["avg_top6_oof_metrics"])
    print_metrics("===== FINAL BLENDED OOF METRICS =====", diagnostics["final_oof_metrics"])
    print_metrics("===== FINAL FIT TRAIN METRICS 참고용 =====", train_metrics)

    print("\nTop6 models:", ", ".join(package["top6_names"]))
    print("Saved:", MODEL_PATH)
    print("Elapsed sec:", round(time.time() - t0, 2))


if __name__ == "__main__":
    main()
