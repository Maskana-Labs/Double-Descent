"""Experiment 02: double descent with random ReLU features.

Fit a noisy linear teacher with p random ReLU features and n training points.
For each model size p we fit the weights by (ridge) least squares and measure
error on fresh test inputs. With no regularisation (the minimum-norm solution,
which is what gradient descent from zero converges to) test error is expected
to spike near p = n, the interpolation threshold, then fall again as p grows.
A little ridge regularisation should flatten the spike.

Usage:
    python experiment.py                  # default run, writes ./results
    python experiment.py --noise 1.0 --n-train 100 --out results/noisy
"""

import argparse
import csv
import os

import numpy as np


def make_features(X, W):
    """Random ReLU features, scaled by 1/sqrt(p) so a fixed ridge strength is comparable across p."""
    return np.maximum(X @ W, 0.0) / np.sqrt(W.shape[1])


def fit_ridge_svd(Phi, y, lam):
    """Weights minimising ||Phi w - y||^2 + lam ||w||^2 via the SVD.

    lam = 0 gives the minimum-norm least-squares solution (tiny singular values dropped).
    """
    U, s, Vt = np.linalg.svd(Phi, full_matrices=False)
    if lam == 0:
        cutoff = max(Phi.shape) * np.finfo(float).eps * s.max()
        filt = np.where(s > cutoff, 1.0 / np.where(s > cutoff, s, 1.0), 0.0)
    else:
        filt = s / (s**2 + lam)
    return Vt.T @ (filt * (U.T @ y))


def one_trial(p, lams, n_train, n_test, d, noise, rng):
    """One random dataset and one random feature draw; returns {lam: (test_mse, train_mse)}."""
    beta = rng.standard_normal(d) / np.sqrt(d)  # teacher: y = x . beta + noise
    X = rng.standard_normal((n_train + n_test, d))
    clean = X @ beta
    y_train = clean[:n_train] + noise * rng.standard_normal(n_train)
    W = rng.standard_normal((d, p)) / np.sqrt(d)
    Phi = make_features(X, W)
    Phi_tr, Phi_te = Phi[:n_train], Phi[n_train:]
    out = {}
    for lam in lams:
        w = fit_ridge_svd(Phi_tr, y_train, lam)
        test_mse = float(np.mean((Phi_te @ w - clean[n_train:]) ** 2))  # error vs the noise-free target
        train_mse = float(np.mean((Phi_tr @ w - y_train) ** 2))
        out[lam] = (test_mse, train_mse)
    return out


def model_sizes(n_train, p_max):
    grid = set(np.unique(np.round(np.geomspace(1, p_max, 45)).astype(int)).tolist())
    grid.update(range(max(1, n_train - 4), n_train + 5))  # dense around the threshold
    return sorted(q for q in grid if 1 <= q <= p_max)


def sweep(n_train=60, n_test=2000, d=10, noise=0.5, p_max=600, seeds=50, lams=(0.0, 1e-3, 1e-1), seed=0):
    rows = []
    for ip, p in enumerate(model_sizes(n_train, p_max)):
        tests = {lam: [] for lam in lams}
        trains = {lam: [] for lam in lams}
        for s in range(seeds):
            rng = np.random.default_rng([seed, ip, s])
            res = one_trial(p, lams, n_train, n_test, d, noise, rng)
            for lam in lams:
                tests[lam].append(res[lam][0])
                trains[lam].append(res[lam][1])
        for lam in lams:
            t = np.array(tests[lam])
            rows.append(dict(
                p=p,
                p_over_n=p / n_train,
                lam=lam,
                test_mse_median=float(np.median(t)),
                test_mse_q25=float(np.quantile(t, 0.25)),
                test_mse_q75=float(np.quantile(t, 0.75)),
                train_mse_median=float(np.median(trains[lam])),
            ))
    return rows


def write_csv(rows, path):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def plot(rows, path, n_train, noise, seeds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lams = sorted({r["lam"] for r in rows})
    colors = ["#D55E00", "#0072B2", "#009E73", "#CC79A7", "#E69F00"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for c, lam in zip(colors, lams):
        sel = [r for r in rows if r["lam"] == lam]
        x = [r["p_over_n"] for r in sel]
        ax.plot(x, [r["test_mse_median"] for r in sel], color=c, label=("min-norm (lambda=0)" if lam == 0 else f"ridge lambda={lam:g}"))
        ax.fill_between(x, [r["test_mse_q25"] for r in sel], [r["test_mse_q75"] for r in sel], color=c, alpha=0.15)
    ax.axvline(1.0, color="k", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("model size / training points  (p / n)")
    ax.set_ylabel("test error vs noise-free target")
    ax.set_title("Test error (median, IQR band)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    sel = [r for r in rows if r["lam"] == 0.0] or [r for r in rows if r["lam"] == lams[0]]
    ax.plot([r["p_over_n"] for r in sel], [r["train_mse_median"] for r in sel], color="#D55E00")
    ax.axvline(1.0, color="k", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("model size / training points  (p / n)")
    ax.set_ylabel("training error")
    ax.set_title("Training error of the same fit", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    fig.suptitle(f"Random ReLU features, n = {n_train}, label noise sd = {noise}, {seeds} trials per point")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def parse_floats(s):
    return tuple(float(x) for x in s.split(","))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-train", type=int, default=60)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--d", type=int, default=10, help="input dimension")
    ap.add_argument("--noise", type=float, default=0.5, help="label noise std dev")
    ap.add_argument("--p-max", type=int, default=600, help="largest number of features")
    ap.add_argument("--seeds", type=int, default=50, help="random trials per model size")
    ap.add_argument("--lams", type=parse_floats, default=(0.0, 1e-3, 1e-1), help="comma-separated ridge strengths")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rows = sweep(args.n_train, args.n_test, args.d, args.noise, args.p_max, args.seeds, args.lams, args.seed)
    write_csv(rows, os.path.join(args.out, "summary.csv"))
    plot(rows, os.path.join(args.out, "double_descent.png"), args.n_train, args.noise, args.seeds)

    print(f"n_train={args.n_train} d={args.d} noise={args.noise} seeds={args.seeds}\n")
    print(f"{'lambda':>8}{'peak p/n':>10}{'peak test err':>15}{'err at p=10n':>14}{'best p/n':>10}")
    for lam in args.lams:
        sel = [r for r in rows if r["lam"] == lam]
        peak = max(sel, key=lambda r: r["test_mse_median"])
        far = min(sel, key=lambda r: abs(r["p_over_n"] - 10))
        best = min(sel, key=lambda r: r["test_mse_median"])
        print(f"{lam:>8g}{peak['p_over_n']:>10.2f}{peak['test_mse_median']:>15.4g}{far['test_mse_median']:>14.4g}{best['p_over_n']:>10.2f}")
    print(f"\nWrote summary.csv and double_descent.png to {args.out}/")


if __name__ == "__main__":
    main()
