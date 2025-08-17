import argparse
import numpy as np

from .estimator import NBMF

def _load_array(path):
    if path.endswith(".npz"):
        return np.load(path)["arr_0"]
    elif path.endswith(".npy"):
        return np.load(path)
    else:
        raise ValueError(f"Unsupported input format: {path}")

def main():
    p = argparse.ArgumentParser(prog="nbmf-mm", description="NBMF-MM CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    fit = sub.add_parser("fit", help="Fit NBMF-MM")
    fit.add_argument("--input", required=True, help="Path to X (.npz/.npy)")
    fit.add_argument("--mask", default=None, help="Optional mask (.npz/.npy)")
    fit.add_argument("--rank", type=int, required=True, help="n_components (K)")
    fit.add_argument("--orientation", choices=["dir-beta","beta-dir"], default="dir-beta")
    fit.add_argument("--alpha", type=float, default=1.2)
    fit.add_argument("--beta", type=float, default=1.2)
    fit.add_argument("--max-iter", type=int, default=2000)
    fit.add_argument("--tol", type=float, default=1e-6)
    fit.add_argument("--seed", type=int, default=None)
    fit.add_argument("--n-init", type=int, default=1)
    fit.add_argument("--out", required=True, help="Output .npz file")
    fit.add_argument("--no-numexpr", action="store_true")
    fit.add_argument("--no-numba", action="store_true")
    fit.add_argument("--verbose", type=int, default=0)

    args = p.parse_args()

    if args.cmd == "fit":
        X = _load_array(args.input)
        mask = None if args.mask is None else _load_array(args.mask)

        model = NBMF(
            n_components=args.rank,
            orientation=args.orientation,
            alpha=args.alpha,
            beta=args.beta,
            max_iter=args.max_iter,
            tol=args.tol,
            n_init=args.n_init,
            random_state=args.seed,
            use_numexpr=not args.no_numexpr,
            use_numba=not args.no_numba,
            verbose=args.verbose,
        ).fit(X, mask=mask)

        W = model.W_
        H = model.components_
        Xhat = model.inverse_transform(W)

        np.savez_compressed(
            args.out,
            W=W, H=H, Xhat=Xhat,
            objective_history=np.asarray(model.objective_history_, dtype=float),
            n_iter=np.asarray(model.n_iter_, dtype=int),
        )

if __name__ == "__main__":
    main()
