# irace_train_eval.py
from __future__ import annotations

import argparse
import json
import math
import sys
import traceback

from dtcalib.iterated_racing.train import run_training_job_fixed_epochs 

PENALTY = 1e100  # coût renvoyé en cas de crash/OOM (irace doit recevoir un nombre)

def main() -> int:
    p = argparse.ArgumentParser()

    # Données / split / compute
    p.add_argument("--root-dir", type=str, required=True)
    p.add_argument("--split-json", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")

    # Paramètres passés par irace
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, required=True)

    p.add_argument("--batch_size", type=int, required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--optimizer", type=str, required=True)

    p.add_argument("--scheduler", type=str, default="plateau")     # "none" / "plateau"
    p.add_argument("--scheduler_factor", type=float, default=0.5)
    p.add_argument("--scheduler_patience", type=int, default=5)

    args = p.parse_args()

    # Construire le dict params attendu par run_training_job_fixed_epochs()
    params = {
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "optimizer": args.optimizer,
    }

    # Si scheduler désactivé: on met des valeurs neutres (ou on ignore)
    if args.scheduler == "none":
        params["scheduler_factor"] = 0.5
        params["scheduler_patience"] = 10**9  # ne déclenche jamais vraiment
    else:
        params["scheduler_factor"] = args.scheduler_factor
        params["scheduler_patience"] = args.scheduler_patience

    params["target_transform"] = "logC"

    try:
        out = run_training_job_fixed_epochs(
            root_dir=args.root_dir,
            split_json_path=args.split_json,
            params=params,
            seed=args.seed,
            epochs=args.epochs,
            device=args.device,
        )
        cost = float(out["val_loss"])  # objectif: minimiser la val_loss 
        if not math.isfinite(cost):
            print(PENALTY)
            return 0
        print(cost)
        return 0

    except Exception:
        # Pour debug: log sur stderr, mais renvoyer une pénalité sur stdout
        traceback.print_exc(file=sys.stderr)
        print(PENALTY)
        return 0

if __name__ == "__main__":
    raise SystemExit(main())