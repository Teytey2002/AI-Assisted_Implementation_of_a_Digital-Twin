"""
Runner officiel pour lancer l'optimisation hyperparamètres via Iterated Racing.

Objectif :
- Exécuter Iterated Racing (epochs fixes, pas d'early stopping)
- Tester des configurations d'hyperparamètres
- Logger toutes les évaluations (CSV)
- Retourner les meilleures configs pour ensuite entraîner "proprement" avec deep_learning/train.py

Ce fichier orchestre simplement la pipeline "iterated_racing/".

Commande d'exemple (à adapter) : (en se mettant ici dans iterated_racing/)
python3 run.py \
  --root-dir ../../../data/ALL_LP_DATASETS_CSV_Deep_learning \
  --split-json ../deep_learning/splits/rc_split_seed42_70_15_15.json \
  --cache
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dtcalib.iterated_racing.search_space import SearchSpace
from dtcalib.iterated_racing.sampler import Sampler
from dtcalib.iterated_racing.logger import RaceLogger
from dtcalib.iterated_racing.iterated_racing import IteratedRacing, RacingSettings
from dtcalib.iterated_racing.evaluator import Evaluator
from dtcalib.iterated_racing.train import run_training_job_fixed_epochs


# =============================================================================
# Evaluator projet : c'est ici qu'on branche le training réel.
# =============================================================================
class ProjectEvaluator(Evaluator):
    """
    Bridge entre Iterated Racing et le code deep learning.

    Rôle :
    - Reçoit params/seed/epochs depuis Iterated Racing
    - Lance un training court (epochs FIXES, pas d'early stopping)
    - Retourne {"val_loss": ..., "train_loss": ...}

    Implémentation :
    - Utilise run_training_job_fixed_epochs() dans iterated_racing/train_api.py
    - Utilise le split fixe JSON (train/val/test) généré une seule fois
    """
    def __init__(self, root_dir: str | Path, split_json_path: str | Path, **kwargs):
        super().__init__(**kwargs)
        self.root_dir = Path(root_dir)
        self.split_json_path = Path(split_json_path)

    def run_training_job(self, params, seed: int, epochs: int):
        return run_training_job_fixed_epochs(
            root_dir=self.root_dir,
            split_json_path=self.split_json_path,
            params=params,
            seed=seed,
            epochs=epochs,
            device=self.device,
        )


# =============================================================================
# Main runner
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Iterated Racing runner (dtcalib/iterated_racing)."
    )

    # Données
    p.add_argument(
        "--root-dir",
        type=str,
        required=True,
        help="Chemin vers le dossier dataset (ex: ../../../data/ALL_LP_DATASETS_CSV_Deep_learning).",
    )

    # Split fixe (train/val/test) généré une fois pour toutes (ex: src/dtcalib/deep_learning/splits/rc_split_seed42_70_15_15.json)
    p.add_argument(
        "--split-json",
        type=str,
        default="src/dtcalib/deep_learning/splits/rc_split_seed42_70_15_15.json",
        help="Chemin vers le JSON de split fixe train/val/test.",
    )

    # Sorties
    p.add_argument(
        "--out-dir",
        type=str,
        default="results/racing",
        help="Dossier où écrire les CSV de racing (evals + decisions).",
    )

    # Compute
    p.add_argument("--device", type=str, default="cuda", help="cuda ou cpu")
    p.add_argument("--seed", type=int, default=42, help="Seed fixe pour tout le racing")

    # Racing parameters (budget / taille)
    p.add_argument("--n-init", type=int, default=20, help="Taille population initiale")
    p.add_argument("--max-iter", type=int, default=5, help="Nombre d'itérations de racing")
    p.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[50, 75, 100, 150, 200],
        help="Liste des epochs par itération (longueur = max-iter).",
    )
    p.add_argument(
        "--keep-top-k",
        type=int,
        nargs="+",
        default=[10, 8, 6, 5, 5],
        help="Nombre de configs à garder après élimination (longueur = max-iter).",
    )
    p.add_argument(
        "--n-new",
        type=int,
        default=10,
        help="Nombre de nouvelles configs générées à chaque itération (resampling).",
    )

    # Statistiques
    p.add_argument("--alpha", type=float, default=0.05, help="Seuil Friedman (p-value)")

    # Divers
    p.add_argument(
        "--cache",
        action="store_true",
        help="Active un cache (évite de recalculer une config identique).",
    )

    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Sanity checks
    if len(args.budgets) != args.max_iter:
        raise ValueError("--budgets doit avoir exactement max-iter valeurs.")
    if len(args.keep_top_k) != args.max_iter:
        raise ValueError("--keep-top-k doit avoir exactement max-iter valeurs.")

    root_dir = Path(args.root_dir)
    out_dir = Path(args.out_dir)

    # -------------------------------------------------------------------------
    # 1) SearchSpace : définit l'espace d'hyperparamètres autorisés
    #    -> utilisé par Sampler pour générer des configurations
    # -------------------------------------------------------------------------
    space = SearchSpace()

    # -------------------------------------------------------------------------
    # 2) Sampler : génère
    #    - population initiale (log-uniform + choix discrets)
    #    - nouvelles configs autour des survivants (adaptive_resample)
    # -------------------------------------------------------------------------
    sampler = Sampler(space, seed=args.seed)

    # -------------------------------------------------------------------------
    # 3) Logger : écrit 2 fichiers CSV :
    #    - racing_evals.csv     (1 ligne par config évaluée)
    #    - racing_decisions.csv (1 ligne par itération avec p-value + survivants)
    # -------------------------------------------------------------------------
    logger = RaceLogger(out_dir=str(out_dir))

    # -------------------------------------------------------------------------
    # 4) Evaluator : pont vers ton training.
    #    Il doit fournir run_training_job(params, seed, epochs) -> {"val_loss": ...}
    # -------------------------------------------------------------------------
    evaluator = ProjectEvaluator(
        root_dir=root_dir,
        split_json_path=args.split_json,
        device=args.device,
        cache=args.cache,
    )

    # -------------------------------------------------------------------------
    # 5) Settings : paramètres globaux du racing
    # -------------------------------------------------------------------------
    settings = RacingSettings(
        seed=args.seed,
        alpha=args.alpha,
        n_init=args.n_init,
        max_iter=args.max_iter,
        budgets=args.budgets,
        keep_top_k=args.keep_top_k,
        n_new_each_iter=args.n_new,
    )

    # -------------------------------------------------------------------------
    # 6) IteratedRacing : orchestrateur
    #    - appelle evaluator pour chaque config/budget
    #    - construit matrice des scores
    #    - appelle statistical_tests (Friedman) via iterated_racing.py
    #    - élimine + resample via sampler
    # -------------------------------------------------------------------------
    racing = IteratedRacing(
        sampler=sampler,
        evaluator=evaluator,
        logger=logger,
        settings=settings,
    )

    # Lance l'optimisation (pour l'instant, run_training_job n'est pas branché)
    final_population = racing.run()

    # Affiche top configs (val_loss dernière itération)
    print("\n=== Top 5 configs (triées par val_loss finale) ===")
    for c in final_population[:5]:
        print(f"- id={c.id}  val_loss={c.history[-1]:.6e}  params={c.params}")

    print(f"\nLogs écrits dans: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())