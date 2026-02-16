# Digital Twin Calibration (dtcalib)

Ce sous-projet fournit une pipeline **propre, testée et maintenable** pour :
- charger des séries temporelles entrée/sortie issues d’expériences,
- simuler un modèle paramétrique (digital twin),
- calibrer ses paramètres via **nonlinear least squares**,
- évaluer la généralisation via **Leave-One-Experiment-Out Cross-Validation (LOO-CV)**,
- mesurer la qualité via des métriques (MSE/RMSE/NMSE),
- garantir la stabilité via des **tests unitaires**.

> À ce stade, le simulateur Python est un modèle 1er ordre pour le circuit **R1–(C || R2)**.
> L’intégration EcoSimPro (simulation haute fidélité) viendra ensuite.

---

## Structure du projet
Digital_Twin_Calibration/
├─ pyproject.toml
├─ scripts/
│ └─ run_calibration_cv.py
├─ src/
│ └─ dtcalib/
│ ├─ init.py
│ ├─ data.py
│ ├─ metrics.py
│ ├─ simulation.py
│ ├─ calibration.py
│ └─ validation.py
└─ tests/
├─ test_data_loading.py
├─ test_metrics.py
├─ test_simulation.py
├─ test_validation.py
└─ test_calibration.py

## Installation (mode développement)
Depuis `Digital_Twin_Calibration/` :

```bash
pip install -e .
```

## Données attendues

Les datasets sont des dossiers contenant 1 CSV par expérience.

Chaque CSV doit contenir au minimum les colonnes suivantes (noms exacts par défaut) :

TIME

Addition_2.s_out.signal[1] (entrée)

SensorVoltage_1.v (sortie)

Les noms sont configurables dans ExperimentsDataset.from_csv_folder(...).
Exemples de dossiers existants (selon ton repo) :

data/LP_Dataset_csv_Reference
data/LP_Dataset_csv_C_Modified

## Modules dtcalib
### dtcalib/data.py

But : gérer les expériences et charger les CSV.

Experiment : dataclass (nom, t, u, y, meta)

ExperimentsDataset : container d’expériences

ExperimentsDataset.from_csv_folder(folder, ...) : charge un dossier de CSV

### dtcalib/metrics.py

But : calculer des métriques entre y_true et y_pred.

Metrics.mse(...)

Metrics.rmse(...)

Metrics.nmse(...) avec garde numérique

Metrics.compute(...) → MetricsResult(rmse, nmse, mse)

### dtcalib/simulation.py

But : fournir une interface de simulation paramétrique uniforme.

Simulator (classe abstraite) : impose simulate(t, u, theta) -> SimulationResult

SimulationResult : sortie simulée y + dictionnaire aux (diagnostics)

Simulateurs disponibles :

ExampleRCCircuitSimulator : modèle RC 1er ordre (template / debug)

LowPassR1CR2Simulator : modèle du circuit réel R1–(C || R2), discretisation exacte ZOH

Circuit Low-pass utilisé :

u -- R1 -- v
         |-- C -- GND
         |-- R2 -- GND


Ce simulateur permet d’estimer directement C (ou tau selon config).

### dtcalib/calibration.py

But : calibration des paramètres par nonlinear least squares.

LeastSquaresCalibrator :

minimise la somme des erreurs au carré sur toutes les expériences

supporte bounds, weights, loss robuste (linear, huber, etc.)

CalibrationReport :

theta_hat, cost, success, message, nfev

per_experiment_metrics pour diagnostiquer chaque expérience

### dtcalib/validation.py

But : évaluer la généralisation via Leave-One-Experiment-Out CV.

LeaveOneExperimentOutCV :

pour chaque expérience, on calibre sur les autres et on teste sur celle-là

CrossValidationResult.summary() :

moyenne/écart-type des RMSE et NMSE

## Scripts (exécution)
### scripts/run_calibration_cv.py

But : exécuter la pipeline complète sur un dataset réel :

chargement dataset

calibration + LOO-CV

affichage summary et quelques folds

(optionnel) affichage stats u/y

Exécution :

python3 scripts/run_calibration_cv.py


À adapter dans le script :

data_folder = Path("...")

choix du simulateur + paramètres (R1, R2, etc.)

theta0, bounds, max_nfev

## Tests unitaires

Lancer tous les tests :

pytest -q


### Tests inclus :

test_data_loading.py : vérifie le chargement CSV et la validation des colonnes

test_metrics.py : vérifie MSE/RMSE/NMSE et erreurs de shape

test_simulation.py : vérifie que le simulateur respecte le contrat (shapes, erreurs)

test_validation.py : vérifie la logique LOO-CV (avec mock calibrator/simulator)

test_calibration.py : vérifie que la calibration retrouve un paramètre connu (dataset synthétique)

Notes de performance

LOO-CV sur 100 expériences = 100 calibrations.

Chaque calibration appelle le simulateur de nombreuses fois (least squares).

Ajouter une progress bar via tqdm est recommandé pour suivre l’avancement.


## Auteurs / Contexte

Travail réalisé dans le cadre d’un TFE (calibration de paramètres d’un digital twin) avec datasets générés par EcoSimPro.
