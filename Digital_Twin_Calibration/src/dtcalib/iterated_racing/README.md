# Iterated Racing – Hyperparameter Optimization

This module implements an Iterated Racing (F-Race inspired) procedure to automatically tune the hyperparameters of the RC inverse CNN model.

The goal is to identify a statistically robust configuration of hyperparameters before performing a final full training with early stopping.

This implementation follows the methodology presented in:

López-Ibáñez et al., The irace package: Iterated Racing for Automatic Algorithm Configuration, 2016.

## 🎯 Objective

We want to:

1) Compare multiple hyperparameter configurations
2) Evaluate them under increasing training budgets (epochs)
3) Statistically eliminate poor performers (Friedman test)
4) Adaptively resample new configurations
5) Return the most promising configurations

All evaluations are done with:

- Fixed number of epochs
- No early stopping
- Deterministic seed
- Same dataset split
- Validation loss as the optimization metric

## 📁 Folder Structure
```
iterated_racing/
│
├── config.py
├── evaluator.py
├── iterated_racing.py
├── logger.py
├── run.py
├── sampler.py
├── search_space.py
├── statistical_tests.py
├── train.py
└── README.md
```

## 🧩 Script Roles
### search_space.py

Defines the hyperparameter search space:

- Learning rate (log-uniform)
- Weight decay (log-uniform)
- Batch size
- Optimizer (Adam / AdamW)
- Scheduler parameters
- Target transform (C or logC)

This is the only place where the search domain is defined.

### sampler.py

- Responsible for generating configurations.
- initial_population(n) → random sampling
- adaptive_resample(survivors, n_new) → generate new configs around best ones

Uses:

- Log-uniform sampling for continuous parameters
- Discrete sampling for categorical parameters
- Deterministic seed for reproducibility

### evaluator.py

Interface between Iterated Racing and the training pipeline.

Responsibilities:

- Fix random seed

- Call run_training_job(...)

- Measure wall time

- Handle optional caching

- Clean GPU memory after evaluation

Returns:
```
{
    "val_loss": float,
    "train_loss": float
}
```
### train.py

Contains:

- run_training_job_fixed_epochs(...)

- This is the short training procedure used during racing.

Characteristics:

- Uses fixed dataset split (JSON)
- Computes normalization on train only
- Applies target transform (C or logC)
- Runs for fixed number of epochs
- No early stopping
- No TensorBoard
- No model saving

This ensures fair and deterministic comparison.

### statistical_tests.py

Implements the Friedman test using SciPy.

Function:

- friedman_race(scores_matrix, alpha, keep_top_k)

Behavior:

- If no statistical difference → keep all
- If significant difference → keep top-K (lowest mean ranks)
- Robust to degenerate cases (identical scores)

Returns a RaceDecision object:
```
RaceDecision(
    p_value,
    keep_indices,
    mean_ranks
)
```

### logger.py

Writes two CSV files:

- racing_evals.csv
- racing_decisions.csv

Each evaluation row contains:

- iteration
- config_id
- seed
- epochs
- val_loss
- train_loss
- wall_time
- params_json

Each decision row contains:

- iteration
- p_value
- kept configuration ids
- mean ranks

This makes the process fully traceable and reproducible.

### iterated_racing.py

Core algorithm.

Responsibilities:

- Generate initial population
- Evaluate configurations
- Build performance matrix
- Apply Friedman test
- Eliminate poor configs
- Resample new configs
- Repeat for increasing budgets

This is the orchestration engine.

### run.py

Main entry point.

Responsible for:

- Parsing CLI arguments
- Instantiating SearchSpace
- Creating Sampler
- Creating Evaluator
- Creating Logger
- Configuring RacingSettings
- Launching Iterated Racing

## 📊 Dataset Requirements

This module expects:

- Dataset root directory
- A fixed split JSON file (train/val/test)

Split file example:
```
src/dtcalib/deep_learning/splits/rc_split_seed42_70_15_15.json
```

The split must be stratified by C value.

## 🚀 How to Run

From the project root:
```
python -m dtcalib.iterated_racing.run \
  --root-dir ../../../data/ALL_LP_DATASETS_CSV_Deep_learning \
  --split-json src/dtcalib/deep_learning/splits/rc_split_seed42_70_15_15.json \
  --device cuda \
  --cache
```

Optional Arguments
```
Argument	Description
--n-init	Initial population size
--max-iter	Number of racing iterations
--budgets	Epochs per iteration
--keep-top-k	Survivors per iteration
--n-new	New configs per iteration
--alpha	Significance threshold
--seed	Global seed
--out-dir	Output directory
```

Example:
```
python -m dtcalib.iterated_racing.run \
  --root-dir ../../../data/ALL_LP_DATASETS_CSV_Deep_learning \
  --split-json src/dtcalib/deep_learning/splits/rc_split_seed42_70_15_15.json \
  --n-init 30 \
  --max-iter 6 \
  --budgets 50 75 100 150 200 250 \
  --keep-top-k 15 10 8 6 5 5 \
  --n-new 15 \
  --alpha 0.05 \
  --cache
```

## 📈 Output

Results are written to:

results/racing/

- racing_evals.csv
- racing_decisions.csv

You can analyze:

- Convergence behavior
- Survival dynamics
- p-values per iteration
- Mean rank evolution

## 🔬 Scientific Justification

This approach is:

- Budget-aware
- Statistically grounded
- Robust to noise
- More efficient than grid search
- More principled than naive random search

## ⚠ Important Notes

- No early stopping during racing
- Final best configuration must be retrained separately using the full training pipeline (deep_learning/train.py)
- Validation loss is the optimization metric
- Test set must only be used once final configuration is selected