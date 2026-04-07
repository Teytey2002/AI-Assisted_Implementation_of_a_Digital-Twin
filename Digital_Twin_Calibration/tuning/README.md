# 📌 Overview

Ce dossier contient toute l’infrastructure de tuning avec irace, utilisée pour deux cas principaux :

1) Calibration physique (estimation de paramètres comme C)
2) Optimisation d’hyperparamètres (CNN, GA, etc.)

Chaque sous-dossier de tuning/ correspond à un use case indépendant.

# 📁 Structure
```
tuning/
│
├── calibration/
│   ├── target-runner
│   ├── calibration_runner.py
│   ├── parameters.txt
│   ├── instances-list.txt
│   ├── scenario.txt
│   ├── configurations.txt (optionnel)
│   ├── irace.Rdata
│   ├── iracedump.rda
│   └── irace_summary.R
│
├── cnn/
├── ga/
```

# ⚙️ 1. Calibration avec irace
## 🎯 Objectif

Estimer un paramètre physique (ex: capacité C) en minimisant l’erreur entre :

Vout_simulé vs Vout_mesuré

👉 irace remplace ici :

least squares
Bayesian calibration
📄 Fichiers principaux
🔹 calibration_runner.py
Rôle

Script Python exécuté par irace pour :

Charger un CSV (une expérience)
Simuler le circuit RC
Calculer l’erreur (RMSE)
Retourner le score à irace
Entrées
--instance <csv>
--logC <valeur>
Sortie
<score numérique>

## 🔹 target-runner
Rôle :

Interface entre irace et Python

👉 Transforme :

logC-4.29

en :

--logC -4.29

👉 Puis appelle :

calibration_runner.py
## 🔹 parameters.txt

Définit les paramètres à optimiser

# name    switch   type   domain
logC      "logC"    r     (-20, -2)

👉 Ici :

optimisation en log(C)
domaine large (1e-20 à 1e-2)

## 🔹 instances-list.txt

Liste des expériences (CSV)

👉 1 ligne = 1 expérience

path/to/file1.csv
path/to/file2.csv
...


## 🔹 scenario.txt

Configuration globale d’irace

Paramètres importants :
maxExperiments = 500
blockSize = 5
eachTest = 1
firstTest = 1
elitist = 1
Explication
Paramètre	Rôle
maxExperiments	budget total
blockSize	nombre d’instances par bloc
eachTest	fréquence des tests statistiques
elitist	garde les meilleurs

## 🔹 configurations.txt (optionnel)

Permet d’initialiser irace avec des solutions connues

👉 Peut être vide ou supprimé

## 🔹 irace.Rdata

Contient :

toutes les évaluations
les scores
les configurations testées

👉 fichier principal de résultat

## 🔹 iracedump.rda

Version brute pour debug

## 🔹 irace_summary.R

Script pour analyser les résultats :

meilleure config
ranking
statistiques

# 🚀 Commandes importantes
## ✅ 1. Tester le runner manuellement
```
cd tuning/calibration
./target-runner 1 1 123 <csv_path> logC-13
```

👉 attendu : <score>

## 🔁 2. Lancer irace

```
cd tuning/calibration
irace
```

👉 lance l’optimisation complète

### 📊 3. Analyser les résultats

Dans R :

load("irace.Rdata")

iraceResults$bestConfigurations

Ou avec le script :

source("irace_summary.R")

#### 🧠 Interprétation des résultats

irace retourne :

logC*

Conversion :

C = np.exp(logC)

# ⚠️ Points importants
## 🔹 1. Format des paramètres

irace envoie :

logC-4.2

👉 PAS :

--logC -4.2

➡️ géré dans target-runner

🔹 2. Instances

👉 1 instance = 1 expérience

⚠️ ne pas mélanger différentes capacités

🔹 3. Bloc size
nb_instances % blockSize == 0

👉 sinon :

erreur irace
ou duplication (comme tu as fait)
🔬 Extensions futures
🔹 1. Multi-paramètres
R1, R2, C

👉 irace devient très puissant

🔹 2. Hyperparam tuning CNN

Dans tuning/cnn/ :

lr
batch_size
weight_decay
🔹 3. Optimisation GA

Dans tuning/ga/ :

population_size
mutation_rate
crossover_rate
🧠 Insight clé

Tu utilises irace comme :

👉 optimiseur global black-box

C’est différent de :

Méthode	Type
Least Squares	gradient-based
Bayesian	probabiliste
irace	stochastic global search


✅ Conclusion

Ce dossier permet :

✔ calibration physique via optimisation
✔ tuning hyperparamètres CNN
✔ pipeline modulaire et extensible

