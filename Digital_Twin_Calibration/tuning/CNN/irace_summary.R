#!/usr/bin/env Rscript

#irace_summary.R
#Analyse rapide d'un fichier irace.Rdata
#
#Usage:
#  Rscript irace_summary.R 
# 
# Example : 
# (torch_gpu) pc_ai@PCAI:/mnt/c/Users/samue/Documents/AI-Assisted_Implementation_of_a_Digital-Twin/Digital_Twin_Calibration/tuning$ Rscript irace_summary.R
#
#Si aucun argument n'est fourni, le script essaie: tuning/irace.Rdata
#Ce script :
#1) charge l'objet iraceResults
#2) affiche le budget configuré et le nombre d'évaluations réellement faites
#3) affiche l'ID de la meilleure configuration finale
#4) affiche la meilleure configuration complète
#5) affiche les élites finales
#6) exporte des CSV utiles pour inspection ultérieure



args <- commandArgs(trailingOnly = TRUE)
log_path <- if (length(args) >= 1) args[1] else "./irace.Rdata"

cat("=== IRACE SUMMARY ===\n")
cat("Log file:", normalizePath(log_path, mustWork = FALSE), "\n\n")

if (!file.exists(log_path)) {
  stop("Fichier introuvable: ", log_path)
}

# Charge le .Rdata et vérifie qu'il contient bien iraceResults
loaded_names <- load(log_path)
if (!"iraceResults" %in% loaded_names) {
  stop("Le fichier ne contient pas d'objet 'iraceResults'. Objets trouvés: ",
       paste(loaded_names, collapse = ", "))
}

# Vérifications minimales
required_fields <- c("scenario", "allConfigurations", "experiments", "iterationElites")
missing_fields <- setdiff(required_fields, names(iraceResults))
if (length(missing_fields) > 0) {
  stop("Champs manquants dans iraceResults: ", paste(missing_fields, collapse = ", "))
}

# ---------- 1 Infos générales ----------
max_exp <- iraceResults$scenario$maxExperiments
exp_matrix <- iraceResults$experiments
used_exp <- sum(!is.na(exp_matrix))
exp_dims <- dim(exp_matrix)

cat("[1] Informations générales\n")
cat("- Budget maxExperiments configuré :", max_exp, "\n")
cat("- Nombre d'évaluations effectivement enregistrées :", used_exp, "\n")
cat("- Dimension de la matrice experiments :",
    paste(exp_dims, collapse = " x "),
    "(instances x configurations)\n")
cat("- Exécution probablement complète ? :", if (used_exp >= max_exp) "oui" else "non / interrompue", "\n\n")

# ---------- 2 Meilleure configuration finale ----------
best_id <- tail(iraceResults$iterationElites, 1)
all_cfg <- iraceResults$allConfigurations

# Compatibilité selon format data.frame / data.table
id_col <- ".ID."
if (!id_col %in% colnames(all_cfg)) {
  stop("La colonne '.ID.' est introuvable dans allConfigurations.")
}

best_cfg <- all_cfg[all_cfg[[id_col]] == best_id, , drop = FALSE]

cat("[2] Meilleure configuration finale\n")
cat("- ID de la meilleure configuration :", best_id, "\n")
print(best_cfg)
cat("\n")

# ---------- 3 Élites finales ----------
cat("[3] Élites finales\n")
if ("allElites" %in% names(iraceResults) && length(iraceResults$allElites) > 0) {
  final_elite_ids <- tail(iraceResults$allElites, 1)[[1]]
  final_elites <- all_cfg[all_cfg[[id_col]] %in% final_elite_ids, , drop = FALSE]
  cat("- IDs des élites finales :", paste(final_elite_ids, collapse = ", "), "\n")
  print(final_elites)
} else {
  cat("- allElites absent ou vide.\n")
}
cat("\n")

# ---------- 4 Configs les plus évaluées ----------
cat("[4] Nombre d'évaluations par configuration\n")
per_cfg <- colSums(!is.na(exp_matrix))
per_cfg_df <- data.frame(configuration_id = names(per_cfg), n_evals = as.integer(per_cfg))
per_cfg_df <- per_cfg_df[order(-per_cfg_df$n_evals), , drop = FALSE]
print(utils::head(per_cfg_df, 10))
cat("\n")

# ---------- 5 Détail des évaluations de la meilleure config ----------
cat("[5] Évaluations de la meilleure configuration\n")
best_col <- as.character(best_id)
if (best_col %in% colnames(exp_matrix)) {
  best_exp <- exp_matrix[, best_col, drop = FALSE]
  print(best_exp)
} else {
  cat("- La colonne correspondant à la meilleure configuration n'a pas été trouvée dans experiments.\n")
}
cat("\n")

# ---------- 6 Exports CSV ----------
#out_dir <- dirname(normalizePath(log_path, mustWork = FALSE))
#summary_csv <- file.path(out_dir, "best_configuration.csv")
#elites_csv <- file.path(out_dir, "final_elites.csv")
#evals_csv <- file.path(out_dir, "evaluations_per_configuration.csv")
#
#utils::write.csv(best_cfg, summary_csv, row.names = FALSE)
#if (exists("final_elites")) {
#  utils::write.csv(final_elites, elites_csv, row.names = FALSE)
#}
#utils::write.csv(per_cfg_df, evals_csv, row.names = FALSE)
#
#cat("[6] Fichiers exportés\n")
#cat("-", summary_csv, "\n")
#if (exists("final_elites")) cat("-", elites_csv, "\n")
#cat("-", evals_csv, "\n\n")

cat("=== FIN DU RÉSUMÉ IRACE ===\n")
