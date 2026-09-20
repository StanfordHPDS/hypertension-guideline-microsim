# Pipeline orchestration for the microsim-DNH-HS-AFC simulation.
# See dependencies.md for the per-stage input/output contract; this Makefile
# encodes the same DAG with sentinel files so rebuilds are content-driven.
# Run `make help` for the target list.

# -------- Configuration --------

FOLDER            ?= final_cohort
NHANES_FILE       ?= NHANES_final_cohort
VALIDATION_FOLDER ?= AFC_Validation_Cohort
VALIDATION_SIZE   ?= 10000

UV_RUN := uv run
PY     := $(UV_RUN) python
QUARTO := $(UV_RUN) quarto render

DATA    := data_and_models
AFC_M   := $(DATA)/afc_models
AFC_O   := $(DATA)/afc_outputs
NHANES  := $(DATA)/nhanes_data_cohorts
NHINPUT := $(DATA)/nhanes_inputs
RES     := results/$(FOLDER)
VRES    := validation_results/$(VALIDATION_FOLDER)
# Validation-arm sentinel stamps. Live outside $(VRES) so they can be tracked
# in git (the $(VRES) tree is gitignored because its contents are PHI-bearing).
# Tracked stamps let `make -t pipeline` succeed on machines that pull from
# AFC without needing the underlying validation artifacts.
VSTAMPS := results/afc_validation/.stamps

# Helper module groupings used as recipe prerequisites. Each variable expands
# to the module file plus the transitive closure of helpers it imports, so a
# rule that lists a variable picks up the full mtime chain affecting its
# recipe. Adding a module here is the only way an mtime change propagates to
# rules that import it; modules NOT in a variable's expansion cannot trigger
# rebuilds of rules that use that variable.
PY_PATHS        := code/Python/paths.py
PY_NUMERIC      := code/Python/numeric.py
PY_MODEL_FIT    := code/Python/model_fitting.py
PY_SDI          := code/Python/sdi.py
PY_BP_GEN       := code/Python/bp_generation.py $(PY_NUMERIC)
PY_MORTALITY    := code/Python/mortality.py $(PY_NUMERIC) $(PY_PATHS)
PY_STATS        := code/Python/stats.py $(PY_NUMERIC)
PY_REPORTING    := code/Python/reporting.py $(PY_NUMERIC)
PY_RESULTS_MISC := code/Python/results_misc.py $(PY_MORTALITY)
PY_DEMOGRAPHICS := code/Python/demographics.py $(PY_PATHS) $(PY_RESULTS_MISC) $(PY_STATS)
PY_PLOTTING     := code/Python/plotting.py $(PY_NUMERIC) $(PY_PATHS) $(PY_RESULTS_MISC) $(PY_STATS)
PY_SIM_FUNCS    := code/Python/vectorized_hypertension_simulation_functions.py $(PY_BP_GEN) $(PY_MORTALITY)
PY_SIM_FUNCS_S  := code/Python/vectorized_hypertension_simulation_functions_sff.py $(PY_BP_GEN) $(PY_MORTALITY)
LIFETABLES     := $(wildcard $(DATA)/lifetables_2021/*.xlsx)

ASSUMPTIONS := hetero_RA hetero_NR nohetero_RA nohetero_NR

STD_SIM_OUTPUTS    := $(foreach a,$(ASSUMPTIONS),$(RES)/cohort_outcomes/standard/results_$(a).csv)
SOCIAL_SIM_OUTPUTS := $(foreach a,$(ASSUMPTIONS),$(RES)/cohort_outcomes/social/results_$(a).csv)

.DEFAULT_GOAL := help

# Delete partially-written targets on recipe failure so retries rebuild cleanly
# instead of treating stale half-outputs as up-to-date.
.DELETE_ON_ERROR:

.PHONY: help \
        prime-tables prime-sample \
        nhanes-download nhanes-copula \
        quantile-regressions-standard quantile-regressions-social \
        correlations-standard correlations-social \
        sdi out-of-system-bp \
        measurement-standard measurement-social \
        treatment-standard treatment-social \
        cohort-standard cohort-social \
        sim-standard-hetero-RA sim-standard-hetero-NR \
        sim-standard-nohetero-RA sim-standard-nohetero-NR \
        sim-social-hetero-RA sim-social-hetero-NR \
        sim-social-nohetero-RA sim-social-nohetero-NR \
        sim-standard sim-social sim-all \
        pipeline-standard pipeline-social \
        pipeline-main-sim pipeline-main-models pipeline-main \
        pipeline-validation pipeline \
        validation-cohort validation-sim-1 validation-sim-2 \
        validation-analysis \
        figures-main-results-by-race-standard figures-main-results-by-race-social \
        render-manuscript reports \
        lint format format-check ci-check \
        clean clean-results clean-validation deepclean

# -------- help --------

help: ## Show available targets grouped by stage
	@awk 'BEGIN {FS = ":.*?## "} \
	     /^[a-zA-Z_-][a-zA-Z0-9._-]*:.*?## / { \
	         printf "  \033[36m%-32s\033[0m %s\n", $$1, $$2 \
	     } \
	     /^# ==== / { printf "\n\033[1m%s\033[0m\n", substr($$0, 6) }' $(MAKEFILE_LIST)

# ==== Stage 0: PRIME BigQuery table build (optional, requires PRIME access) ====

prime-tables: code/SQL/hypertension_sql_queries.sql code/Python/build_prime_tables.py ## Build/refresh PRIME tables in mmcusick_afc from hypertension_sql_queries.sql
	$(PY) code/Python/build_prime_tables.py

prime-sample: code/SQL/sample_snapshot.sql code/Python/build_prime_tables.py ## MANUAL: rebuild the 1M-row sample snapshots used by quantile-regressions and measurement-frequency stages
	$(PY) code/Python/build_prime_tables.py --sql code/SQL/sample_snapshot.sql

# ==== Stage 1a: NHANES raw download (R/Quarto) ====

nhanes-download: $(NHANES)/nhanes.csv ## Download NHANES XPT files from CDC and join into nhanes.csv (R)

$(NHANES)/nhanes.csv: code/R/NHANES_download.qmd
	$(QUARTO) code/R/NHANES_download.qmd

# ==== Stage 1b: NHANES cohort (R/Quarto) ====

nhanes-copula: $(NHANES)/NHANES_final_cohort.csv ## Fit NHANES copula and sample the 100k cohort (R)

$(NHANES)/NHANES_final_cohort.csv: \
        code/R/NHANES_multivariate_copula.qmd \
        $(NHANES)/nhanes.csv \
        $(NHINPUT)/change_corr.csv
	$(QUARTO) code/R/NHANES_multivariate_copula.qmd

# ==== Stage 2: Starting-BP quantile regressions (Python) ====

quantile-regressions-standard: $(AFC_O)/standard/systolic_splines.csv ## SBP/DBP quantile regressions (standard)
quantile-regressions-social:   $(AFC_O)/social/systolic_splines.csv   ## SBP/DBP quantile regressions (SFF)

$(AFC_O)/standard/systolic_splines.csv: code/Python/systolic_diastolic_quantile_regressions.py $(PY_MODEL_FIT) $(PY_PATHS)
	$(PY) code/Python/systolic_diastolic_quantile_regressions.py -sff "0"

$(AFC_O)/social/systolic_splines.csv: code/Python/systolic_diastolic_quantile_regressions.py $(PY_MODEL_FIT) $(PY_PATHS)
	$(PY) code/Python/systolic_diastolic_quantile_regressions.py -sff "1"

# ==== Stage 3: Quantile correlations (Python) ====

correlations-standard: $(AFC_O)/standard/AFC_age_group_correlations/saved_0.csv ## Age/sex quantile correlations (standard)
correlations-social:   $(AFC_O)/social/AFC_age_group_correlations/saved_0.csv   ## Age/sex/SDI quantile correlations (SFF)

$(AFC_O)/standard/AFC_age_group_correlations/saved_0.csv: \
        $(AFC_O)/standard/systolic_splines.csv \
        code/Python/systolic_diastolic_correlations.py $(PY_MODEL_FIT) $(PY_PATHS)
	$(PY) code/Python/systolic_diastolic_correlations.py -sff "0"

$(AFC_O)/social/AFC_age_group_correlations/saved_0.csv: \
        $(AFC_O)/social/systolic_splines.csv \
        code/Python/systolic_diastolic_correlations.py $(PY_MODEL_FIT) $(PY_PATHS)
	$(PY) code/Python/systolic_diastolic_correlations.py -sff "1"

# ==== Stage 4: SDI imputation (Python) ====

sdi: $(NHANES)/$(NHANES_FILE)_sdi.csv ## Bayesian SDI imputation onto NHANES cohort

$(NHANES)/$(NHANES_FILE)_sdi.csv: \
        $(NHANES)/NHANES_final_cohort.csv \
        code/Python/SDI_code.py $(PY_SDI) $(NHINPUT)/SDI_CT.csv
	$(PY) code/Python/SDI_code.py -d "$(NHANES_FILE)"

# ==== Stage 5: Out-of-system BP adjustment (R/Quarto) ====

out-of-system-bp: $(NHINPUT)/out_of_system_sys_quantile_betas.csv ## Out-of-system BP regression coefficients (R)

$(NHINPUT)/out_of_system_sys_quantile_betas.csv: \
        $(NHANES)/nhanes.csv \
        code/R/NHANES_out_of_system_BP.qmd
	$(QUARTO) code/R/NHANES_out_of_system_BP.qmd

# ==== Stage 6: Measurement-frequency quantile regressions (Python) ====

measurement-standard: $(AFC_O)/standard/afc_measurement_quantiles.csv ## Months-to-next-measurement regressions (standard)
measurement-social:   $(AFC_O)/social/afc_measurement_quantiles.csv   ## Months-to-next-measurement regressions (SFF)

$(AFC_O)/standard/afc_measurement_quantiles.csv \
$(AFC_O)/standard/month_difference_mean.json \
$(AFC_O)/standard/month_difference_std.json &: \
        code/Python/measurement_frequency.py $(PY_BP_GEN) $(PY_MODEL_FIT) $(PY_PATHS)
	$(PY) code/Python/measurement_frequency.py -sff "0"

$(AFC_O)/social/afc_measurement_quantiles.csv: \
        code/Python/measurement_frequency.py $(PY_BP_GEN) $(PY_MODEL_FIT) $(PY_PATHS)
	$(PY) code/Python/measurement_frequency.py -sff "1"

# ==== Stage 7: Treatment-initiation logistic regression (Python) ====

treatment-standard: $(AFC_M)/standard/treatment_models/logistic_treatment_model.pickle ## Treatment logit (standard)
treatment-social:   $(AFC_M)/social/treatment_models/logistic_treatment_model.pickle   ## Treatment logit (SFF)

$(AFC_M)/standard/treatment_models/logistic_treatment_model.pickle: code/Python/treatment_logit_model.py
	$(PY) code/Python/treatment_logit_model.py -sff "0"

$(AFC_M)/social/treatment_models/logistic_treatment_model.pickle: code/Python/treatment_logit_model.py
	$(PY) code/Python/treatment_logit_model.py -sff "1"

# ==== Stage 8: Cohort assembly for simulation (Python) ====

cohort-standard: $(RES)/cohort_files/standard/sampled_cohort.csv ## Assign starting BP + lifetime quantiles (standard)
cohort-social:   $(RES)/cohort_files/social/sampled_cohort.csv   ## Assign starting BP + lifetime quantiles (SFF)

$(RES)/cohort_files/standard/sampled_cohort.csv: \
        $(NHANES)/$(NHANES_FILE)_sdi.csv \
        $(AFC_O)/standard/systolic_splines.csv \
        $(AFC_O)/standard/AFC_age_group_correlations/saved_0.csv \
        $(AFC_O)/standard/afc_measurement_quantiles.csv \
        $(AFC_M)/standard/treatment_models/logistic_treatment_model.pickle \
        code/Python/cohort_sampling_NHANES.py \
        $(PY_BP_GEN) $(PY_DEMOGRAPHICS) $(PY_MODEL_FIT) $(PY_NUMERIC) $(PY_PATHS)
	$(PY) code/Python/cohort_sampling_NHANES.py -f "$(FOLDER)" -d "$(NHANES_FILE)" -sff "0"

$(RES)/cohort_files/social/sampled_cohort.csv: \
        $(NHANES)/$(NHANES_FILE)_sdi.csv \
        $(AFC_O)/social/systolic_splines.csv \
        $(AFC_O)/social/AFC_age_group_correlations/saved_0.csv \
        $(AFC_O)/social/afc_measurement_quantiles.csv \
        $(AFC_M)/social/treatment_models/logistic_treatment_model.pickle \
        code/Python/cohort_sampling_NHANES.py \
        $(PY_BP_GEN) $(PY_DEMOGRAPHICS) $(PY_MODEL_FIT) $(PY_NUMERIC) $(PY_PATHS)
	$(PY) code/Python/cohort_sampling_NHANES.py -f "$(FOLDER)" -d "$(NHANES_FILE)" -sff "1"

# ==== Stage 9: Simulation runs (Python, 2 frameworks x hetero/nohetero x RA/NR) ====

sim-standard-hetero-RA:   $(RES)/cohort_outcomes/standard/results_hetero_RA.csv    ## Standard: hetero, race-adjusted
sim-standard-hetero-NR:   $(RES)/cohort_outcomes/standard/results_hetero_NR.csv    ## Standard: hetero, race-neutral
sim-standard-nohetero-RA: $(RES)/cohort_outcomes/standard/results_nohetero_RA.csv  ## Standard: nohetero, race-adjusted
sim-standard-nohetero-NR: $(RES)/cohort_outcomes/standard/results_nohetero_NR.csv  ## Standard: nohetero, race-neutral
sim-social-hetero-RA:     $(RES)/cohort_outcomes/social/results_hetero_RA.csv      ## SFF: hetero, race-adjusted
sim-social-hetero-NR:     $(RES)/cohort_outcomes/social/results_hetero_NR.csv      ## SFF: hetero, race-neutral
sim-social-nohetero-RA:   $(RES)/cohort_outcomes/social/results_nohetero_RA.csv    ## SFF: nohetero, race-adjusted
sim-social-nohetero-NR:   $(RES)/cohort_outcomes/social/results_nohetero_NR.csv    ## SFF: nohetero, race-neutral

# Static pattern rule: stem captures "<assumption>_<adjustment>". Split on '_' for
# the script's -a/-ra args. 4 targets share this rule per framework.
$(STD_SIM_OUTPUTS): $(RES)/cohort_outcomes/standard/results_%.csv: \
        $(RES)/cohort_files/standard/sampled_cohort.csv \
        code/Python/vectorized_hypertension_simulation_model.py \
        $(PY_SIM_FUNCS) $(PY_DEMOGRAPHICS) $(PY_NUMERIC) $(PY_PLOTTING) $(PY_STATS) \
        $(LIFETABLES)
	$(PY) code/Python/vectorized_hypertension_simulation_model.py \
	    -f "$(FOLDER)" \
	    -a "$(word 1,$(subst _, ,$*))" \
	    -ra "$(word 2,$(subst _, ,$*))"

# Standard hetero_RA reads the validation-arm KM curve to draw the
# cum_treatment_exposure_hetero_RA.png calibration overlay against PRIME data.
# The other 7 sims have no cross-arm reads.
$(RES)/cohort_outcomes/standard/results_hetero_RA.csv: \
        $(AFC_O)/standard/treatment_cumulative_density.csv

$(SOCIAL_SIM_OUTPUTS): $(RES)/cohort_outcomes/social/results_%.csv: \
        $(RES)/cohort_files/social/sampled_cohort.csv \
        $(NHINPUT)/out_of_system_sys_quantile_betas.csv \
        code/Python/vectorized_hypertension_simulation_model_sff.py \
        $(PY_SIM_FUNCS_S) $(PY_DEMOGRAPHICS) $(PY_NUMERIC) $(PY_PLOTTING) $(PY_STATS) \
        $(LIFETABLES)
	$(PY) code/Python/vectorized_hypertension_simulation_model_sff.py \
	    -f "$(FOLDER)" \
	    -a "$(word 1,$(subst _, ,$*))" \
	    -ra "$(word 2,$(subst _, ,$*))"

# ==== Aggregates ====

sim-standard: sim-standard-hetero-RA sim-standard-hetero-NR sim-standard-nohetero-RA sim-standard-nohetero-NR ## All 4 standard-framework sims
sim-social:   sim-social-hetero-RA sim-social-hetero-NR sim-social-nohetero-RA sim-social-nohetero-NR         ## All 4 SFF sims
sim-all: sim-standard sim-social ## All 8 simulation scenarios

pipeline-standard: sim-standard ## Full standard pipeline (upstream prereqs included via DAG)
pipeline-social:   sim-social   ## Full SFF pipeline (upstream prereqs included via DAG)

pipeline-main-sim: sim-all ## Cohort assembly + all 8 sims from checked-in AFC artifacts (no BigQuery)
pipeline-main-models: quantile-regressions-standard quantile-regressions-social correlations-standard correlations-social sdi measurement-standard measurement-social treatment-standard treatment-social ## Rebuild all PRIME-derived models/outputs (requires AFC/BigQuery)
pipeline-main: pipeline-main-models pipeline-main-sim ## Full main analysis arm (stages 1-10)

pipeline: pipeline-main pipeline-validation reports ## Full pipeline end-to-end: main + validation + reports

# ==== Validation arm ====

validation-cohort: $(VSTAMPS)/cohort.stamp ## Sample validation cohort from PRIME (requires BigQuery)

# Cohort .csv lives under $(VRES)/ (gitignored, PHI). The make sentinel is the
# tracked stamp so downstream `make -t` works on non-AFC machines.
$(VSTAMPS)/cohort.stamp: code/Python/cohort_sampling_validation.py $(PY_NUMERIC) \
        $(AFC_O)/standard/systolic_splines.csv \
        $(AFC_O)/standard/AFC_age_group_correlations/saved_0.csv
	$(PY) code/Python/cohort_sampling_validation.py -f "$(VALIDATION_FOLDER)" -n "$(VALIDATION_SIZE)"
	@mkdir -p $(dir $@) && touch $@

# Validation simulation writes parquet datasets under $(VRES)/ (gitignored);
# the tracked stamps under $(VSTAMPS)/ are the make-visible sentinels.
validation-sim-1: $(VSTAMPS)/sim-1.stamp ## Run validation simulation (first half)
validation-sim-2: $(VSTAMPS)/sim-2.stamp ## Run validation simulation (second half)

$(VSTAMPS)/sim-1.stamp: \
        $(VSTAMPS)/cohort.stamp \
        code/Python/hypertension_simulation_model_validation.py \
        $(PY_BP_GEN) $(PY_MORTALITY) $(PY_NUMERIC) $(LIFETABLES)
	$(PY) code/Python/hypertension_simulation_model_validation.py -f "$(VALIDATION_FOLDER)" -n "1"
	@mkdir -p $(dir $@) && touch $@

$(VSTAMPS)/sim-2.stamp: \
        $(VSTAMPS)/cohort.stamp \
        code/Python/hypertension_simulation_model_validation.py \
        $(PY_BP_GEN) $(PY_MORTALITY) $(PY_NUMERIC) $(LIFETABLES)
	$(PY) code/Python/hypertension_simulation_model_validation.py -f "$(VALIDATION_FOLDER)" -n "2"
	@mkdir -p $(dir $@) && touch $@

validation-analysis: results/afc_validation/sdi_mean.json ## Post-process validation simulation outputs

results/afc_validation/sdi_mean.json $(AFC_O)/standard/treatment_cumulative_density.csv &: \
        $(VSTAMPS)/sim-1.stamp $(VSTAMPS)/sim-2.stamp \
        code/Python/validation_analysis.py $(PY_RESULTS_MISC)
	$(PY) code/Python/validation_analysis.py -f "$(VALIDATION_FOLDER)"

pipeline-validation: validation-analysis ## Full validation arm: cohort + two sim halves + analysis (requires AFC/BigQuery)

# ==== Stage 10: Per-figure rendering (Python) ====

MAIN_RESULTS_OUTCOMES := years_sick years_to_death years_sick_treated years_treated
STD_MAIN_RESULTS_PNGS    := $(foreach o,$(MAIN_RESULTS_OUTCOMES),figures/standard/main_results_by_race_$(o).png)
SOCIAL_MAIN_RESULTS_PNGS := $(foreach o,$(MAIN_RESULTS_OUTCOMES),figures/social/main_results_by_race_$(o).png)

figures-main-results-by-race-standard: $(STD_MAIN_RESULTS_PNGS) ## Bar plots of main outcomes by race (standard arm)
figures-main-results-by-race-social:   $(SOCIAL_MAIN_RESULTS_PNGS) ## Bar plots of main outcomes by race (SFF arm)

$(STD_MAIN_RESULTS_PNGS) &: \
        code/Python/main_results_by_race_figure.py \
        $(STD_SIM_OUTPUTS) $(PY_PATHS)
	$(PY) code/Python/main_results_by_race_figure.py --analysis standard -f "$(FOLDER)"

$(SOCIAL_MAIN_RESULTS_PNGS) &: \
        code/Python/main_results_by_race_figure.py \
        $(SOCIAL_SIM_OUTPUTS) $(PY_PATHS)
	$(PY) code/Python/main_results_by_race_figure.py --analysis social -f "$(FOLDER)"

# ==== Report rendering ====

render-manuscript: manuscript_draft.html manuscript_draft.docx ## Render manuscript_draft.qmd

manuscript_draft.html manuscript_draft.docx &: manuscript_draft.qmd \
        $(PY_REPORTING) \
        $(STD_SIM_OUTPUTS) $(SOCIAL_SIM_OUTPUTS) \
        $(STD_MAIN_RESULTS_PNGS) $(SOCIAL_MAIN_RESULTS_PNGS) \
        results/afc_validation/sdi_mean.json \
        $(AFC_O)/standard/month_difference_mean.json \
        $(AFC_O)/standard/month_difference_std.json
	$(QUARTO) manuscript_draft.qmd

reports: render-manuscript ## Render the manuscript

# ==== Dev convenience ====

lint: ## Run ruff check (matches CI)
	$(UV_RUN) ruff check .

format: ## Apply ruff format
	$(UV_RUN) ruff format .

format-check: ## Check ruff format without applying (matches CI)
	$(UV_RUN) ruff format --check .

ci-check: format-check lint ## Run both CI lint jobs locally

sync-mtimes: ## Reset tracked-file mtimes to last-commit time (fixes spurious rebuilds after git checkout)
	$(PY) scripts/sync_mtimes.py

clean-results: ## Remove results/$(FOLDER) after confirmation
	@read -p "rm -rf results/$(FOLDER) [y/N]? " ans && [ "$$ans" = "y" ] && rm -rf "results/$(FOLDER)"

clean-validation: ## Remove validation_results/$(VALIDATION_FOLDER) and tracked stamps after confirmation
	@read -p "rm -rf validation_results/$(VALIDATION_FOLDER) and $(VSTAMPS) [y/N]? " ans && [ "$$ans" = "y" ] && rm -rf "validation_results/$(VALIDATION_FOLDER)" "$(VSTAMPS)"

clean: ## Wipe per-run outputs: results/$(FOLDER), validation_results/$(VALIDATION_FOLDER), validation stamps, rendered reports
	@read -p "rm -rf results/$(FOLDER), validation_results/$(VALIDATION_FOLDER), $(VSTAMPS), rendered .html + .docx + _files/ [y/N]? " ans && [ "$$ans" = "y" ] && rm -rf \
	    "results/$(FOLDER)" \
	    "validation_results/$(VALIDATION_FOLDER)" \
	    "$(VSTAMPS)" \
	    manuscript_draft.html \
	    manuscript_draft.docx \
	    manuscript_draft_files

deepclean: ## clean + wipe all generated model artifacts (forces full upstream rebuild; preserves prime-tables in BQ)
	@read -p "DEEPCLEAN: clean targets PLUS $(AFC_M)/{standard,social}, $(AFC_O)/{standard,social}, NHANES cohort + nhanes.csv + SDI, copula_{inputs,outputs}/, bp_means_by_age_5yr.csv, out_of_system_*_betas.csv [y/N]? " ans && [ "$$ans" = "y" ] && rm -rf \
	    "results/$(FOLDER)" \
	    "validation_results/$(VALIDATION_FOLDER)" \
	    "$(VSTAMPS)" \
	    manuscript_draft.html \
	    manuscript_draft.docx \
	    manuscript_draft_files \
	    "$(AFC_M)/standard" "$(AFC_M)/social" \
	    "$(AFC_O)/standard" "$(AFC_O)/social" \
	    "$(NHANES)/NHANES_final_cohort.csv" \
	    "$(NHANES)/$(NHANES_FILE)_sdi.csv" \
	    "$(NHANES)/nhanes.csv" \
	    "$(NHINPUT)/bp_means_by_age_5yr.csv" \
	    "$(NHINPUT)/copula_inputs" \
	    "$(NHINPUT)/copula_outputs" \
	    "$(NHINPUT)/out_of_system_sys_quantile_betas.csv" \
	    "$(NHINPUT)/out_of_system_dia_quantile_betas.csv"
