# Hypertension Disease Natural History and Healthcare Utilization When Treatment Guidelines Do Not Account for Race

This repository contains code for the paper "Hypertension Disease Natural History and Healthcare Utilization When Treatment Guidelines Do Not Account for Race" by Marika Cusick, Malcolm Barrett, Fernando Alarid-Escudero, Douglas Owens, Jeremy Goldhaber-Fiebert, and Sherri Rose.

## Environment setup

### Python (uv)

Run the following command to set up the Python virtual environment with the necessary packages.

```{bash}
uv sync
```

### R (rig + renv)

Install R 4.5.3 with [rig](https://github.com/r-lib/rig), then restore packages from `renv.lock`.

```{bash}
rig add 4.5.3
rig default 4.5.3
```

```{r}
renv::restore()
```

### GNU Make (≥ 4.3)

The Makefile uses grouped-target syntax (`&:`), introduced in GNU Make 4.3 (Jan 2020). Older versions silently degrade and produce wrong results. Check with `make --version` or `gmake --version`.

On macOS, the system `make` is BSD make and usually older than 4.3. Install GNU Make via Homebrew (provided as `gmake`):

```{bash}
brew install make
```

Use `gmake` directly, or prepend Homebrew's gnubin to `PATH` so `make` resolves to GNU Make:

```{bash}
export PATH="/opt/homebrew/opt/make/libexec/gnubin:$PATH"  # Apple Silicon
# export PATH="/usr/local/opt/make/libexec/gnubin:$PATH"   # Intel
```

On Linux, if `make` is older than 4.3, update via your package manager (e.g. `apt install make`).

## Code

### Main results

The pipeline is orchestrated by `Makefile`; `make help` lists every target and `dependencies.md` is the stage-by-stage I/O contract. The full run covers our cohort of 100,000 individuals across eight scenarios (2×2 × two frameworks):
1) Heterogeneous treatment effects by race with race-adjusted treatment guidelines
2) Heterogeneous treatment effects by race with race neutral treatment guideelines
3) No heterogeneous treatment effects by race with race-adjusted treatment guidelines
4) No heterogeneous treatment effects by race with race neutral treatment guidelines

```{bash}
make pipeline
```

`make pipeline` runs every upstream stage, all 8 simulation scenarios, the validation arm, and the rendered reports. Add `-jN` (e.g. `make -j4 pipeline`) to run independent stages in parallel.

**First-time setup.** `make prime-tables prime-sample` rebuilds the PRIME BigQuery tables from `code/SQL/`. It is not a transitive prerequisite of any other target, so run it manually once per project setup, or after editing the SQL file.

In most cases `make pipeline` is all you need: it walks the full DAG, runs only what's stale, and produces the rendered reports. The sections below describe each stage in order so you can understand the analysis. Every stage also has its own `make <target>` if you ever need fine-grained control (re-running a single step after an edit, debugging a failure, skipping something), but that is the exception and not the expected day-to-day workflow.

Note that partial runs only control whether downstream targets are run; anything upstream that has changed will also run. Use the `-n` flag in `make` to see what make wants to run and `make --debug=b -n <target>` for additional information as to why it wants to run.

If `make -n` flags stages that you haven't actually edited (most often right after a fresh clone or a branch switch), run `make sync-mtimes`. Git doesn't preserve file mtimes across checkout, so sub-second write order can invert the build DAG even when file contents are unchanged. `make sync-mtimes` rewrites each tracked file's mtime to the time of its last commit and leaves any file with staged or working-tree changes alone.

For partial runs, the following convenience targets cover the most common subsets:

- `make pipeline-main` runs the full main analysis arm (both frameworks, all 8 simulations, and the upstream models that feed them) without the validation arm or the report render.
- `make pipeline-main-sim` runs only the cohort assembly and the 8 simulations from the checked-in PRIME-derived artifacts. This path does not require BigQuery access.
- `make pipeline-main-models` rebuilds the PRIME-derived models and outputs (quantile regressions, correlations, measurement frequency, treatment logit, SDI imputation, out-of-system BP). This path requires BigQuery access.
- `make pipeline-standard` and `make pipeline-social` run only one framework end to end.
- `make sim-all`, `make sim-standard`, and `make sim-social` run only the simulation step. Individual scenarios are also available, for example `make sim-standard-hetero-RA` and the seven sibling targets.

### Standard model

1.  We obtain our simulation cohort. Our simulation cohort is sampled from a multivariate copula fit to National Health and Nutrition Examination Survey (NHANES) data. The following quarto document fits the multivariate copula and sampled a cohort of 100,000 individuals (50% non-Hispanic Black and 50% non-Hispanic white) to a file `data_and_models/nhanes_data_cohorts/NHANES_final_cohort.csv`

```{bash}
make nhanes-copula
```

2.  We estimate quantile regressions to predict systolic and diastolic blood pressure values. In the code, we queried blood pressure measurement observations from the PRIME registry. Next, we estimated the prevalence of each hypertension stage (normal, elevated, stage 1, and stage 2) by age from the PRIME registry and saved as the file `data_and_models/afc_outputs/standard/AFC_BP_Stage_Prev.csv.` We then ran 5-fold cross validation to determine the optimal b-splines for age in predicting systolic and diastolic blood pressure. We saved the computed b-spline for each age into `data_and_models/afc_outputs/standard/systolic_splines.csv` and `data_and_models/afc_outputs/standard/diastolic_splines.csv.` Finally, we ran the quantile regressions to predict systolic and diastolic blood pressure as a function of age, and sex. We saved these regressions into the `data_and_models/afc_models/standard/starting_systolic_bp_regressions` and `data_and_models/afc_models/standard/starting_diastolic_bp_regressions` folders.

```{bash}
make quantile-regressions-standard
```

3.  We estimate correlations between age-sex-specific percentiles of systolic and diastolic blood pressure values. These correlations are used to generate lifetime systolic and diastolic quantiles for individuals in our simulated cohort. The correlations are exported into the `data_and_models/afc_outputs/standard/AFC_age_group_correlations` folder.

```{bash}
make correlations-standard
```

4.  We estimate quantile regressions to predict the number of months until the next blood pressure measurement as a function of age, sex, and systolic and diastolic blood pressure values. These quantile regressions are in the `data_and_models/afc_models/standard/measurement_models_pre_treatment` folder.

```{bash}
make measurement-standard
```

5.  We estimate a logistic regression to predict the probability of initiating anti-hypertensive treatment as a function of age, sex, and systolic and diastolic blood pressure values. This outputs the regression coefficients into the `data_and_models/afc_models/standard/treatment_models` folder.

```{bash}
make treatment-standard
```

6.  We set the simulated cohort to be run in the model. The following python script assigns initial starting systolic and diastolic blood pressure values, and lifetime systolic and diastolic quantiles (determines which quantile regression to predict next blood pressure value) for each individual. The script requires an input for `-f "{destination folder for output}`, which is created in the `results` folder. In addition, we must specify `-d "{NHANES file name}`, which is read from the `data_and_models/nhanes_data_cohorts` folder, and `-sff "{use of social factors framework}.`

```{bash}
make cohort-standard
```

7.  We can now run the simulation model. We run the model using the following python script to run the full 100,000 individual cohort. This file requires four arguments 1) -f cohort folder 2) -a heterogeneous treatment function (hetero or nohetero) 3) -ra race adjustment (RA) or no race adjustment (NR). The following code runs all four standard scenarios (individual targets like `sim-standard-hetero-RA` are available if you only need one).

```{bash}
make sim-standard
```

### Social factors framework

1.  We obtain our simulation cohort. Our simulation cohort is sampled from a multivariate copula fit to National Health and Nutrition Examination Survey (NHANES) data. The following quarto document fits the multivariate copula and sampled a cohort of 100,000 individuals (50% non-Hispanic Black and 50% non-Hispanic white) to a file `data_and_models/nhanes_data_cohorts/NHANES_final_cohort.csv`

```{bash}
make nhanes-copula
```

2.  We estimate quantile regressions to predict systolic and diastolic blood pressure values. In the code, we queried blood pressure measurement observations from the PRIME registry. Next, we estimated the prevalence of each hypertension stage (normal, elevated, stage 1, and stage 2) by age from the PRIME registry and saved as the file `data_and_models/afc_outputs/social/AFC_BP_Stage_Prev.csv.` We then ran 5-fold cross validation to determine the optimal b-splines for age in predicting systolic and diastolic blood pressure. We saved the computed b-spline for each age into `data_and_models/afc_outputs/social/systolic_splines.csv` and `data_and_models/afc_outputs/social/diastolic_splines.csv.`Finally, we ran the quantile regressions to predict systolic and diastolic blood pressure as a function of age, sex, and census tract-level social deprivation index (SDI). We saved these regressions into the `data_and_models/afc_models/social/starting_systolic_bp_regressions` and `data_and_models/afc_models/social/starting_diastolic_bp_regressions` folders.

```{bash}
make quantile-regressions-social
```

3.  We estimate correlations between age-sex-SDI-specific percentiles of systolic and diastolic blood pressure values. These correlations are used to generate lifetime systolic and diastolic quantiles for individuals in our simulated cohort. The correlations are exported into the `data_and_models/afc_outputs/social/AFC_age_group_correlations` folder.

```{bash}
make correlations-social
```

4.  We generate census tract-level SDI to the simulation cohort obtained from NHANES via the `code/R/NHANES_multivariate_copula.qmd` file. Because NHANES does not have geocoded location information, we assigned SDI using a Bayesian approach based on individual-level characteristics from NHANES. The following script runs the `code/python/SDI_code.py` file in increments and saves the file: `data_and_models/nhanes_data_cohorts/NHANES_final_cohort_sdi.csv`.

```{bash}
make sdi
```

5.  We estimate the effect of being out of the health system on blood pressure values. This outputs the regression coefficients for blood pressure adjustments into the `data_and_models/nhanes_inputs/` folder.

```{bash}
make out-of-system-bp
```

6.  We estimate quantile regressions to predict the number of months until the next blood pressure measurement as a function of age, sex, SDI, and systolic and diastolic blood pressure values. These quantile regressions are in the `data_and_models/afc_models/social/measurement_models_pre_treatment` folder.

```{bash}
make measurement-social
```

7.  We estimate a logistic regression to predict the probability of initiating anti-hypertensive treatment as a function of age, sex, SDI, and systolic and diastolic blood pressure values. This outputs the regression coefficients into the `data_and_models/afc_models/social/treatment_models` folder.

```{bash}
make treatment-social
```

8.  We set the simulated cohort to be run in the model. The following python script assigns the generated SDI values from the previous step, initial starting systolic and diastolic blood pressure values, and lifetime systolic and diastolic quantiles (determines which quantile regression to predict next blood pressure value) for each individual. The script requires an input for `-f "{destination folder for output}`, which is created in the `results` folder. In addition, we must specify `-d "{NHANES file name}`, which is read from the `data_and_models/nhanes_data_cohorts` folder, and  and `-sff "{use of social factors framework}.`

```{bash}
make cohort-social
```

9.  We can now run the simulation model. We run the model using the following python script to run the full 100,000 individual cohort. This file requires four arguments 1) -f cohort folder 2) -a heterogeneous treatment function (hetero or nohetero) 3) -ra race adjustment (RA) or no race adjustment (NR). The following code runs all four SFF scenarios (individual targets like `sim-social-hetero-RA` are available if you only need one).

```{bash}
make sim-social
```

### Validation Model

Our validation model simulates individuals sampled directly from the AFC PRIME registry data assuming no treatment effects. Because the cohort is sampled directly from the AFC PRIME registry, outputs from this code are not available on this GitHub repository.

The full validation arm (cohort sampling, both simulation halves, and post-processing) runs end-to-end via a single target. Override `VALIDATION_FOLDER` and `VALIDATION_SIZE` on the command line if needed (defaults: `AFC_Validation_Cohort`, `10000`).

```{bash}
make pipeline-validation
```

For fine-grained control, the validation arm decomposes into four sub-targets that can be run in isolation when debugging or re-running a single step:

- `make validation-cohort` samples the validation cohort from PRIME.
- `make validation-sim-1` and `make validation-sim-2` run the two halves of the validation simulation. The split exists because the non-vectorized validation driver runs in two passes via the `-n 1|2` argument.
- `make validation-analysis` post-processes the simulation outputs into the JSON and CSV files that the manuscript reads.

### Reports

`make pipeline` renders the manuscript at the end. Since the report is downstream of everything, running its subtarget is equivalent to running the entire pipeline.

```{bash}
make reports             # same as render-manuscript below
make render-manuscript   # manuscript_draft.qmd    -> manuscript_draft.html
```

### Development

The Makefile also exposes the linting, formatting, and cleanup targets used by CI and during local development.

- `make lint` runs `ruff check .` and matches what CI does on every push.
- `make format` applies `ruff format .` to the codebase.
- `make format-check` reports formatting violations without modifying files. CI uses this in addition to `lint`.
- `make ci-check` runs both `format-check` and `lint` together, mirroring the full CI run locally.

For removing generated outputs:

- `make clean-results` deletes `results/$(FOLDER)` after a confirmation prompt.
- `make clean-validation` deletes `validation_results/$(VALIDATION_FOLDER)` after a confirmation prompt.
- `make clean` wipes both of the above plus the rendered reports (`manuscript_draft.html` and the associated `_files/` directory).
- `make deepclean` runs `clean` and additionally wipes all generated model artifacts, forcing a full upstream rebuild on the next `make pipeline`. The PRIME tables in BigQuery are not touched.
