"""Centralized BigQuery configuration.

Credentials come from ADC (~/.config/gcloud/application_default_credentials.json);
set GOOGLE_APPLICATION_CREDENTIALS to override.
"""

from google.cloud import bigquery

# Billing project for SELECT-only scripts (6 files use this via bq_client()).
BILLING_PROJECT_MIC = "som-nero-phi-sherrir-afc-mic"

# Billing project for writes (build_prime_tables) and the treatment-logit query.
BILLING_PROJECT = "som-nero-phi-sherrir-afc"

# Project where the PRIME-derived tables live (used inside query strings).
DATA_PROJECT = "som-nero-phi-sherrir-afc"

# Dataset name (used inside query strings).
DATASET = "mmcusick_afc"


def bq_client(billing_project: str | None = None) -> bigquery.Client:
    """Return a BigQuery client. Defaults to the -mic billing project."""
    return bigquery.Client(project=billing_project or BILLING_PROJECT_MIC)


def table(name: str) -> str:
    """Return fully-qualified `project.dataset.name` (callers add backticks if wanted)."""
    return f"{DATA_PROJECT}.{DATASET}.{name}"
