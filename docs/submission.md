# Submission Evidence and Rubric Mapping

The course announcement permits GitHub + GitHub Actions + Streamlit Community Cloud as an equivalent end-to-end workflow. This project deliberately uses that alternative; Hugging Face-specific rubric steps map to the outcomes below.

| Criterion | Points | Equivalent implementation and evidence |
|---|---:|---|
| Data registration | 3 | Master repository with `data/tourism.csv`; Git commit history and `data/manifest.json` register the snapshot and SHA-256. |
| Data preparation | 7 | Shared data module loads the registered checkout, validates/cleans, removes IDs/duplicates, saves group-exclusive train/test CSVs and fold manifest; Actions publishes outputs. |
| Model building and tracking | 13 | Random Forest/XGBoost grids for both stages; MLflow child runs log every parameter combination; full CV tables, selected thresholds and held-out reports; best models registered in GitHub Releases. |
| Deployment | 11 | Dockerfile, pinned deployment requirements, model-registry download/integrity checks, Streamlit DataFrame input, and `scripts/deploy.py` hosting/publishing script. |
| GitHub Actions MLOps | 15 | `.github/workflows/pipeline.yml`: checks, preparation, tuning, evaluation, notebook, Docker verification, versioned release, automatic generated main updates, and tested code promotion to `streamlit`. |
| Output evaluation | 4 | Repository/workflow/app links plus genuine screenshots listed below. Live evidence remains pending until publication and browser setup succeed. |
| Notebook quality | 7 | Structured EDA and explanation, commented shared pipeline calls, executed model/search results and output evaluation; reproducible repository layout. |
| **Total** | **60** | **Platform substitution follows the supplied course clarification.** |

## Links

- Configured GitHub repository: https://github.com/zfaida459-sudo/wellness_tourism_package_system
- Workflow page: https://github.com/zfaida459-sudo/wellness_tourism_package_system/actions/workflows/pipeline.yml
- Model releases: https://github.com/zfaida459-sudo/wellness_tourism_package_system/releases
- Successful workflow run: **Pending first published run.**
- Live Streamlit app: **Pending one-time Community Cloud connection and deployment.**

The workflow/release links are expected destinations, not evidence that the new implementation has already been pushed or executed on GitHub.

## Required Screenshots

- **Pending:** actual GitHub repository folder structure after source publication.
- **Pending:** successful full Actions run with all stage results visible.
- **Pending:** live Streamlit app showing a prediction and its model release; capture both stages when possible.

Place genuine captured images under `docs/screenshots/` and link them here after deployment. Never substitute a mock image, local screenshot or fabricated URL for required live evidence. The local app can be captured separately as development evidence.

## Initial Local Evaluation

4,011 cleaned records; 3,208 development rows and 803 test rows. Thresholds were selected on validation data, not the test labels.

| Stage | Model | Threshold | Test F1 | Test recall | Test average precision |
|---|---|---:|---:|---:|---:|
| Pre-contact | Random Forest | 0.32 | 0.719 | 0.741 | 0.768 |
| Post-interaction | Random Forest | 0.28 | 0.754 | 0.844 | 0.861 |

Historical tourism labels may not generalize to a new wellness offering. Data lacks event timing and does not establish causal marketing effects. Public sharing is limited to the authorized course dataset and project artifacts; uploaded app customer data is never saved.