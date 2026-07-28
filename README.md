# Predictive Maintenance Machine Learning

An end-to-end machine learning project for predicting industrial equipment failures using the **AI4I 2020 Predictive Maintenance Dataset**. The repository covers data validation, exploratory analysis, feature engineering, model comparison, hyperparameter tuning, decision-threshold selection, calibration, cost-sensitive evaluation, interpretability, subgroup analysis and reproducible artifact generation.

The project is designed as an example of a modular and reproducible binary-classification workflow built with Python, scikit-learn, XGBoost, pandas and Matplotlib.

## Table of Contents

- [Project Overview](#project-overview)
- [Key Results](#key-results)
- [Dataset](#dataset)
- [Machine Learning Workflow](#machine-learning-workflow)
- [Engineered Features](#engineered-features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Generated Outputs](#generated-outputs)
- [Model Evaluation](#model-evaluation)
- [Interpretability and Error Analysis](#interpretability-and-error-analysis)
- [Reproducibility](#reproducibility)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [License](#license)

## Project Overview

The objective is to predict the binary `Machine failure` target while accounting for the strong class imbalance in the dataset.

The workflow compares several baseline and imbalance-aware models:

- Dummy classifier
- Logistic Regression
- Balanced Logistic Regression
- L1-regularized Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- Class-weighted XGBoost

The final pipeline uses cross-validation for model comparison and hyperparameter tuning, out-of-fold predictions for threshold selection and a held-out test set for final evaluation.

Identifiers such as `UDI` and `Product ID` are excluded from model inputs.

## Key Results

The **Tuned Random Forest** was selected as the final model.

| Metric | Result |
|---|---:|
| Cross-validated Average Precision | 0.8937 |
| Test Average Precision | 0.876 |
| Test Precision | 0.947 |
| Test Recall | 0.794 |
| Test F1-score | 0.864 |
| Test F2-score | 0.821 |
| Selected decision threshold | 0.575 |
| Cost-minimizing threshold | 0.105 |

Bootstrap 95% confidence intervals:

| Metric | 95% confidence interval |
|---|---:|
| Average Precision | 0.798–0.942 |
| Recall | 0.687–0.886 |
| Precision | 0.881–1.000 |
| F1-score | 0.794–0.922 |

The final confusion matrix contained:

- 54 correctly detected failures
- 14 missed failures
- 3 false alarms
- 1,929 correctly identified non-failures

Hyperparameter tuning produced only a marginal improvement for Random Forest, from **0.8931** to **0.8937** cross-validated Average Precision. XGBoost improved from **0.8651** to **0.8812** but remained below the Random Forest.

## Dataset

The project uses the **AI4I 2020 Predictive Maintenance Dataset** from the UCI Machine Learning Repository.

The dataset contains 10,000 synthetic industrial-machine observations with six model-input features:

- Product type
- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear

The binary target is:

- `0`: no machine failure
- `1`: machine failure

The dataset also includes five failure-mode indicators used only for post-hoc analysis:

- TWF: Tool Wear Failure
- HDF: Heat Dissipation Failure
- PWF: Power Failure
- OSF: Overstrain Failure
- RNF: Random Failure

These failure-mode columns are excluded from training because they directly encode the target-generation mechanism.

## Machine Learning Workflow

The project follows this sequence:

1. Load the AI4I dataset from UCI.
2. Inspect the dataset schema and exclude identifier columns.
3. Validate missing values, duplicates, ranges and target integrity.
4. Perform exploratory data analysis.
5. Split the data into stratified training and test sets.
6. Add domain-informed engineered features.
7. Build separate preprocessing pipelines for linear and tree-based models.
8. Compare baseline models using stratified cross-validation.
9. Tune Random Forest and XGBoost using Average Precision.
10. Select the strongest model using cross-validated performance.
11. Generate out-of-fold probabilities on the training set.
12. Select a recall-constrained decision threshold.
13. Compare the selected threshold with a cost-minimizing threshold.
14. Evaluate the frozen model once on the held-out test set.
15. Estimate uncertainty with bootstrap confidence intervals.
16. Perform feature-importance, slice-performance, failure-mode and error analyses.
17. Save figures, tables, model artifacts and metadata.

## Engineered Features

Three interaction features are created from the original measurements.

### Power

Mechanical power is estimated from torque and rotational speed:

```python
rotational_speed_rad_s = rotational_speed_rpm * (2 * np.pi / 60)
power = torque * rotational_speed_rad_s
```

This feature captures low-power and high-power operating regions associated with power failures.

### Temperature difference

```python
temperature_difference = process_temperature - air_temperature
```

This represents the process-to-air temperature gap used by the heat-dissipation failure mechanism.

### Torque × Tool wear

```python
torque_x_tool_wear = torque * tool_wear
```

This interaction represents accumulated mechanical strain and supports overstrain-failure detection.

An ablation analysis compares raw features against raw plus engineered features.

## Repository Structure

```text
predictive-maintenance-ml/
├── artifacts/
│   ├── final_model.joblib
│   └── model_metadata.json
├── notebooks/
│   └── predictive_maintenance_ml_final.ipynb
├── reports/
│   ├── cli/
│   │   ├── figures/
│   │   └── tables/
│   └── notebook/
│       ├── figures/
│       └── tables/
├── src/
│   └── predictive_maintenance/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── data.py
│       ├── data_quality.py
│       ├── eda_visualization.py
│       ├── evaluation.py
│       ├── interpretability.py
│       ├── models.py
│       ├── plotting.py
│       ├── processing.py
│       └── reporting.py
├── tests/
├── .gitignore
├── LICENSE
├── pyproject.toml
└── README.md
```

Module responsibilities:

| Module | Responsibility |
|---|---|
| `data.py` | Dataset loading, schema extraction, target preparation and splitting |
| `data_quality.py` | Missing values, duplicate checks, range validation and class distribution |
| `processing.py` | Feature engineering and preprocessing pipelines |
| `models.py` | Candidate model construction, tuning and final model selection |
| `evaluation.py` | Cross-validation, threshold selection, calibration, cost analysis and test metrics |
| `interpretability.py` | Feature importance, subgroup evaluation, failure-mode analysis and error analysis |
| `eda_visualization.py` | Exploratory plots used by the notebook |
| `plotting.py` | Reusable model-evaluation and interpretability plots |
| `reporting.py` | CSV reports, confidence intervals, summaries and model artifacts |
| `cli.py` | End-to-end workflow orchestration |

## Installation

Clone the repository:

```bash
git clone https://github.com/LittleBigPluton/predictive-maintenance-ml
cd predictive-maintenance-ml
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

Install the project:

```bash
pip install -e .
```

Install the development dependencies when the optional development group is configured:

```bash
pip install -e ".[dev]"
```

## Usage

### Run the command-line workflow

```bash
predictive-maintenance
```

The command runs the complete workflow, including:

- data loading
- feature engineering
- cross-validation
- Random Forest and XGBoost tuning
- model selection
- threshold analysis
- calibration
- cost-sensitive evaluation
- final test evaluation
- bootstrap confidence intervals
- interpretability and error analysis
- report and artifact generation

### Run the notebook

Start Jupyter from the repository root:

```bash
jupyter lab
```

Open:

```text
notebooks/predictive_maintenance_ml_final.ipynb
```

The notebook contains the full analytical narrative, EDA, feature-ablation study, model evaluation, interpretation and limitations.

### Run tests

```bash
pytest -v
```

## Generated Outputs

The CLI and notebook generate reusable outputs rather than relying only on interactive notebook cells.

### Model artifacts

```text
artifacts/
├── final_model.joblib
└── model_metadata.json
```

The metadata file stores information such as:

- selected model name
- final decision threshold
- cost-minimizing threshold
- random seed
- test-set size
- final evaluation metrics

The decision threshold is saved separately because it is not part of the fitted scikit-learn pipeline.

### Reports

Generated tables include:

- cross-validation comparison
- tuning comparison
- threshold results
- calibration values
- expected-cost curve
- final test metrics
- classification report
- bootstrap confidence intervals
- permutation importance
- native feature importance
- product-type slice performance
- failure-mode breakdown
- false negatives
- false positives

Generated figures include:

- baseline model comparison
- precision-recall threshold trade-off
- calibration curve
- expected-cost curve
- confusion matrix
- ROC curve
- precision-recall curve
- permutation importance
- native feature importance

## Model Evaluation

Average Precision is used as the main model-selection metric because machine failures are rare and ranking positive observations is more informative than accuracy alone.

The final decision threshold is selected using out-of-fold training predictions under a minimum-recall requirement. This avoids choosing the threshold directly on the held-out test set.

A second threshold is obtained from an illustrative cost function. The cost-minimizing threshold of **0.105** is substantially lower than the recall-constrained threshold of **0.575**, showing how operational assumptions can change the preferred balance between missed failures and false alarms.

The final test set is used only after model selection and threshold determination.

## Interpretability and Error Analysis

Both native and permutation importance identify `Rotational speed` as the strongest predictor. The engineered features `Temperature difference`, `Torque x Tool wear` and `Power` also contribute meaningfully.

Performance varies by product type:

- Type H has the weakest recall at 0.600 and precision at 0.750, although it contains only five failures.
- Type M has the lowest slice Average Precision at 0.841.
- Type L provides more balanced performance.

Failure-mode recall is strongest for:

- OSF: 1.000
- HDF: 0.931
- PWF: 0.923

The weakest failure modes are:

- TWF: 1 of 10 cases detected
- RNF: 0 of 4 cases detected

These subgroup results are based on small samples and should be interpreted cautiously.

## Reproducibility

The project uses:

```text
RANDOM_STATE = 42
TEST_SIZE = 0.20
```

The train-test split is stratified by the target. Model comparison and tuning use stratified cross-validation.

The final tuned Random Forest parameters are:

```python
{
    "classifier__n_estimators": 400,
    "classifier__min_samples_split": 2,
    "classifier__min_samples_leaf": 1,
    "classifier__max_features": "log2",
    "classifier__max_depth": 15
}
```

The fitted model, selected threshold, metadata, figures and tables can be regenerated through the CLI.

## Limitations

- The AI4I dataset is synthetic and does not represent the full complexity of real industrial systems.
- Observations are treated as independent, while real predictive-maintenance data commonly contain machine-level and temporal dependencies.
- Failure-mode subgroup sizes are small, especially for RNF and Type H failures.
- The selected probability model is not perfectly calibrated.
- The cost values used for threshold analysis are illustrative rather than derived from a real maintenance operation.
- Feature importance describes predictive contribution, not causality.
- Test-set performance does not establish production readiness.

## License

This project is licensed under the [MIT License](LICENSE).
