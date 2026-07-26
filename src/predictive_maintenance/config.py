# --- Reproducibility / split -------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20

# --- Cross-validation ---------------------------------------------------------
N_SPLITS = 5

# --- Hyperparameter search spaces (Hyperparameter Tuning section) ------------
RF_SEARCH_SPACE = {
    "classifier__n_estimators": [200, 400, 600],
    "classifier__max_depth": [None, 5, 10, 15],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
    "classifier__max_features": ["sqrt", "log2"],
}

XGB_SEARCH_SPACE = {
    "classifier__n_estimators": [150, 250, 350, 500],
    "classifier__max_depth": [2, 3, 4, 5, 6],
    "classifier__learning_rate": [0.01, 0.03, 0.05, 0.1],
    "classifier__min_child_weight": [1, 3, 5, 7],
    "classifier__subsample": [0.7, 0.8, 0.9, 1.0],
    "classifier__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "classifier__gamma": [0, 0.1, 0.3, 0.5],
    "classifier__reg_alpha": [0, 0.01, 0.1, 0.5],
    "classifier__reg_lambda": [1, 2, 5, 10],
}

# --- Decision threshold (Threshold and Calibration section) ------------------
# Recall floor used when selecting the operating threshold from training-set OOF predictions.
MIN_RECALL = 0.80

# --- Business-cost trade-off (Threshold and Calibration section) -------------
# Illustrative only. Replace with real maintenance/downtime budget figures before
# using the cost-minimizing threshold for an actual deployment decision.
COST_FALSE_NEGATIVE = 5000  # missed failure: unplanned downtime + potential damage
COST_FALSE_POSITIVE = 150  # false alarm: technician inspection time
 
