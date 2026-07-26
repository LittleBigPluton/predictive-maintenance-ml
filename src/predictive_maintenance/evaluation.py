import pandas as pd
from sklearn.metrics import (
    f1_score,
    fbeta_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
)

from .config import N_SPLITS, RANDOM_STATE

CV_SCORING = {"average_precision": "average_precision", "roc_auc": "roc_auc", "precision": "precision", "recall": "recall", "f1": "f1", "f2": make_scorer(fbeta_score, beta=2, zero_division=0)}

def make_cv(random_state=RANDOM_STATE, n_splits=N_SPLITS):
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def cross_validate_models(candidate_models, feature_data_train, target_data_train, cv):
    rows = []

    for model_name, model_pipeline in candidate_models.items():
        scores = cross_validate(model_pipeline, feature_data_train, target_data_train, cv=cv, scoring=CV_SCORING, n_jobs=-1)
        rows.append({"Model": model_name,"CV Average Precision": scores["test_average_precision"].mean(),"CV Average Precision Std": scores["test_average_precision"].std(),"CV ROC-AUC": scores["test_roc_auc"].mean(),
                                         "CV Precision": scores["test_precision"].mean(),"CV Recall": scores["test_recall"].mean(),"CV F1": scores["test_f1"].mean(),"CV F2": scores["test_f2"].mean()})

    return (pd.DataFrame(rows).sort_values("CV Average Precision",ascending=False).reset_index(drop=True))
