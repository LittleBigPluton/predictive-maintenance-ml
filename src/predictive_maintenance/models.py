from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

from .config import RANDOM_STATE, RF_SEARCH_SPACE, XGB_SEARCH_SPACE


def compute_scale_pos_weight(target_data_train):
    negative_count = (target_data_train == 0).sum()
    positive_count = (target_data_train == 1).sum()
    return negative_count / positive_count


def build_models(tree_preprocessor, linear_preprocessor, scale_pos_weight):
    return {
        "Dummy Baseline": Pipeline([("preprocessor", tree_preprocessor),
                                    ("classifier", DummyClassifier(strategy="prior", random_state=RANDOM_STATE))]),
        "Logistic Regression": Pipeline([("preprocessor", linear_preprocessor),
                                         ("classifier", LogisticRegression(random_state=RANDOM_STATE, max_iter=1000))]),
        "Balanced Logistic Regression": Pipeline([("preprocessor", linear_preprocessor),
                                                  ("classifier",LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, class_weight="balanced"))]),
        "L1 Logistic Regression": Pipeline([("preprocessor", linear_preprocessor),
                                            ("classifier",LogisticRegression(l1_ratio=1.0, solver="liblinear", class_weight="balanced", random_state=RANDOM_STATE, max_iter=1000))]),
        "Decision Tree": Pipeline([("preprocessor", tree_preprocessor),
                                   ("classifier",DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=RANDOM_STATE))]),
        "Random Forest": Pipeline([("preprocessor", tree_preprocessor),
                                   ("classifier",RandomForestClassifier(n_estimators=300,class_weight="balanced",random_state=RANDOM_STATE,n_jobs=1))]),
        "XGBoost": Pipeline([("preprocessor", tree_preprocessor),
                             ("classifier",XGBClassifier(objective="binary:logistic",n_estimators=300,learning_rate=0.05,max_depth=4,min_child_weight=1,subsample=0.8,colsample_bytree=0.8,
                                                         scale_pos_weight=1,eval_metric="logloss",random_state=RANDOM_STATE,n_jobs=1))]),
        "Balanced XGBoost": Pipeline([("preprocessor", tree_preprocessor),
                                      ("classifier",XGBClassifier(objective="binary:logistic",n_estimators=300,learning_rate=0.05,max_depth=4,min_child_weight=1,subsample=0.8,colsample_bytree=0.8,
                                                                 scale_pos_weight=scale_pos_weight,eval_metric="logloss",random_state=RANDOM_STATE,n_jobs=1))])
    }


def tune_random_forest(candidate_models, feature_data_train, target_data_train, cv, n_iter=20):
    search = RandomizedSearchCV(estimator=candidate_models["Random Forest"],param_distributions=RF_SEARCH_SPACE,n_iter=n_iter,scoring="average_precision",cv=cv,n_jobs=-1,random_state=RANDOM_STATE)
    search.fit(feature_data_train, target_data_train)
    return search


def tune_xgboost(candidate_models, feature_data_train, target_data_train, cv, n_iter=30):
    search = RandomizedSearchCV(estimator=candidate_models["XGBoost"],param_distributions=XGB_SEARCH_SPACE,n_iter=n_iter,scoring="average_precision",cv=cv,n_jobs=-1,random_state=RANDOM_STATE,verbose=1,refit=True)
    search.fit(feature_data_train, target_data_train)
    return search


def select_final_model(search_results):
    final_model_name = max(search_results, key=lambda name: search_results[name].best_score_)
    final_model = search_results[final_model_name].best_estimator_
    return final_model_name, final_model
