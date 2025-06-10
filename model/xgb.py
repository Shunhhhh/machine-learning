from hyperopt import fmin, tpe, hp, Trials
from xgboost import XGBClassifier, train as xgb_train
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np


class XGBoostOptimizer:
    def __init__(self, X_train, y_train, dtrain, dval, y_val, cv=10, max_evals=200):
        self.X_train = X_train
        self.y_train = y_train
        self.dtrain = dtrain
        self.dval = dval
        self.y_val = y_val
        self.cv = cv
        self.max_evals = max_evals
        self.best_params = None
        self.bst = None

    def _objective(self, params):
        params.update({'device': 'cuda'})
        model = XGBClassifier(**params)
        cv_results = cross_val_score(model, self.X_train, self.y_train, cv=self.cv, scoring='roc_auc')
        return -cv_results.mean()  # 最小化负AUC

    def search_best_params(self):
        space = {
            'max_depth': hp.randint('max_depth', 4, 7),
            'learning_rate': hp.uniform('learning_rate', 0.1, 1),
            'gamma': hp.uniform('gamma', 0, 5),
            'subsample': hp.uniform('subsample', 0.5, 1),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
            'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
            'tree_method': 'hist',
            'device': 'cuda'
        }

        trials = Trials()
        best_params = fmin(fn=self._objective,
                           space=space,
                           algo=tpe.suggest,
                           max_evals=self.max_evals,
                           trials=trials)

        print("Best parameters found:", best_params)
        self.best_params = best_params
        return best_params

    def train_best_model(self):
        if self.best_params is None:
            raise ValueError("请先调用 search_best_params 找到最佳参数")

        # 添加额外参数
        final_params = self.best_params.copy()
        final_params.update({
            'tree_method': 'hist',
            'device': 'cuda:1',
            'objective': 'binary:logistic',
            'eval_metric': 'logloss'
        })

        bst = xgb_train(
            params=final_params,
            dtrain=self.dtrain,
            num_boost_round=200,
            early_stopping_rounds=100,
            evals=[(self.dval, 'eval')],
            verbose_eval=10
        )

        self.bst = bst
        print("Best iteration:", bst.best_iteration)
        return bst

    def evaluate_model(self):
        if self.bst is None:
            raise ValueError("请先调用 train_best_model 训练模型")

        y_pred_proba = self.bst.predict(self.dval, iteration_range=(0, self.bst.best_iteration + 1))
        y_pred = (y_pred_proba > 0.5).astype(int)

        auc = roc_auc_score(self.y_val, y_pred_proba)
        accuracy = accuracy_score(self.y_val, y_pred)

        print("Validation AUC:", auc)
        print("Validation Accuracy:", accuracy)

        return {'auc': auc, 'accuracy': accuracy}

    def get_best_model(self):
        return self.bst

    def get_best_params(self):
        return self.best_params