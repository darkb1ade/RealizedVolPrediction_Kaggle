from model import BaseModel
# Visullize
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Modeling
import lightgbm as lgb
from helper import utils

class LGBM(BaseModel):
    def __init__(self):
        # Parameters of Light GBM
        self.params_lgbm = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'learning_rate': 0.01,
            'objective': 'regression',
            'metric': 'None',
            'max_depth': -1,
            'n_jobs': -1,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'lambda_l2': 1,
            'verbose': -1
            # 'bagging_freq': 5
        }

        # k-flods Ensemble Training
        self.n_folds = 4
        self.n_rounds = 10000

    # Define loss function for lightGBM training
    def feval_RMSPE(self, preds, train_data):
        labels = train_data.get_label()
        return 'RMSPE', round(utils.rmspe(y_true=labels, y_pred=preds), 5), False

    def train(self, X_train, y_train, X_val ,y_val):

        # Get feature name (Not sure if this work), might put this in __init__
        features = [col for col in X_train.columns if col not in {"time_id", "target", "row_id"}]
        # features = ['stock_id', 'log_return1', 'log_return2', 'trade_log_return1']

        # Create dataset
        train_data = lgb.Dataset(X_train, label=y_train, weight=1/np.power(y_train,2))
        val_data = lgb.Dataset(X_val, label=y_val, weight=1/np.power(y_val,2))
        # val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=cats, weight=1/np.power(y_val,2))

        # training
        model = lgb.train(params=self.params_lgbm ,
                          num_boost_round=1300,
                          train_set=train_data,
                          valid_sets=[train_data, val_data],
                          verbose_eval=250,
                          early_stopping_rounds=50,
                          feval=self.feval_RMSPE)

        # Prediction w/ validation data
        preds_val = model.predict(X_val[features])
        # train.loc[val_index, pred_name] = preds_val

        # RMSPE calculation
        score = round(utils.rmspe(y_true = y_val, y_pred = preds_val),5)



        # delete dataset
        del train_data, val_data

        return score, test_preds, model

    def test(self, model, test_input):
        # Get feature name (Not sure if this work), might put this in __init__
        features = [col for col in test_input.columns if col not in {"time_id", "target", "row_id"}]
        # features = ['stock_id', 'log_return1', 'log_return2', 'trade_log_return1']

        # Prediction w/ validation data
        test_preds = model.predict(test_input[features]).clip(0, 1e10)