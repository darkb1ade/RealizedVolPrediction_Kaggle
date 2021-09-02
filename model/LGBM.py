from model import BaseModel
# Visuallize
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn import model_selection

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

        # seed0 = 2000
        # self.params_lgbm = {
        #     'objective': 'rmse',
        #     'boosting_type': 'gbdt',
        #     'max_depth': -1,
        #     'max_bin': 100,
        #     'min_data_in_leaf': 500,
        #     'learning_rate': 0.05,
        #     'subsample': 0.72,
        #     'subsample_freq': 4,
        #     'feature_fraction': 0.5,
        #     'lambda_l1': 0.5,
        #     'lambda_l2': 1.0,
        #     'categorical_column': [0],
        #     'seed': seed0,
        #     'feature_fraction_seed': seed0,
        #     'bagging_seed': seed0,
        #     'drop_seed': seed0,
        #     'data_random_seed': seed0,
        #     'n_jobs': -1,
        #     'verbose': -1}

        # k-folds Ensemble Training
        self.n_folds = 5
        self.n_rounds = 10000

        # Get feature name (Not sure if this work), Data must have column name of following features and output is 'target'
        # features = [col for col in X_train.columns if col not in {"time_id", "target", "row_id"}]
        self.features = ['stock_id', 'log_return1', 'log_return2', 'trade_log_return1']  # Need to change

    # Define loss function for lightGBM training
    def feval_RMSPE(self, preds, train_data):
        labels = train_data.get_label()
        return 'RMSPE', round(utils.rmspe(y_true=labels, y_pred=preds), 5), False

    def train(self, X_train, y_train, X_val, y_val):
        # Create dataset
        train_data = lgb.Dataset(X_train, label=y_train, weight=1 / np.power(y_train, 2))
        val_data = lgb.Dataset(X_val, label=y_val, weight=1 / np.power(y_val, 2))
        # val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=cats, weight=1/np.power(y_val,2))

        # training
        model = lgb.train(params=self.params_lgbm,
                          num_boost_round=1300,
                          train_set=train_data,
                          valid_sets=[train_data, val_data],
                          verbose_eval=250,
                          early_stopping_rounds=50,
                          feval=self.feval_RMSPE)

        # Prediction w/ validation data
        preds_val = model.predict(X_val[self.features])
        # train.loc[val_index, pred_name] = preds_val

        # RMSPE calculation
        score = round(utils.rmspe(y_true=y_val, y_pred=preds_val), 5)

        # delete dataset
        del train_data, val_data

        return score, preds_val, model

    def test(self, model, test_input):
        # Get feature name (Not sure if this work), might put this in __init__
        features = [col for col in test_input.columns if col not in {"time_id", "target", "row_id"}]
        # features = ['stock_id', 'log_return1', 'log_return2', 'trade_log_return1']

        # Prediction w/ validation data
        test_preds = model.predict(test_input[features]).clip(0, 1e10)
        return test_preds

    # Combine train and test, with KFold CV
    def train_and_test(self, train_input, test_input):
        cv_trial = 1
        kf = model_selection.KFold(n_splits=self.n_folds, shuffle=True, random_state=15)
        # Create out of folds array
        oof_predictions = np.zeros(train_input.shape[0])
        # Create test array to store predictions
        test_predictions = np.zeros(test_input.shape[0])

        for train_index, val_index in kf.split(range(len(train_input))):
            print(f'CV trial : {cv_trial} /{self.n_folds}')

            # Divide dataset into train and validation data such as Cross Validation
            X_train = train_input.loc[train_index, self.features]
            y_train = train_input.loc[train_index, 'target'].values
            X_val = train_input.loc[val_index, self.features]
            y_val = train_input.loc[val_index, 'target'].values

            score, preds_val, model = self.train(X_train, y_train, X_val, y_val)
            test_preds = self.test(model, test_input)
            oof_predictions[val_index] = preds_val
            test_predictions += test_preds / self.n_folds
            cv_trial += 1

        rmspe_score = self.rmspe(train_input['target'], oof_predictions)
        print(f'Our out of folds RMSPE is {rmspe_score}')
        lgb.plot_importance(model, max_num_features=20)

        return test_predictions
