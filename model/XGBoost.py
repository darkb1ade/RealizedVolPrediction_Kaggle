from model.BaseModel import BaseModel
# Visuallize
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn import model_selection

# Modeling
from xgboost import XGBRegressor
from helper import utils
import optuna


class LGBM(BaseModel):
    """
    This is lightGBM class. Insert list of str contain feature column here.
    Output column should be named "output"
    To run, use lgbm.train_and_test() <- recommend
    """
    def __init__(self, feature_column=None):
        super().init()
        # Parameters of Light GBM
        self.param = {
            'tree_method': 'gpu_hist',
            'lambda': 1,
            'alpha': 1,
            'colsample_bytree': 1.0,
            'subsample': 1.0,
            'learning_rate': 0.01,
            'n_estimators': 1000,
            'max_depth': 20,
            'random_state': 2020,
            'min_child_weight': 300
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
        if feature_column is None:
            self.features = ['stock_id', 'log_return1', 'log_return2', 'trade_log_return1']  # Need to change
        else:
            self.features = feature_column
        self.output_feature = "output"

    # Define loss function for lightGBM training
    def feval_RMSPE(self, preds, train_data):
        labels = train_data.get_label()
        return 'RMSPE', round(utils.rmspe(y_true=labels, y_pred=preds), 5), False

    def train(self, X_train, y_train, X_val, y_val, param=None):
        if param is None:
            param = self.param
        # training
        model = XGBRegressor(**param)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=False)

        # Prediction w/ validation data
        preds_val = model.predict(X_val[self.features])
        # train.loc[val_index, pred_name] = preds_val

        # RMSPE calculation
        score = round(utils.rmspe(y_true=y_val, y_pred=preds_val), 5)


        return score, preds_val, model

    def test(self, model, test_input):
        # Prediction w/ validation data
        test_preds = model.predict(test_input[self.features]).clip(0, 1e10)
        return test_preds

    # Combine train and test, with KFold CV
    def train_and_test(self, train_input, test_input, param=None):
        """

        :param train_input: pd array. Contain both feature data and "output" data
        :param test_input: pd array. Contain feature data to test
        :return: test_prediction (??): predicted output data of 'test_input'
        """
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
            y_train = train_input.loc[train_index, self.output_feature].values
            X_val = train_input.loc[val_index, self.features]
            y_val = train_input.loc[val_index, self.output_feature].values

            score, preds_val, model = self.train(X_train, y_train, X_val, y_val, param)
            test_preds = self.test(model, test_input)
            oof_predictions[val_index] = preds_val
            test_predictions += test_preds / self.n_folds
            cv_trial += 1

        rmspe_score = self.rmspe(train_input[self.output_feature], oof_predictions)
        print(f'Our out of folds RMSPE is {rmspe_score}')

        return test_predictions, rmspe_score

    def optimize_params(self, train_input, test_input):
        def objective(trial):
            param = {
                'tree_method': 'gpu_hist',
                'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
                'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
                'colsample_bytree': trial.suggest_categorical('colsample_bytree',
                                                              [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
                'subsample': trial.suggest_categorical('subsample', [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]),
                'learning_rate': trial.suggest_categorical('learning_rate',
                                                           [0.008, 0.009, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02]),
                'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
                'max_depth': trial.suggest_categorical('max_depth', [5, 7, 9, 11, 13, 15, 17, 20]),
                'random_state': trial.suggest_categorical('random_state', [24, 48, 2020]),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 300)}

            # params = {"objective": "reg:squarederror",
            #           "eval_metric": "rmse",
            #           "tree_method": "hist",
            #           "grow_policy": "lossguide",
            #           'silent': 1,
            #           "seed": 1,
            #           "colsample_bytree": 1,
            #           "subsample": 1,
            #           'max_leaves': 31,  # lossguideの場合、当該項目の設定が必要（default: 0）
            #           "max_depth": trial.suggest_int('max_depth', 2, 12),
            #           "eta": trial.suggest_loguniform('eta', 10e-2, 1),
            #           "alpha": trial.suggest_uniform('alpha', 0.0, 1.0),
            #           "lambda": trial.suggest_uniform('lambda', 0.0, 1.0)}
            print(params)
            _, score = self.train_and_test(train_input,test_input,params)
            return score

        opt = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.RandomSampler(seed=1))
        opt.optimize(objective,
                     n_trials=self.ntrial)
        trial = opt.best_trial
        params = self.params_lgbm.copy()
        # params.update(**params, **trial.params)
        for key, value in trial.params.items():
            print('"{}" : {}'.format(key, value))
            params[key] = value

        test_predictions , score = self.train_and_test(train_input,test_input,params)
        return test_predictions, score


if __name__ == "__main__":
    lgbm = LGBM()