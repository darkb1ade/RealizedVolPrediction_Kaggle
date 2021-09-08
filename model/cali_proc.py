import lightgbm as lgb
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from sklearn.model_selection import KFold
import optuna
from sklearn.metrics import mean_squared_error
import os
import matplotlib.pyplot as plt
import logging

def straight_throught_forecast(x_train, y_train, x_test, y_test,conf): #conf = config['interp']['model']
    os.makedirs('train_log', exist_ok=True)
    logging.basicConfig(filename=f"train_log/xgb_train.log", filemode='w',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s: %(message)s', level=logging.INFO)


    log = logging.getLogger()
    NFOLD = conf['NFOLD']
    NTRIAL = conf['NTRIAL']
    def objective(trial):
        # ==== 定義パラメータ ====
        params = {"objective": "reg:squarederror",
                  "eval_metric": "rmse",
                  "tree_method": "hist",
                  "grow_policy": "lossguide",
                  'silent': 1,
                  "seed": 1,
                  "colsample_bytree": 1,
                  "subsample": 1,
                  'max_leaves': 31,  # lossguideの場合、当該項目の設定が必要（default: 0）
                  "max_depth": trial.suggest_int('max_depth', 2, 12),
                  "eta": trial.suggest_loguniform('eta', 10e-2, 1),
                  "alpha": trial.suggest_uniform('alpha', 0.0, 1.0),
                  "lambda": trial.suggest_uniform('lambda', 0.0, 1.0)}
        if int(0.95 * 2 ** params['max_depth']) > 31:
            params['max_leaves'] = int(0.95 * 2 ** params['max_depth'])
        print(params)

        scores = 0
        folds = KFold(n_splits=NFOLD, shuffle=True)

        for fold_n, (train_index, test_index) in enumerate(folds.split(x_train)):
            dtrain = xgb.DMatrix(data=x_train[train_index], label=y_train[train_index])
            dvalid = xgb.DMatrix(data=x_train[test_index], label=y_train[test_index])

            xgb_reg = xgb.train(params,
                                dtrain=dtrain,
                                num_boost_round=10 ** 6,
                                early_stopping_rounds=100,
                                evals=[(dvalid, 'validation')],
                                verbose_eval=10 ** 3)
            score = xgb_reg.best_score
            print("{}Fold: {}".format(fold_n,
                                      score))
            log.info("{}Fold: {}".format(fold_n,
                                      score))
            scores += score / NFOLD
        print("RMSE:", scores)
        return scores

    # scoreの最大化は"maximize"。最小化の場合は"minimize"
    opt = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.RandomSampler(seed=1))
    opt.optimize(objective,
                 n_trials=NTRIAL)
    trial = opt.best_trial
    params = {"objective": "reg:squarederror",
              "eval_metric": "rmse",
              "tree_method": "hist",
              "grow_policy": "lossguide",
              'silent': 1,
              "seed": 1,
              "colsample_bytree": 1,
              "subsample": 1,
              "max_depth": 0,
              "eta": 0,
              "alpha": 0,
              "lambda": 0, }
    # params.update(**params, **trial.params)
    for key, value in trial.params.items():
        print('"{}" : {}'.format(key, value))
        params[key] = value
    if int(0.95 * 2 ** params['max_depth']) > 31:
        params['max_leaves'] = int(0.95 * 2 ** params['max_depth'])

    folds = KFold(n_splits=NFOLD, shuffle=True)
    train_index, valid_index = list(folds.split(x_train))[-1]
    dtrain = xgb.DMatrix(data=x_train[train_index], label=y_train[train_index])
    dvalid = xgb.DMatrix(data=x_train[valid_index], label=y_train[valid_index])

    _xgb_reg = xgb.train(
        params,
        dtrain=dtrain,
        num_boost_round=10 ** 6,
        early_stopping_rounds=100,
        evals=[(dvalid, 'last_valid'), ],
        verbose_eval=1000)

    xgb_reg = xgb.train(params,
                        xgb.DMatrix(data=x_train,
                                    label=y_train),
                        num_boost_round=_xgb_reg.best_ntree_limit)

    predicted = xgb_reg.predict(xgb.DMatrix(data=x_test,
                                            label=y_test))
    # 結果出力
    pred_df = pd.DataFrame(y_test)
    pred_df['pred'] = predicted
    pred_df.columns = ['true', 'pred']
    return pred_df, xgb_reg, params, opt.best_trial

def straight_throught_forecast_lgb(X_train, y_train, X_test, y_test,conf): #conf = config['interp']['model']
    os.makedirs('train_log', exist_ok=True)
    logging.basicConfig(filename=f"train_log/xgb_train.log", filemode='w',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s: %(message)s', level=logging.INFO)


    log = logging.getLogger()
    NFOLD = conf['NFOLD']
    NTRIAL = conf['NTRIAL']
    def objective(trial):
        # ==== 定義パラメータ ====
        params = {
            'verbose': -1,
            'objective': 'regression',
            'metric': 'rmse',
            'max_bin': trial.suggest_int('max_bin', 1, 512),
            'num_leaves': trial.suggest_int('num_leaves', 2, 512),

            'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
            'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),

            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 50),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
            "max_depth": trial.suggest_int('max_depth', 2, 12),

            'sub_feature': trial.suggest_uniform('sub_feature', 0.0, 1.0),
            'sub_row': trial.suggest_uniform('sub_row', 0.0, 1.0)
        }
        print(params)

        scores = 0
        folds = KFold(n_splits=NFOLD, shuffle=True)

        for fold_n, (train_index, test_index) in enumerate(folds.split(X_train)):
            dtrain = lgb.Dataset(X_train[train_index], label=y_train[train_index])
            dvalid = lgb.Dataset(X_train[test_index], label=y_train[test_index])

            booster = lgb.train(params,
                                dtrain,
                                num_boost_round=10 ** 6,
                                early_stopping_rounds=100,
                                valid_sets=dvalid,
                                verbose_eval=10 ** 3)

            preds = booster.predict(X_train[test_index])
            score = mean_squared_error(preds,y_train[test_index])
            scores += score / NFOLD

        return scores

    opt = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.RandomSampler(seed=1))
    opt.optimize(objective,
                 n_trials=NTRIAL)

    trial = opt.best_trial
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': ['rmse'],
        'learning_rate': 0.005,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.7,
        'bagging_freq': 10,
        'verbose': -1,
        "max_depth": 8,
        "num_leaves": 128,  
        "max_bin": 512,
        "num_iterations": 1000,
        "n_estimators": 1000
    } 
    # params.update(**params, **trial.params)
    for key, value in trial.params.items():
        print('"{}" : {}'.format(key, value))
        params[key] = value
    if int(0.95 * 2 ** params['max_depth']) > 31:
        params['max_leaves'] = int(0.95 * 2 ** params['max_depth'])

    folds = KFold(n_splits=NFOLD, shuffle=True)
    train_index, valid_index = list(folds.split(X_train))[-1]
    dtrain = lgb.Dataset(X_train[train_index], label=y_train[train_index])
    dvalid = lgb.Dataset(X_train[valid_index], label=y_train[valid_index])

    gbm_reg = lgb.train(
        params,
        dtrain,
        num_boost_round=10 ** 6,
        early_stopping_rounds=100,
        valid_sets=dvalid,
        verbose_eval=1000)
    print(f"PARAMETER: {params}")
    gbm_reg = lgb.train(params,lgb.Dataset(X_train,label=y_train))
    predicted = gbm_reg.predict(X_test)

    # 結果出力
    pred_df = pd.DataFrame(y_test)
    pred_df['pred'] = predicted
    pred_df.columns = ['true', 'pred']

    return pred_df, gbm_reg, params, opt.best_trial

def train_no_optim(x_train, y_train):
    # XGBoostのデフォルトパラメータ
    default_xgb_params = {"objective": "reg:squarederror",
                          "eval_metric": "rmse",
                          "tree_method": "hist",
                          "grow_policy": "lossguide",
                          'verbose': 0,
                          "seed": 1,
                          "colsample_bytree": 1,
                          "subsample": 1,
                          "max_depth": 6,
                          'max_leaves': 31,
                          "eta": 0.3,
                          "alpha": 1.0,
                          "lambda": 0, }
    dtrain = xgb.DMatrix(data=x_train, label=y_train)
    xgb_reg = xgb.train(default_xgb_params,
                        dtrain=dtrain,
                        num_boost_round=10 ** 6,
                        early_stopping_rounds=1000,
                        verbose_eval=100)
    return xgb_reg, default_xgb_params
