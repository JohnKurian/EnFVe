import numpy as np
import pandas as pd
import os
import lightgbm as lgb
import optuna
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import joblib
import catboost as cb


train_data = ''
test_data = ''

train = []
test = []

model = 'lightgbm'

if model == 'lightgbm':
    optuna_model = 'lightgbm_optuna_model'
else:
    optuna_model = 'optuna_model'


def convert_to_onehot(values):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)

    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded


if os.path.exists('train.csv'):
    train = pd.read_csv('train.csv', header=None)
    test = pd.read_csv('test.csv', header=None)


ntrain = train.to_numpy()
ntest = test.to_numpy()

X_train, y_train = ntrain[:, 1:], ntrain[:, 0]
X_test, y_test = ntest[:, 1:], ntest[:, 0]

# y_test = convert_to_onehot(y_test)
# y_train = convert_to_onehot(y_train)

lgb_train = lgb.Dataset(X_train, free_raw_data=True, label=y_train)
lgb_test = lgb.Dataset(X_test, free_raw_data=True, label=y_test)


def lightGBM_objective(trial):
    params = {
        'max_bin': trial.suggest_int('max_bin', 128, 1024),
        'raw_data': None,
        'metric': ['multi_logloss'],
        'num_leaves': trial.suggest_int('num_leaves', 5, 20),
        'boosting_type': 'dart',
        'objective': 'multiclass',
        'num_iterations': 200,
        'learning_rate': trial.suggest_uniform('learning_rate', 0.01, 0.1),
        'tree_learner': trial.suggest_categorical("tree_learner", ["serial", "feature", "data", "voting"]),
        'task': 'train',
        'is_training_metric': False,
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
        'min_sum_hessian_in_leaf': trial.suggest_int('min_sum_hessian_in_leaf', 10, 100),
        'ndcg_eval_at': [1, 3, 5, 10],
        'sparse_threshold': trial.suggest_uniform('sparse_threshold', 0.01, 1),
        'device': 'cpu',
        'num_classes': 3

    }

    gbm = lgb.train(params,
                    lgb_train,
                    keep_training_booster=True,
                    verbose_eval=False,
                    #                 init_model='temp_model.txt',
                    valid_sets=[lgb_test]
                    )

    preds = gbm.predict(X_test)
    preds = np.argmax(preds, axis=1)
    # preds = [x[0] for x in preds]

    accuracy = np.sum(np.equal(preds, y_test)) / len(y_test)
    print('accuracy:', accuracy)


    return accuracy


def catboost_objective(trial):
    train_x, valid_x, train_y, valid_y = X_train, X_test, y_train, y_test

    param = {
        # "objective": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
        # "colsample_bylevel": trial.suggest_uniform("colsample_bylevel", 0.01, 0.1),
        # "depth": trial.suggest_int("depth", 1, 12),
        # "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
        # "bootstrap_type": trial.suggest_categorical(
        #     "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"],
        # ),
        "depth": trial.suggest_int("depth", 5, 16),
        "loss_function": "MultiClass",
        "learning_rate": trial.suggest_uniform("learning_rate", 0.001, 0.1),
        "l2_leaf_reg": trial.suggest_uniform('l2_leaf_reg', 1.5, 4.5),
        "task_type": 'GPU'

    }

    # if param["bootstrap_type"] == "Bayesian":
    #     param["bagging_temperature"] = trial.suggest_uniform("bagging_temperature", 0, 10)
    # elif param["bootstrap_type"] == "Bernoulli":
    #     param["subsample"] = trial.suggest_uniform("subsample", 0.1, 1)

    gbm = cb.CatBoostClassifier(**param)

    gbm.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], verbose=0, early_stopping_rounds=100)

    preds = gbm.predict(valid_x)
    # pred_labels = np.rint(preds)
    # accuracy = accuracy_score(valid_y, pred_labels)

    preds = [x[0] for x in preds]
    accuracy = np.sum(np.equal(preds, valid_y)) / len(valid_y)

    return accuracy


study = optuna.create_study(direction="maximize")

if os.path.exists('lightgbm_optuna_model.pkl'):
    study = joblib.load('lightgbm_optuna_model.pkl')


study.optimize(lightGBM_objective, n_trials=20, timeout=600)

print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))


joblib.dump(study, 'optuna_study.pkl')
