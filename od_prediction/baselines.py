

import random
from keras.models import Input, Model, Sequential
from keras.layers import Dense, Activation, Dropout, Lambda
import keras.backend as K
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import sys
from utilities import *
from config import *
from keras_dgl.layers import MultiGraphCNN
from keras.optimizers import Adam, RMSprop
from copy import deepcopy
import time
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc, precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVC
from xgboost.sklearn import XGBClassifier, XGBRegressor


def train_each_model(reg, selected_flatten_train_x, selected_flatten_train_y, flatten_test_x):
    start_time = time.time()

    # train model and predict
    reg.fit(selected_flatten_train_x, selected_flatten_train_y)
    flatten_pred_y = reg.predict(flatten_test_x)

    timelast = time.time() - start_time
    return flatten_pred_y, timelast


def evaluate_each_model(flatten_pred_y, timelast, test_y, nb_node, save_file_name):
    # evaluation
    nb_samples = int(flatten_pred_y.shape[0] / nb_node)
    pred_y = flatten_pred_y.reshape(nb_samples, nb_node, test_y.shape[2])
    test_mse = metrics.mean_squared_error(test_y[:, :, 0], pred_y[:, :, 0])
    test_mae = metrics.mean_absolute_error(test_y[:, :, 0], pred_y[:, :, 0])
    print(test_mse)

    # final record
    final_records = {}
    final_records["test_y"] = test_y
    final_records["pred_y"] = pred_y

    # save results
    results = {"final_records": final_records,
               "train_time": timelast
               }
    pickle.dump(results, open(result_folder + save_file_name + ".pickle", 'wb'))


def run_model():
    # load data and prepare samples
    K.clear_session()
    train_x, test_x, train_y, test_y, adj_matrices = prepare_train_and_test_samples(
        TRAIN_START_DATE, TRAIN_END_DATE, TEST_START_DATE, TEST_END_DATE, num_selected_od_pairs=NUM_OD_PAIRS)
    print("Mean of train y and test y")
    print(np.mean(train_y), np.mean(test_y))

    # flatten the samples
    nb_node = train_x.shape[1]
    flatten_train_x = train_x.reshape(-1, train_x.shape[2]).astype("float")
    flatten_test_x = test_x.reshape(-1, test_x.shape[2]).astype("float")
    flatten_train_y = train_y.reshape(-1, train_y.shape[2]).astype("float")
    flatten_test_y = test_y.reshape(-1, test_y.shape[2]).astype("float")
    print("shape of flattened samples")
    print(flatten_train_x.shape, flatten_test_x.shape, flatten_train_y.shape, flatten_test_y.shape)
    print(type(flatten_train_x), flatten_train_x.dtype)

    # select samples if there are two many samples
    max_train_num = 1000000
    train_num = min(flatten_train_x.shape[0], max_train_num)
    train_sample_index = random.sample(list(range(0, flatten_train_x.shape[0])), train_num)
    selected_flatten_train_x = flatten_train_x[train_sample_index, :]
    selected_flatten_train_y = flatten_train_y[train_sample_index, :]

    # run models
    methodlist = ['HA', 'XGB', 'MLP', 'GBDT', 'RF', 'LR', 'LASSO']
    # methodlist = ['XGB']

    save_file_name_prefix = CITY + "_" + str(SAMPLE_CASE) + "\\"
    for eachmethod in methodlist:
        print("Model: %s" % eachmethod)
        if eachmethod == "HA":
            flatten_pred_y = flatten_test_x[:, 2]
            timelast = 0
            save_file_name = save_file_name_prefix + "HA---"
            evaluate_each_model(flatten_pred_y, timelast, test_y, nb_node, save_file_name)
        else:
            if eachmethod == 'LR':
                reg = LinearRegression()
                save_file_name = save_file_name_prefix + "LR---"
                flatten_pred_y, timelast = train_each_model(reg, selected_flatten_train_x,
                                                            selected_flatten_train_y, flatten_test_x)
                evaluate_each_model(flatten_pred_y, timelast, test_y, nb_node, save_file_name)
            elif eachmethod == 'LASSO':
                alpha_list = [0.1, 1, 10]
                for alpha in alpha_list:
                    reg = Lasso(alpha=alpha)
                    save_file_name = save_file_name_prefix + "LASSO---" + str(alpha)
                    flatten_pred_y, timelast = train_each_model(reg, selected_flatten_train_x,
                                                                selected_flatten_train_y, flatten_test_x)
                    evaluate_each_model(flatten_pred_y, timelast, test_y, nb_node, save_file_name)
            elif eachmethod == 'XGB':
                max_depth_list = [3, 5, 7]
                for max_depth in max_depth_list:
                    reg = XGBRegressor(max_depth=max_depth, learning_rate=0.01, subsample=0.8,
                                       nthread=10, n_estimators=1500)
                    save_file_name = save_file_name_prefix + "XGB---" + str(max_depth)
                    flatten_pred_y, timelast = train_each_model(reg, selected_flatten_train_x,
                                                                selected_flatten_train_y, flatten_test_x)
                    evaluate_each_model(flatten_pred_y, timelast, test_y, nb_node, save_file_name)
            elif eachmethod == 'MLP':
                param_lr_list = [0.001, 0.01, 0.1]
                for param_lr in param_lr_list:
                    reg = MLPRegressor(hidden_layer_sizes=(128, 64), early_stopping=True, solver="adam",
                                       learning_rate_init=param_lr)
                    save_file_name = save_file_name_prefix + "MLP---" + str(int(param_lr * 1000))
                    flatten_pred_y, timelast = train_each_model(reg, selected_flatten_train_x,
                                                                selected_flatten_train_y, flatten_test_x)
                    evaluate_each_model(flatten_pred_y, timelast, test_y, nb_node, save_file_name)
            elif eachmethod == 'GBDT':
                max_depth_list = [3, 5, 7]
                for max_depth in max_depth_list:
                    reg = GradientBoostingRegressor(max_depth=max_depth)
                    save_file_name = save_file_name_prefix + "GBDT---" + str(max_depth)
                    flatten_pred_y, timelast = train_each_model(reg, selected_flatten_train_x,
                                                                selected_flatten_train_y, flatten_test_x)
                    evaluate_each_model(flatten_pred_y, timelast, test_y, nb_node, save_file_name)
            elif eachmethod == 'RF':
                n_estimators_list = [10, 100, 200]
                for n_estimators in n_estimators_list:
                    reg = RandomForestRegressor(n_estimators=n_estimators)
                    save_file_name = save_file_name_prefix + "RF---" + str(n_estimators)
                    flatten_pred_y, timelast = train_each_model(reg, selected_flatten_train_x,
                                                                selected_flatten_train_y, flatten_test_x)
                    evaluate_each_model(flatten_pred_y, timelast, test_y, nb_node, save_file_name)


if __name__ == "__main__":
    run_model()
