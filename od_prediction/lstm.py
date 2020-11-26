
from keras.models import Input, Model, Sequential
from keras.layers import Dense, Activation, Dropout, Lambda, LSTM
import keras.backend as K
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from keras.regularizers import l1, l2, l1_l2
import sys
from utilities import *
from config import *
from keras_dgl.layers import MultiGraphCNN, GraphConvLSTM
from keras.optimizers import Adam, RMSprop
from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler
import gc
import random
import time


def run_model():
    # load data and prepare samples
    K.clear_session()
    train_x, test_x, train_y, test_y, adj_matrices = prepare_train_and_test_samples(
        TRAIN_START_DATE, TRAIN_END_DATE, TEST_START_DATE, TEST_END_DATE, num_selected_od_pairs=NUM_OD_PAIRS)
    print("Mean of train y and test y")
    print(np.mean(train_y), np.mean(test_y))

    # normalize data
    dn_range, up_range = 0, 1
    train_x, test_x = scale_features(train_x, test_x, dn_range, up_range)

    # flatten the samples
    nb_node = train_x.shape[1]
    flatten_train_x = train_x.reshape(-1, train_x.shape[2]).astype("float")[..., np.newaxis]
    flatten_test_x = test_x.reshape(-1, test_x.shape[2]).astype("float")[..., np.newaxis]
    flatten_train_y = train_y.reshape(-1, train_y.shape[2]).astype("float")
    flatten_test_y = test_y.reshape(-1, test_y.shape[2]).astype("float")
    print("shape of flattened samples")
    print(flatten_train_x.shape, flatten_test_x.shape, flatten_train_y.shape, flatten_test_y.shape)

    # select samples if there are two many samples
    max_train_num = 1000000
    train_num = min(flatten_train_x.shape[0], max_train_num)
    train_sample_index = random.sample(list(range(0, flatten_train_x.shape[0])), train_num)
    flatten_train_x = flatten_train_x[train_sample_index, :]
    flatten_train_y = flatten_train_y[train_sample_index, :]

    # build model
    X_input = Input(shape=(flatten_train_x.shape[1], flatten_train_x.shape[2]))
    output = LSTM(NETWORK_STRUCTURE[0], return_sequences=True, activation=ACTIVATION,
                  activity_regularizer=REGULARIZER)(X_input)
    for l in range(1, len(NETWORK_STRUCTURE)):
        print("layer", l)
        nb_nodes = NETWORK_STRUCTURE[l]
        output = LSTM(nb_nodes, return_sequences=True, activation=ACTIVATION,
                      activity_regularizer=REGULARIZER)(output)
    output = LSTM(flatten_train_y.shape[1], return_sequences=False, activation=ACTIVATION,
                  activity_regularizer=REGULARIZER)(output)
    model = Model(inputs=X_input, outputs=output)
    sgd = Adam(lr=LR_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['mse'])
    print(model.summary())

    # train the model
    start_time = time.time()
    epoch_records = {}
    for n in range(0, NB_EPOCHS):
        print("Epoch: ", n)
        for batch in iterate_minibatches(flatten_train_x, flatten_train_y, BATCH_SIZE, shuffle=True):
            x_batch, y_batch = batch
            model.train_on_batch(x_batch, y_batch)

        # evaluations per epoch
        flatten_pred_y = model.predict(flatten_test_x)
        nb_samples = int(flatten_pred_y.shape[0] / nb_node)
        pred_y = flatten_pred_y.reshape(nb_samples, nb_node, test_y.shape[2])
        test_mse = metrics.mean_squared_error(test_y[:, :, 0], pred_y[:, :, 0])
        test_mae = metrics.mean_absolute_error(test_y[:, :, 0], pred_y[:, :, 0])

        flatten_train_pred_y = model.predict(flatten_train_x)
        train_mse = metrics.mean_squared_error(flatten_train_pred_y, flatten_train_y)
        train_mae = metrics.mean_absolute_error(flatten_train_pred_y, flatten_train_y)

        print(train_mse, test_mse)

        # record
        epoch_records[n] = [pred_y, flatten_train_pred_y]

    # final record
    timelast = time.time() - start_time
    final_records = {}
    final_records["flatten_train_y"] = flatten_train_y
    final_records["train_y"] = train_y
    final_records["test_y"] = test_y
    final_records["pred_y"] = pred_y

    # save results
    results = {"final_records": final_records,
               "epoch_records": epoch_records,
               "train_time": timelast,
               "lr_rate": LR_RATE,
               "network_structure": NETWORK_STRUCTURE,
               "regularizer_rate": REGULARIZER_RATE,
               "activation": ACTIVATION
               }
    save_file_name = CITY + "_" + str(SAMPLE_CASE) + "\\lstm---" + str(int(LR_RATE * 10000)) + "-"
    for l in range(0, len(NETWORK_STRUCTURE)):
        save_file_name = save_file_name + str(NETWORK_STRUCTURE[l]) + "-"
    save_file_name = save_file_name + ACTIVATION + "_" + str(int(REGULARIZER_RATE * 10000))
    pickle.dump(results, open(result_folder + save_file_name + ".pickle", 'wb'))

    # save models
    model.save(result_folder + save_file_name + '.h5')


if __name__ == "__main__":
    # hyper parameters (重点调整 LR_RATE 和 NETWORK_STRUCTURE)
    SYM_NORM = True
    NB_EPOCHS = 30
    BATCH_SIZE = 64
    NETWORK_STRUCTURE = [128, 64]
    LR_RATE = 0.0001
    REGULARIZER_RATE = 0
    REGULARIZER = l1(REGULARIZER_RATE)
    ACTIVATION = "relu"

    # run model
    run_model()
