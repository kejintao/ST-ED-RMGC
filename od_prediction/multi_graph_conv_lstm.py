
from keras.models import Input, Model, Sequential
from keras.layers import Dense, Activation, Dropout, Lambda
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

    # adjacent matrices combinations
    A1 = adj_matrices["identity"]
    A2 = adj_matrices["OD_based_corr"]
    A3 = adj_matrices["OD_based_ori_eucli_rev"]
    A4 = adj_matrices["OD_based_dest_eucli_rev"]
    A5 = adj_matrices["OD_based_ori_neighbor"]
    A6 = adj_matrices["OD_based_dest_neighbor"]
    A7 = adj_matrices["OD_based_ori_dist_rev"]
    A8 = adj_matrices["OD_based_dest_dist_rev"]
    adj_matrix_list = [A1]
    graph_conv_tensor = np.concatenate(adj_matrix_list, axis=0)
    print(graph_conv_tensor.shape)
    num_filters = len(adj_matrix_list)

    # reform input output data
    reshaped_train_x = np.transpose(train_x, (0, 2, 1))
    reshaped_train_x = reshaped_train_x[..., np.newaxis]
    reshaped_test_x = np.transpose(test_x, (0, 2, 1))
    reshaped_test_x = reshaped_test_x[..., np.newaxis]
    print(reshaped_train_x.shape, train_y.shape)

    # build model
    X_input = Input(shape=(reshaped_train_x.shape[1], reshaped_train_x.shape[2], reshaped_train_x.shape[3]))
    output = GraphConvLSTM(NETWORK_STRUCTURE[0], graph_conv_tensor, activation=ACTIVATION,
                           activity_regularizer=REGULARIZER, return_sequences=True)(X_input)
    # final output
    output = GraphConvLSTM(train_y.shape[2], graph_conv_tensor, activation=ACTIVATION,
                           activity_regularizer=REGULARIZER, return_sequences=False)(output)

    model = Model(inputs=X_input, outputs=output)
    sgd = Adam(lr=LR_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['mse'])
    print(model.summary())

    # train the model
    epoch_records = {}
    for n in range(0, NB_EPOCHS):
        print("Epoch: ", n)
        for batch in iterate_minibatches(reshaped_train_x, train_y, BATCH_SIZE, shuffle=True):
            x_batch, y_batch = batch
            model.train_on_batch(x_batch, y_batch)

        # evaluations per epoch
        pred_y = model.predict(reshaped_test_x)
        test_mse = metrics.mean_squared_error(test_y[:, :, 0], pred_y[:, :, 0])
        test_mae = metrics.mean_absolute_error(test_y[:, :, 0], pred_y[:, :, 0])

        train_pred_y = model.predict(reshaped_train_x)
        train_mse = metrics.mean_squared_error(train_y[:, :, 0], train_pred_y[:, :, 0])
        train_mae = metrics.mean_absolute_error(train_y[:, :, 0], train_pred_y[:, :, 0])

        print(train_mse, test_mse)

        # record
        epoch_records[n] = [pred_y, train_pred_y]

    # final record
    final_records = {}
    final_records["train_y"] = train_y
    final_records["test_y"] = test_y
    final_records["pred_y"] = pred_y

    # save results
    results = {"final_records": final_records,
               "epoch_records": epoch_records,
               "lr_rate": LR_RATE,
               "network_structure": NETWORK_STRUCTURE
               }
    save_file_name = CITY + "_" + str(SAMPLE_CASE) + "\\multi_graph_conv_lstm_" + str(int(LR_RATE * 10000)) + "_"
    for l in range(0, len(NETWORK_STRUCTURE)):
        save_file_name = save_file_name + "-" + str(NETWORK_STRUCTURE[l])
    pickle.dump(results, open(result_folder + save_file_name + ".pickle", 'wb'))

    # save models
    model.save(result_folder + save_file_name + '.h5')


if __name__ == "__main__":
    # hyper parameters (重点调整 LR_RATE 和 NETWORK_STRUCTURE)
    SYM_NORM = True
    NB_EPOCHS = 500
    BATCH_SIZE = 64
    NETWORK_STRUCTURE = [256, 128, 64]
    LR_RATE = 0.0001
    REGULARIZER = l1(0)
    ACTIVATION = "relu"

    # run model
    run_model()
