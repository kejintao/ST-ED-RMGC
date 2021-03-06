
from keras.models import Input, Model, Sequential
from keras.layers import Dense, Activation, Dropout, Lambda, Flatten, Reshape, Permute, LSTM, Concatenate
import keras.backend as K
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from keras.regularizers import l1, l2, l1_l2
import sys
from utilities import *
from config import *
from keras_dgl.layers import MultiGraphCNN
from keras.optimizers import Adam, RMSprop
from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler
import gc
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

    # adjacent matrices combinations
    A1 = adj_matrices["identity"]
    A2 = adj_matrices["OD_based_corr"]
    A3 = adj_matrices["OD_based_ori_eucli_rev"]
    A4 = adj_matrices["OD_based_dest_eucli_rev"]
    A5 = adj_matrices["OD_based_ori_neighbor"]
    A6 = adj_matrices["OD_based_dest_neighbor"]
    A7 = adj_matrices["OD_based_ori_dist_rev"]
    A8 = adj_matrices["OD_based_dest_dist_rev"]
    adj_matrix_list = [A1, A2, A3, A4, A5, A6, A7, A8]
    A = np.concatenate(adj_matrix_list, axis=1)
    print(A.shape)
    num_filters = len(adj_matrix_list)

    # build model
    # encoded graph model
    X_input = Input(shape=(train_x.shape[1], train_x.shape[2]))
    graph_conv_filters_input = Input(shape=(A.shape[1], A.shape[2]))

    graph_encoded = MultiGraphCNN(NETWORK_STRUCTURE_E[0], num_filters, activation=ACTIVATION,
                                  activity_regularizer=REGULARIZER)([X_input, graph_conv_filters_input])
    # graph_encoded = Dropout(0.2)(graph_encoded)
    for l in range(1, len(NETWORK_STRUCTURE_E)):
        print("layer", l)
        nb_nodes = NETWORK_STRUCTURE_E[l]
        graph_encoded = MultiGraphCNN(nb_nodes, num_filters, activation=ACTIVATION,
                                      activity_regularizer=REGULARIZER)([graph_encoded, graph_conv_filters_input])
        # graph_encoded = Dropout(0.2)(graph_encoded)

    # Shape info needed to build Decoder Model
    shape = K.int_shape(graph_encoded)
    print("shape of decoder")
    print(shape)

    # Generate the latent vector
    graph_encoded = Flatten()(graph_encoded)
    graph_encoded = Dense(LATENT_DIM, name='latent_vector')(graph_encoded)
    print(K.int_shape(graph_encoded))

    # encoded LSTM model
    lstm_encoded = Permute((2, 1))(X_input)
    lstm_encoded = LSTM(NETWORK_STRUCTURE_L[0], return_sequences=True, activation=ACTIVATION,
                        activity_regularizer=REGULARIZER)(lstm_encoded)
    for l in range(1, len(NETWORK_STRUCTURE_L)):
        nb_nodes = NETWORK_STRUCTURE_L[l]
        lstm_encoded = LSTM(nb_nodes, return_sequences=True, activation=ACTIVATION,
                            activity_regularizer=REGULARIZER)(lstm_encoded)

    lstm_encoded = LSTM(LATENT_DIM, return_sequences=False, activation=ACTIVATION,
                        activity_regularizer=REGULARIZER)(lstm_encoded)
    lstm_encoded = Dense(LATENT_DIM_L, activation=ACTIVATION, activity_regularizer=REGULARIZER)(lstm_encoded)

    # merge two models
    merge_encoded = Concatenate(axis=-1)([graph_encoded, lstm_encoded])

    # decoder model
    x = Dense(shape[1] * shape[2])(merge_encoded)
    x = Reshape((shape[1], shape[2]))(x)
    print(K.int_shape(x))

    for l in range(0, len(NETWORK_STRUCTURE_D)):
        nb_nodes = NETWORK_STRUCTURE_D[l]
        x = MultiGraphCNN(nb_nodes, num_filters, activation=ACTIVATION,
                          activity_regularizer=REGULARIZER)([x, graph_conv_filters_input])
        # x = Dropout(0.2)(x)

    output = MultiGraphCNN(train_y.shape[2], num_filters, activation='linear',
                           activity_regularizer=REGULARIZER)([x, graph_conv_filters_input])
    model = Model(inputs=[X_input, graph_conv_filters_input], outputs=output)
    sgd = Adam(lr=LR_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['mse'])
    print(model.summary())

    # train the model
    epoch_records = {}
    start_time = time.time()
    for n in range(0, NB_EPOCHS):
        print("Epoch: ", n)
        for batch in iterate_minibatches(train_x, train_y, BATCH_SIZE, shuffle=True):
            x_batch, y_batch = batch
            A_titled = np.tile(A, (BATCH_SIZE, 1, 1))
            graph_conv_filters = preprocess_adj_tensor(A_titled, SYM_NORM)
            model.train_on_batch([x_batch, graph_conv_filters], y_batch)

        # evaluations per epoch
        test_A_titled = np.tile(A, (test_x.shape[0], 1, 1))
        test_graph_conv_filters = preprocess_adj_tensor(test_A_titled, SYM_NORM)
        pred_y = model.predict([test_x, test_graph_conv_filters])
        test_mse = metrics.mean_squared_error(test_y[:, :, 0], pred_y[:, :, 0])
        test_mae = metrics.mean_absolute_error(test_y[:, :, 0], pred_y[:, :, 0])

        train_A_titled = np.tile(A, (train_x.shape[0], 1, 1))
        train_graph_conv_filters = preprocess_adj_tensor(train_A_titled, SYM_NORM)
        train_pred_y = model.predict([train_x, train_graph_conv_filters])
        train_mse = metrics.mean_squared_error(train_y[:, :, 0], train_pred_y[:, :, 0])
        train_mae = metrics.mean_absolute_error(train_y[:, :, 0], train_pred_y[:, :, 0])

        print(train_mse, test_mse)

        # record
        epoch_records[n] = [pred_y, train_pred_y]

    # final record
    timelast = time.time() - start_time
    final_records = {}
    final_records["train_y"] = train_y
    final_records["test_y"] = test_y
    final_records["pred_y"] = pred_y

    # save results
    results = {"final_records": final_records,
               "epoch_records": epoch_records,
               "train_time": timelast,
               "lr_rate": LR_RATE,
               "network_structure_e": NETWORK_STRUCTURE_E,
               "network_structure_l": NETWORK_STRUCTURE_L,
               "network_structure_d": NETWORK_STRUCTURE_D,
               "latent_dim": LATENT_DIM,
               "latent_dim_l": LATENT_DIM_L,
               "regularizer_rate": REGULARIZER_RATE,
               "activation": ACTIVATION
               }
    save_file_name = CITY + "_" + str(SAMPLE_CASE) + "\\multi_graph_cnn_lstm_ed---" + str(int(LR_RATE * 10000)) + "-"
    for l in range(0, len(NETWORK_STRUCTURE_L)):
        save_file_name = save_file_name + str(NETWORK_STRUCTURE_L[l]) + "-"
    save_file_name = save_file_name + "-_"
    for l in range(0, len(NETWORK_STRUCTURE_E)):
        save_file_name = save_file_name + str(NETWORK_STRUCTURE_E[l]) + "-"
    save_file_name = save_file_name + "_" + str(LATENT_DIM) + "_" + str(LATENT_DIM_L) + "_-"
    for l in range(0, len(NETWORK_STRUCTURE_D)):
        save_file_name = save_file_name + str(NETWORK_STRUCTURE_D[l]) + "-"
    save_file_name = save_file_name + ACTIVATION + "_" + str(int(REGULARIZER_RATE * 10000))
    pickle.dump(results, open(result_folder + save_file_name + ".pickle", 'wb'))

    # save models
    model.save(result_folder + save_file_name + '.h5')


if __name__ == "__main__":
    # hyper parameters (重点调整 LR_RATE 和 NETWORK_STRUCTURE)
    SYM_NORM = True
    NB_EPOCHS = 300
    BATCH_SIZE = 16
    NETWORK_STRUCTURE_E = [256]
    NETWORK_STRUCTURE_D = [256]
    NETWORK_STRUCTURE_L = [256, 128, 64]
    LR_RATE = 0.0001
    REGULARIZER_RATE = 0
    REGULARIZER = l1(REGULARIZER_RATE)
    ACTIVATION = 'relu'
    LATENT_DIM = 12
    LATENT_DIM_L = 4

    # run model
    run_model()
