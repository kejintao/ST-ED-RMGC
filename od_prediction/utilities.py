import numpy as np
import pandas as pd
from copy import deepcopy
import pickle
from datetime import datetime, timedelta
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh, ArpackNoConvergence
from sklearn import metrics
from paths import *
from sklearn.preprocessing import MinMaxScaler
import gc


# features 正规化
def scale_features(s_train_x, s_test_x, dn_range, up_range):
    # remove abnormal data (convert to 0)
    s_train_x = s_train_x.astype("float")
    s_test_x = s_test_x.astype("float")
    s_train_x[np.isnan(s_train_x)] = 0
    s_test_x[np.isnan(s_test_x)] = 0
    s_train_x[np.isinf(s_train_x)] = 0
    s_test_x[np.isinf(s_test_x)] = 0

    n_train_x = s_train_x.reshape(s_train_x.shape[0] * s_train_x.shape[1], s_train_x.shape[-1])
    n_test_x = s_test_x.reshape(s_test_x.shape[0] * s_test_x.shape[1], s_test_x.shape[-1])
    # scale
    scaler = MinMaxScaler(feature_range=(dn_range, up_range))
    i_train_x = scaler.fit_transform(n_train_x)
    i_test_x = scaler.transform(n_test_x)

    b_train_x = i_train_x.reshape(s_train_x.shape[0], s_train_x.shape[1], s_train_x.shape[2])
    b_test_x = i_test_x.reshape(s_test_x.shape[0], s_test_x.shape[1], s_test_x.shape[2])

    del n_train_x, n_test_x, s_train_x, s_test_x, i_train_x, i_test_x
    gc.collect()
    return b_train_x, b_test_x


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def reg_eval(y_true, y_pred):
    """
    """
    mse = metrics.mean_squared_error(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mmape = sum(abs(2 * (y_true - y_pred) / (y_true + y_pred))) / len(y_true)
    return mse, mae, mmape


def reg_eval_y_true(y_true, y_pred):
    """
    """
    mse = metrics.mean_squared_error(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mmape = sum(abs(y_true - y_pred) / (y_true + 10e-5)) / len(y_true)
    return mse, mae, mmape


def reg_eval_y_pred(y_true, y_pred):
    """
    """
    mse = metrics.mean_squared_error(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mmape = sum(abs((y_true - y_pred) / y_pred)) / len(y_true)
    return mse, mae, mmape


def reg_eval_with_threshold(y_true, y_pred):
    df_local = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    thresholds = [0, 1, 2, 5, 10, 20, 50]
    mmape_list = []
    for threshold in thresholds:
        _, _, mmape = reg_eval(
            df_local[df_local["y_true"] >= threshold]["y_true"],
            df_local[df_local["y_true"] >= threshold]["y_pred"]
        )
        mmape_list.append(mmape)
    return mmape_list


def normalize_adj(adj, symmetric=True):
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return a_norm


def normalize_adj_numpy(adj, symmetric=True):
    if symmetric:
        d = np.diag(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d)
    else:
        d = np.diag(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj)
    return a_norm


def preprocess_adj(adj, symmetric=True):
    adj = adj + sp.eye(adj.shape[1])
    adj = normalize_adj(adj, symmetric)
    return adj


def preprocess_adj_numpy(adj, symmetric=True):
    adj = adj + np.eye(adj.shape[0])
    adj = normalize_adj_numpy(adj, symmetric)
    return adj


def preprocess_adj_tensor(adj_tensor, symmetric=True):
    adj_out_tensor = []
    # print(adj_tensor.shape)
    for i in range(adj_tensor.shape[0]):
        adj = adj_tensor[i]
        adj_count = int(adj.shape[0] / adj.shape[1])
        adj_list = []
        for m in range(0, adj_count):
            sub_adj = adj[int(m * adj.shape[1]): int((m+1) * adj.shape[1]), :]
            sub_adj = sub_adj + np.eye(sub_adj.shape[0])
            sub_adj = normalize_adj_numpy(sub_adj, symmetric)
            adj_list.append(sub_adj)
        adj = np.concatenate(adj_list, axis=0)
        adj_out_tensor.append(adj)
    adj_out_tensor = np.array(adj_out_tensor)
    return adj_out_tensor


def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def normalized_laplacian(adj, symmetric=True):
    adj_normalized = normalize_adj(adj, symmetric)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    return laplacian


def rescale_laplacian(laplacian):
    try:
        print('Calculating largest eigenvalue of normalized graph Laplacian...')
        largest_eigval = eigsh(laplacian, 1, which='LM', return_eigenvectors=False)[0]
    except ArpackNoConvergence:
        print('Eigenvalue calculation did not converge! Using largest_eigval=2 instead.')
        largest_eigval = 2

    scaled_laplacian = (2. / largest_eigval) * laplacian - sp.eye(laplacian.shape[0])
    return scaled_laplacian


def chebyshev_polynomial(X, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    T_k = list()
    T_k.append(sp.eye(X.shape[0]).tocsr())
    T_k.append(X)

    def chebyshev_recurrence(T_k_minus_one, T_k_minus_two, X):
        X_ = sp.csr_matrix(X, copy=True)
        return 2 * X_.dot(T_k_minus_one) - T_k_minus_two

    for i in range(2, k+1):
        T_k.append(chebyshev_recurrence(T_k[-1], T_k[-2], X))

    return T_k


def get_features_by_looking_back(graph_od_demand, train_date_list, test_date_list, lag=1):
    graph_od_demand_t1 = deepcopy(graph_od_demand)
    graph_od_demand_t1[["date", "hour"]] = graph_od_demand_t1[["date", "hour"]].shift(-lag)
    graph_od_demand_t1 = graph_od_demand_t1[graph_od_demand_t1["date"].notnull()]

    train_graph_t1 = graph_od_demand_t1.loc[graph_od_demand_t1["date"].isin(pd.to_datetime(train_date_list)), :]
    test_graph_t1 = graph_od_demand_t1.loc[graph_od_demand_t1["date"].isin(pd.to_datetime(test_date_list)), :]

    train_graph_t1_arr = train_graph_t1.values[..., np.newaxis]
    test_graph_t1_arr = test_graph_t1.values[..., np.newaxis]

    return train_graph_t1, test_graph_t1, train_graph_t1_arr, test_graph_t1_arr


def load_feature_data():
    """
    graph od demand & od list
    :return:
    """
    data_file_name = "fhv_features_2018"
    feature_data = pd.read_pickle(data_folder + data_file_name + ".pickle")
    graph_od_demand = feature_data["graph_od_demand"]
    graph_od_demand["date"] = pd.to_datetime(graph_od_demand["date"])
    od_list_df = feature_data["od_list_df"]

    adj_matrix_file_name = "NewYork_adj_matrices"
    adj_matrices = pd.read_pickle(data_folder + adj_matrix_file_name + ".pickle")
    return graph_od_demand, od_list_df, adj_matrices


def select_od_pairs(graph_od_demand, od_list_df, num_selected_od_pairs):
    od_names = od_list_df["name"].values
    od_mean_values = graph_od_demand[od_names].mean()
    od_mean_values_sorted = od_mean_values.sort_values(ascending=False)
    selected_od_pairs = list(od_mean_values_sorted[0: num_selected_od_pairs].index)
    print("minimum of mean demand of selected od pairs:", od_mean_values_sorted[num_selected_od_pairs])
    return selected_od_pairs


def prepare_train_and_test_samples(train_start, train_end, test_start, test_end, num_selected_od_pairs=1000):
    """
    obtain
    1) train dataset and test dataset
    2) adjacent matrices (with selected od pairs)
    :param graph_od_demand:
    :param od_list_df:
    :param num_selected_od_pairs:
    :return:
    """
    # date list
    start = datetime.strptime(train_start, "%d-%m-%Y")
    end = datetime.strptime(train_end, "%d-%m-%Y")
    train_date_list = [start + timedelta(days=x) for x in range(0, (end - start).days)]

    start = datetime.strptime(test_start, "%d-%m-%Y")
    end = datetime.strptime(test_end, "%d-%m-%Y")
    test_date_list = [start + timedelta(days=x) for x in range(0, (end - start).days)]

    # load data and select od pairs
    graph_od_demand, od_list_df, adj_matrices = load_feature_data()
    selected_od_pairs = select_od_pairs(graph_od_demand, od_list_df, num_selected_od_pairs)
    selected_graph_od_demand = graph_od_demand[["date", "hour"] + selected_od_pairs]
    print("Total od demand matrix shape: ", selected_graph_od_demand.shape)

    # load adjacent matrices
    new_adj_matrices = {}
    matrices_to_used = ["OD_based_corr", "OD_based_ori_eucli_rev", "OD_based_dest_eucli_rev",
                        "OD_based_ori_neighbor", "OD_based_dest_neighbor",
                        "OD_based_ori_dist_rev", "OD_based_dest_dist_rev"]
    print("adj matrices")
    for key in matrices_to_used:
        A = adj_matrices[key]
        # normalize data and fill the diagonal
        A = A / A.max().max()
        np.fill_diagonal(A.values, 0)
        # select features
        A_ = A.loc[selected_od_pairs, selected_od_pairs]
        A_array = A_.values[np.newaxis, ...]
        new_adj_matrices[key] = A_array
        print(A_array.max(), A_array.min())

    A_array = np.eye(A_array.shape[1])
    A_array = A_array[np.newaxis, ...]
    new_adj_matrices["identity"] = A_array

    # obtain features by slicing back windows
    train_graph_label, test_graph_label, train_graph_label_arr, test_graph_label_arr = get_features_by_looking_back(
        selected_graph_od_demand, train_date_list, test_date_list, lag=0)
    train_graph_t1, test_graph_t1, train_graph_t1_arr, test_graph_t1_arr = get_features_by_looking_back(
        selected_graph_od_demand, train_date_list, test_date_list, lag=1)
    train_graph_t2, test_graph_t2, train_graph_t2_arr, test_graph_t2_arr = get_features_by_looking_back(
        selected_graph_od_demand, train_date_list, test_date_list, lag=2)
    train_graph_d1, test_graph_d1, train_graph_d1_arr, test_graph_d1_arr = get_features_by_looking_back(
        selected_graph_od_demand, train_date_list, test_date_list, lag=24)
    train_graph_w1, test_graph_w1, train_graph_w1_arr, test_graph_w1_arr = get_features_by_looking_back(
        selected_graph_od_demand, train_date_list, test_date_list, lag=24 * 7)

    print("features graphs' dimensions:")
    print(train_graph_label.shape, train_graph_t1.shape, train_graph_t2.shape, train_graph_d1.shape,
          train_graph_w1.shape)
    print(test_graph_label.shape, test_graph_t1.shape, test_graph_t2.shape, test_graph_d1.shape, test_graph_w1.shape)

    # train and test samples
    train_x = np.concatenate((train_graph_t1_arr, train_graph_t2_arr, train_graph_d1_arr, train_graph_w1_arr),
                             axis=2)[:, 2:, :]
    test_x = np.concatenate((test_graph_t1_arr, test_graph_t2_arr, test_graph_d1_arr, test_graph_w1_arr),
                            axis=2)[:, 2:, :]
    train_y = train_graph_label_arr[:, 2:, :]
    test_y = test_graph_label_arr[:, 2:, :]
    print("samples' dimensions:")
    print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)

    return train_x, test_x, train_y, test_y, new_adj_matrices


if __name__ == "__main__":
    train_start = "08-01-2018"
    train_end = "09-01-2018"
    test_start = "23-04-2018"
    test_end = "24-04-2018"
    prepare_train_and_test_samples(train_start, train_end, test_start, test_end, num_selected_od_pairs=1000)

