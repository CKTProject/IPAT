import tensorflow as tf
from tensorflow import keras
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

import time
import os
import shutil
import sys

from doubanmodel import GCN, MLP
from tasks import Classifier, Cluster, LinkPredictor
import utils
from utils import Graphdata
from utils import ConfigerLoader

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle
from sklearn.preprocessing import label_binarize, LabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split



import randomwalk
from randomwalk import RandomWalk

import networkx as nx
import argparse
import json
import pickle as pkl




#fix random seed when debug
#seed = 123
#np.random.seed(seed)
#tf.compat.v1.set_random_seed(seed)

class RedirectStdStreams:
    def __init__(self, stdout=None, stderr=None):
        self._stdout = stdout or sys.stdout
        self._stderr = stderr or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush()
        self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush()
        self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr


def training_embedding(adj, X, Y, data_index, emb_flag, args, multi_flag):

    train_index, val_index, test_index = data_index

    sp_input_flag = False

    tf.keras.backend.set_floatx('float64')


    features = X
    adj = adj

    pre_flag = True
    if pre_flag:
        features = utils.tuple_to_sparse(utils.preprocess_features(features))

    star_time = time.time()
    if not sp_input_flag:
        svdk = 1000
        start_time = time.time()
        if os.path.exists("temp/%s_svd_%d_%s.pkl"%(args.dataset, svdk, pre_flag)):
            with open("temp/%s_svd_%d_%s.pkl"%(args.dataset, svdk, pre_flag), 'rb') as fin:
                u = pkl.load(fin)
        else:
            u, _, _ = svds(features, k=svdk)
            with open("temp/%s_svd_%d_%s.pkl"%(args.dataset, svdk, pre_flag), 'wb') as fout:
                pkl.dump(u, fout)
        features = u
    print("svd costs %.2fs"%(time.time()-star_time))

    num_features = features.shape[1]
    num_nodes = adj.shape[0]
    num_class = Y.shape[1]

    nb_sampled_adj = adj.copy()
    for i in range(args.neighbor_hop-1):
       nb_sampled_adj = nb_sampled_adj.dot(adj)


    with open(args.network_config, 'r') as f:
        network_config = json.load(f)


    if args.model_name == 'GCN':
        model = GCN(input_dim=num_features, hidden_dim=args.hidden_dim, output_dim=num_class, sparse_inputs=False, act=tf.nn.relu, dropout=args.dropout, bias=False, l2_r=args.l2_r,
                 learning_rate=args.learning_rate, multi_flag=multi_flag
            )
    if args.model_name == 'MLP':
        model = MLP(input_dim=num_features, hidden_dim=args.hidden_dim, output_dim=num_class, sparse_inputs=False, act=tf.nn.relu, dropout=args.dropout, bias=False, l2_r=args.l2_r,
                 learning_rate=args.learning_rate, multi_flag=multi_flag
            )
    else:
        assert False, "No such model"

    model.compile()


    # The number of epoch it has waited when loss is no longer minimum.
    wait = 0
    best = np.Inf
    # store best weights
    best_weights = None
    #initialize the link predict acc as zero(use to stop GAE based model)
    best_acc = 0.
    #flag of imporve loss or link prediction acc
    imporve = True
    lr_dec = 0

    patience = args.early_stop
    lr_patience = args.max_lr_dec

    last_gradint_dir = np.random.normal(size=(num_nodes, args.hidden_dim))



    #partial training model
    def batches(l, n):
        for i in range(0, len(l), n):
            yield l[i:i+n]


    def top_k_acc(y_pred, y_true):
        class_num = y_true.shape[1]
        top_k_list = np.sum(y_true, axis=1, dtype=np.int32).tolist()
        top_k_pred = []
        for index, k in enumerate(top_k_list):
            top_k_args = np.argsort(y_pred[index])[-k:]
            y = np.zeros(class_num)
            y[top_k_args] = 1
            top_k_pred.append(y)
        top_k_pred = np.array(top_k_pred, dtype=np.int32)

        micro_f1 = f1_score(top_k_pred, y_true, average='micro')
        macro_f1 = f1_score(top_k_pred, y_true, average='macro')

        return micro_f1, macro_f1


    #debug
    print("start training")
    for i in range(args.epochs):
        star_time = time.time()

        batch_train_loss = 0.
        batch_val_loss = 0.
        batch_iter_num = 0.

        batch_train_labels = []
        batch_train_preds = []
        batch_val_labels = []
        batch_val_preds = []

        train_index = shuffle(train_index)
        for batch_index in batches(train_index, args.batch_size):

            b_size = len(batch_index)

            x = features[batch_index, :]
            batch_labels = Y[batch_index, :]

            nb_index = utils.sample_k_neighbors_from_adj(nb_sampled_adj, args.nb_size, batch_index, False)
            nb_index = np.reshape(nb_index, (-1))
            nb_x = features[nb_index, :]
            nb_x = np.reshape(nb_x, (x.shape[0], args.nb_size, x.shape[1]))

            nbnb_index = utils.sample_k_neighbors_from_adj(nb_sampled_adj, args.nb_size, nb_index, False)
            nbnb_index = np.reshape(nbnb_index, (-1))
            nbnb_x = features[nbnb_index, :]
            nbnb_x = np.reshape(nbnb_x, (x.shape[0]*args.nb_size, args.nb_size, x.shape[1]))



            with tf.GradientTape() as tape:
                if args.model_name == "GCN":
                    loss, acc, y_pred = model((x, nb_x, nbnb_x, batch_labels), training=True)
                elif args.model_name == "MLP":
                    loss, acc, y_pred = model((x, batch_labels), training=True)
                

                trainable_vars = model.trainable_variables
                gradients = tape.gradient(loss, trainable_vars)
                model.optimizer.apply_gradients(zip(gradients, trainable_vars))

                batch_train_loss += loss * b_size

            batch_train_labels.append(batch_labels)
            batch_train_preds.append(y_pred)

            #print("Epoch %d, Batch %d, train loss %.5f, train acc %.2f"%(i, batch_iter_num, loss, acc))

            batch_iter_num += 1

        batch_train_loss = batch_train_loss / len(train_index)

#calculate val node classification accuracy
        for batch_index in batches(val_index, args.batch_size):

            b_size = len(batch_index)

            x = features[batch_index, :]
            batch_labels = Y[batch_index, :]

            nb_index = utils.sample_k_neighbors_from_adj(nb_sampled_adj, args.nb_size, batch_index, False)
            nb_index = np.reshape(nb_index, (-1))
            nb_x = features[nb_index, :]
            nb_x = np.reshape(nb_x, (x.shape[0], args.nb_size, x.shape[1]))

            nbnb_index = utils.sample_k_neighbors_from_adj(nb_sampled_adj, args.nb_size, nb_index, False)
            nbnb_index = np.reshape(nbnb_index, (-1))
            nbnb_x = features[nbnb_index, :]
            nbnb_x = np.reshape(nbnb_x, (x.shape[0]*args.nb_size, args.nb_size, x.shape[1]))

            with tf.GradientTape() as tape:
                if args.model_name == "GCN":
                    loss, acc, y_pred = model((x, nb_x, nbnb_x, batch_labels), training=False)
                elif args.model_name == "MLP":
                    loss, acc, y_pred = model((x, batch_labels), training=False)

                batch_val_loss += loss * b_size
                batch_val_labels.append(batch_labels)
                batch_val_preds.append(y_pred)

        batch_val_loss = batch_val_loss / len(val_index)


        batch_train_labels = np.concatenate(batch_train_labels)
        batch_train_preds = np.concatenate(batch_train_preds)
        batch_val_labels = np.concatenate(batch_val_labels)
        batch_val_preds = np.concatenate(batch_val_preds)
        train_micro_f1, train_macro_f1 = top_k_acc(batch_train_preds, batch_train_labels)
        val_micro_f1, val_macro_f1 = top_k_acc(batch_val_preds, batch_val_labels)
        print("Epoch %d, train loss %.5f, train micro-f1 %.5f, train macro-f1 %.5f, val loss %.5f, val micro-f1 %.5f, val macro-f1 %.5f"%(i, batch_train_loss, train_micro_f1, train_macro_f1, batch_val_loss, val_micro_f1, val_macro_f1))


        if True:
            if batch_val_loss < best:
                best = batch_val_loss
                wait = 0
                best_weights = model.get_weights()
                imporve = True
            else:
                imporve = False
        if imporve:
            stopped_epoch = i
        else:
            wait += 1
            if wait > patience:
                lr_dec += 1
                wait = 0
                if lr_dec > lr_patience:
                    print("Restoring model weights from the end of the best epoch %d."%stopped_epoch)
                    model.set_weights(best_weights)
                    print("Epoch %05d: early stopping"%(i))
                    break
                else:
                    if not hasattr(model.optimizer, "lr"):
                        raise ValueError('Optimizer must have a "lr" attribute.')
                    # Get the current learning rate from model's optimizer.
                    lr = float(tf.keras.backend.get_value(model.optimizer.learning_rate))
                    # Set the value back to the optimizer before this epoch starts
                    scheduled_lr = lr/10.0
                    tf.keras.backend.set_value(model.optimizer.lr, scheduled_lr)
                    print("\nEpoch %05d: Learning rate is %f." % (i, scheduled_lr))
                    model.set_weights(best_weights)


    embeddings = None
    if emb_flag:
        star_time = time.time()
        index_arr = np.array(range(num_nodes))
        embeddings = np.zeros(shape=(num_nodes, args.hidden_dim))
        k = 5
        for batch_index in batches(index_arr, args.batch_size):
            x = features[batch_index, :]

            tmp_embeddings = np.zeros(shape=(x.shape[0], args.hidden_dim))
            for i in range(k):
                nb_index = utils.sample_k_neighbors_from_adj(nb_sampled_adj, args.nb_size, batch_index, False)
                nb_index = np.reshape(nb_index, (-1))
                nb_x = features[nb_index, :]
                nb_x = np.reshape(nb_x, (x.shape[0], args.nb_size, x.shape[1]))
                emb = model.run_embedding((x, nb_x))
                emb = tf.keras.backend.eval(emb)
                tmp_embeddings = tmp_embeddings + emb
            tmp_embeddings = tmp_embeddings / float(k)
            embeddings[batch_index, :] = tmp_embeddings
        print("Load embedding cost %.2fs"%(time.time() - star_time))


    #batch_test_loss = 0.
    batch_test_labels = []
    batch_test_preds = []
    for batch_index in batches(test_index, args.batch_size):

        b_size = len(batch_index)

        x = features[batch_index, :]
        batch_labels = Y[batch_index, :]

        nb_index = utils.sample_k_neighbors_from_adj(nb_sampled_adj, args.nb_size, batch_index, False)
        nb_index = np.reshape(nb_index, (-1))
        nb_x = features[nb_index, :]
        nb_x = np.reshape(nb_x, (x.shape[0], args.nb_size, x.shape[1]))

        nbnb_index = utils.sample_k_neighbors_from_adj(nb_sampled_adj, args.nb_size, nb_index, False)
        nbnb_index = np.reshape(nbnb_index, (-1))
        nbnb_x = features[nbnb_index, :]
        nbnb_x = np.reshape(nbnb_x, (x.shape[0]*args.nb_size, args.nb_size, x.shape[1]))



        with tf.GradientTape() as tape:
            if args.model_name == "GCN":
                loss, acc, y_pred = model((x, nb_x, nbnb_x, batch_labels), training=False)
            elif args.model_name == "MLP":
                loss, acc, y_pred = model((x, batch_labels), training=False)

            batch_test_labels.append(batch_labels)
            batch_test_preds.append(y_pred)

            #batch_test_loss += loss * b_size

    batch_test_labels = np.concatenate(batch_test_labels)
    batch_test_preds = np.concatenate(batch_test_preds)
    test_micro_f1, test_macro_f1 = top_k_acc(batch_test_preds, batch_test_labels)
    #batch_test_loss = batch_test_loss / len(test_index)

    return val_micro_f1, val_macro_f1, test_micro_f1, test_macro_f1, embeddings

def clf_function(X, Y, data_index, args):

    train_index, val_index, test_index = data_index
    trainX, trainY = X[train_index, :], Y[train_index, :]
    valX, valY = X[val_index, :], Y[val_index, :]
    testX, testY = X[test_index, :], Y[test_index, :]

    print("load node classification data end")
    print("train data size " + str(trainX.shape))
    print("val data size " + str(valX.shape))
    print("test data size " + str(testX.shape))

    class_num = np.shape(trainY)[1]


    best_val_micro = 0.
    best_test_micro = 0.
    best_val_macro = 0.
    best_test_macro = 0.

    #for c in [1e-3, 1e-2, 1e-1, 1.0, 10., 100., 1000.]:
    for c in [1]:
        #clf = Classifier('SVM', class_num, c)
        clf = Classifier('LR', class_num)
        clf.train(trainX, trainY)
        val_result = clf.evaluate(valX, valY)
        print("C:%f, val result of %s-%d: micro F1 %.5f, macro F1 %.5f"%(c, args.model_name, args.dimension, val_result['micro'], val_result['macro']))
        result = clf.evaluate(testX, testY)
        print("C:%f, test result of %s-%d: micro F1 %.5f, macro F1 %.5f"%(c, args.model_name, args.dimension, result['micro'], result['macro']))
        if best_val_micro < val_result['micro']:
            best_val_micro = val_result['micro']
            best_test_micro = result['micro']
            best_val_macro = val_result['macro']
            best_test_macro = result['macro']
    #return val_result['micro'], val_result['macro'], result['micro'], result['macro']
    return best_val_micro, best_val_macro, best_test_micro, best_test_macro


def main(args):
    dataset_str = args.dataset
    model_str = args.model_name
    #data_path = args.data_path
    output_file = args.output
    emb_dimension = args.dimension


    #load douban data
    def load_file(fname):
        if not os.path.exists(fname):
            print("FILE NOT EXISTS ERROR !!")
            sys.exit(0)
        with open(fname, 'rb') as fin:
            res = pkl.load(fin)
            return res

    adj = load_file("data/%s/%s_adj.pkl"%(dataset_str, dataset_str))

    max_df = 1.0
    min_df = 5
    X = load_file("data/%s/%sX_%.1f_%.1f.pkl"%(dataset_str, dataset_str, max_df, min_df))

    genresY = np.array(load_file("data/%s/%s_genresY.pkl"%(dataset_str, dataset_str)))
    scoreY = np.array(load_file("data/%s/%s_scoreY.pkl"%(dataset_str, dataset_str)))
    short_commentsY = np.array(load_file("data/%s/%s_short_commentsY.pkl"%(dataset_str, dataset_str)))

    print("shape of graph [%d, %d]"%(adj.shape[0], adj.shape[1]))
    print("number of undirected edges %d"%(adj.nnz/2))
    print("shape of input feature [%d, %d]"%(X.shape[0], X.shape[1]))
    print("shape of label [%d, %d]"%(genresY.shape[0], genresY.shape[1]))
    sys.exit(0)

    train_size = int(X.shape[0] * args.train_label_ratio)
    test_size = int(X.shape[0] * args.test_label_ratio)
    val_size = X.shape[0] - train_size - test_size

    index_array = np.array(list(range(X.shape[0])))
    index_array = shuffle(index_array)

    train_index = index_array[:train_size]
    test_index = index_array[train_size:train_size+test_size]
    val_index = index_array[train_size+test_size:]


    task_res = []

    if args.trans:
        print("*************test on genres labels***************")
        v_mi, v_ma, t_mi, t_ma, embeddings = training_embedding(adj, X, genresY, [train_index, val_index, test_index], True, args, True)
        task_res.append([v_mi, v_ma, t_mi, t_ma])
        print("*************test on score labels****************")
        v_mi, v_ma, t_mi, t_ma = clf_function(embeddings, scoreY, [train_index, val_index, test_index], args)
        task_res.append([v_mi, v_ma, t_mi, t_ma])
        print("*************test on short_comments labels************")
        v_mi, v_ma, t_mi, t_ma = clf_function(embeddings, short_commentsY, [train_index, val_index, test_index], args)
        task_res.append([v_mi, v_ma, t_mi, t_ma])
    else:
        v_mi, v_ma, t_mi, t_ma, _ = training_embedding(adj, X, genresY, [train_index, val_index, test_index], False, args, True)
        task_res.append([v_mi, v_ma, t_mi, t_ma])
        v_mi, v_ma, t_mi, t_ma, _ = training_embedding(adj, X, scoreY, [train_index, val_index, test_index], False, args, False)
        task_res.append([v_mi, v_ma, t_mi, t_ma])
        v_mi, v_ma, t_mi, t_ma, _ = training_embedding(adj, X, short_commentsY, [train_index, val_index, test_index], False, args, False)
        task_res.append([v_mi, v_ma, t_mi, t_ma])

    return task_res




def multi_run(args):
    micro_f1 = []
    macro_f1 = []

    task1_res = []
    task2_res = []
    task3_res = []
    k = 1
    if args.hp:
        k = 3
    for i in range(k):
        #with tf.variable_scope("main", reuse=tf.AUTO_REUSE):
        #with tf.variable_scope("main_%d"%i):
        task_res = main(args)
        micro_f1.append(task_res[0][0])
        macro_f1.append(task_res[0][1])
        task1_res.append(task_res[0])
        task2_res.append(task_res[1])
        task3_res.append(task_res[2])

    task1_res = np.array(task1_res)
    task2_res = np.array(task2_res)
    task3_res = np.array(task3_res)

    def show_clf_result(res_array, task_name='Genres'):
        print("************Result on Task %s ********************"%task_name)
        mean = np.mean(res_array, axis=0)
        std = np.std(res_array, axis=0)
        print("Val mean loss %.5f, std %.5f, acc %.5f, std %.5f"%(mean[0], std[0], mean[1], std[1]))
        print("Test mean loss %.5f, std %.5f, acc %.5f, std %.5f"%(mean[2], std[2], mean[3], std[3]))
        print("********************************************************")

    show_clf_result(task1_res, 'Genres')
    show_clf_result(task2_res, 'Score')
    show_clf_result(task3_res, 'Short_comments')

    return np.mean(micro_f1), np.mean(macro_f1), np.std(micro_f1), np.std(macro_f1)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', type=str, default='douban', help='数据集名称')
    parser.add_argument("--model_name", type=str, default='LearnR', help='训练节点表示使用的模型')
    #parser.add_argument('--data_path', type=str, default='data', help='训练数据路径')
    parser.add_argument("--network_config", type=str, default="network_config.json", help="网络配置文件")
    parser.add_argument("-o", "--output", type=str, default='output.csv', help="输出文件名")

    parser.add_argument('--link_predict', dest='lp', default=False, action='store_true', help='run link prediction task')
    parser.add_argument('--node_classify', dest='clf', default=False, action='store_true', help='run node classification task')
    parser.add_argument('--embedding', dest='embedding', default=False, action='store_true', help='run node embedding')
    parser.add_argument('--dimension', type=int, default=512, help='dimension of embedding')


    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--epochs', type=int, default=3000, help='Number of epochs to train')
    parser.add_argument('--early_stop', type=int, default=10, help='number of step to decrease learning rate with the loss stops decreasing')
    parser.add_argument('--max_lr_dec', type=int, default=0, help='max number of decreasing learning rate')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')

    parser.add_argument('--hidden_dim', type=int, default=64, help='dimension of hidden embeddings')


    parser.add_argument('--l2_r', type=float, default=1e-5, help='l2 regularization on parameters')
    parser.add_argument('--negative_num', type=int, default=10, help='number of negative samples used in instance classify loss')
    parser.add_argument('--neighbor_hop', type=int, default=1, help='hop of neighbors using to sample negative sample in instance loss')
    parser.add_argument("--nb_size", type=int, default=5, help='number of neighbor nodes using in gcn aggregation')
    parser.add_argument("--batch_size", type=int, default=5, help='number of nodes in a batch')


    parser.add_argument("--train_label_ratio", type=float, default=0.7, help='ratio of labeled nodes in training')
    parser.add_argument("--test_label_ratio", type=float, default=0.2, help='ratio of labeled nodes in test')
    parser.add_argument("--set_label_dis", dest='set_label_dis', default=False, action='store_true', help='split dataset according to label sperately')
    parser.add_argument('--sample_node_label', dest='sample_node_label', default=False, action='store_true', help='random sample train/val/test label dataset')
    parser.add_argument("--hp", dest='hp', default=False, action='store_true', help='hyperparameters seclection flag')
    parser.add_argument("--enhance", dest='enhance', default=False, action='store_true', help='using augmentation node in downstream tasks.')
    parser.add_argument("--trans", dest='trans', default=False, action='store_true', help='trans GCN embedding to other tasks.')

    parser.add_argument("--gpuid", type=int, default=2, help='index of gpu')

    args = parser.parse_args()

    # Train on CPU (hide GPU) due to memory constraints
    os.environ['CUDA_VISIBLE_DEVICES'] = "%d"%args.gpuid
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    if args.hp:
    #if False:
        with RedirectStdStreams(stdout=sys.stderr):
            start_time = time.time()
            #mi_f1, ma_f1, _, _ = main()
            mi_f1, ma_f1, mi_f1_std, ma_f1_std = multi_run(args)
            if args.lp:
                print("mean auc %.5f, std %.5f, ap %.5f, std %.5f"%(mi_f1, mi_f1_std, ma_f1, ma_f1_std))
            else:
                print("mean micro f1 %.5f, std %.5f, macro f1 %.5f, std %.5f"%(mi_f1, mi_f1_std, ma_f1, ma_f1_std))
            print("train and evaluation this model totally costs %f seconds"%(time.time()-start_time))
        print('%.4f, %.4f' %(mi_f1, ma_f1))
    else:
        start_time = time.time()
        #mi_f1, ma_f1, _, _ = main()
        mi_f1, ma_f1, mi_f1_std, ma_f1_std = multi_run(args)
        if args.lp:
            print("mean auc %.5f, std %.5f, ap %.5f, std %.5f"%(mi_f1, mi_f1_std, ma_f1, ma_f1_std))
        else:
            print("mean micro f1 %.5f, std %.5f, macro f1 %.5f, std %.5f"%(mi_f1, mi_f1_std, ma_f1, ma_f1_std))
        print("train and evaluation this model totally costs %f seconds"%(time.time()-start_time))

