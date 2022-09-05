import tensorflow as tf
from tensorflow import keras
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

import time
import os
import shutil
import sys

from doubanmodel import FowardR, FixR, HyperR, LearnR, LearnSimpleR, AdvR
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


def training_embedding(gdata, args):

    sp_input_flag = False

    tf.keras.backend.set_floatx('float64')

    def get_hyper_radius(adj, k_hop, n):
        ppmi = utils.PPMI(adj, k_hop)
        max_pmi = ppmi.max()
        ppmi.data = ppmi.data / max_pmi
        radius = ppmi.max(axis=1)
        radius = 1.0 - radius.toarray()
        return radius

    def get_adv_hyper_radius(adj, k_hop, n):
        ppmi = utils.PPMI(adj, k_hop)
        max_pmi = ppmi.max()
        ppmi.data = ppmi.data / max_pmi
        ppmi.data = 1.0 - ppmi.data
        return ppmi.tocsr()

    features = gdata[1].astype(np.float64)
    adj = gdata[0].astype(np.float64)
    #print(adj.nnz)
    #sys.exit(0)

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

    star_time = time.time()
    if args.model_name == 'HyperR':
        if os.path.exists("temp/%s_HyperR.pkl"%args.dataset):
            with open("temp/%s_HyperR.pkl"%args.dataset, 'rb') as fin:
                hyper_radius = pkl.load(fin)
        else:
            hyper_radius = get_hyper_radius(adj, 2, num_nodes)
            with open("temp/%s_HyperR.pkl"%args.dataset, 'wb') as fout:
                pkl.dump(hyper_radius, fout)
    if args.model_name == 'AdvR':
        if os.path.exists("temp/%s_AdvR.pkl"%args.dataset):
            with open("temp/%s_AdvR.pkl"%args.dataset, 'rb') as fin:
                hyper_radius = pkl.load(fin)
        else:
            hyper_radius = get_adv_hyper_radius(adj, 2, num_nodes)
            with open("temp/%s_AdvR.pkl"%args.dataset, 'wb') as fout:
                pkl.dump(hyper_radius, fout)
    print("cal radius costs %.2fs"%(time.time()-star_time))




    nb_sampled_adj = adj.copy()
    for i in range(args.neighbor_hop-1):
       nb_sampled_adj = nb_sampled_adj.dot(adj)


    with open(args.network_config, 'r') as f:
        network_config = json.load(f)

    gnn_units_list = network_config['gnn_units_list']
    dnn_units_list = network_config['dnn_units_list']

    if args.model_name == 'FowardR':
        model = FowardR(input_dim=num_features, gnn_units_list=gnn_units_list, sparse_inputs=sp_input_flag, act=tf.nn.relu, dropout=args.dropout, bias=False, l2_r=args.l2_r,
                 negative_num=args.negative_num, aug_gae_loss_w=args.aug_gae_loss_w, ins_loss_w=args.ins_loss_w, learning_rate=args.learning_rate, temperature=args.temperature, augment_num=args.augment_num,
                 norm_loss_w=args.norm_loss_w, hinge_loss_w=args.hinge_loss_w,
            )
    elif args.model_name == 'FixR':
        model = FixR(input_dim=num_features, gnn_units_list=gnn_units_list, sparse_inputs=sp_input_flag, act=tf.nn.relu, dropout=args.dropout, bias=False, l2_r=args.l2_r,
                 negative_num=args.negative_num, aug_gae_loss_w=args.aug_gae_loss_w, ins_loss_w=args.ins_loss_w, learning_rate=args.learning_rate, temperature=args.temperature, augment_num=args.augment_num,
                 norm_loss_w=args.norm_loss_w, hinge_loss_w=args.hinge_loss_w,
                 )
    elif args.model_name == 'HyperR':
        model = HyperR(input_dim=num_features, gnn_units_list=gnn_units_list, sparse_inputs=sp_input_flag, act=tf.nn.relu, dropout=args.dropout, bias=False, l2_r=args.l2_r,
                 negative_num=args.negative_num, aug_gae_loss_w=args.aug_gae_loss_w, ins_loss_w=args.ins_loss_w, learning_rate=args.learning_rate, temperature=args.temperature, augment_num=args.augment_num,
                 norm_loss_w=args.norm_loss_w, hinge_loss_w=args.hinge_loss_w,
                 )
    elif args.model_name == 'LearnR':
        model = LearnR(input_dim=num_features, gnn_units_list=gnn_units_list, sparse_inputs=sp_input_flag, act=tf.nn.relu, dropout=args.dropout, bias=False, l2_r=args.l2_r,
                 negative_num=args.negative_num, aug_gae_loss_w=args.aug_gae_loss_w, ins_loss_w=args.ins_loss_w, learning_rate=args.learning_rate, temperature=args.temperature, augment_num=args.augment_num,
                 norm_loss_w=args.norm_loss_w, hinge_loss_w=args.hinge_loss_w,
                 )
    elif args.model_name == 'LearnSimpleR':
        model = LearnSimpleR(input_dim=num_features, gnn_units_list=gnn_units_list, sparse_inputs=sp_input_flag, act=tf.nn.relu, dropout=args.dropout, bias=False, l2_r=args.l2_r,
                 negative_num=args.negative_num, aug_gae_loss_w=args.aug_gae_loss_w, ins_loss_w=args.ins_loss_w, learning_rate=args.learning_rate, temperature=args.temperature, augment_num=args.augment_num,
                 norm_loss_w=args.norm_loss_w, hinge_loss_w=args.hinge_loss_w,
                 )
    elif args.model_name == 'AdvR':
        model = AdvR(input_dim=num_features, gnn_units_list=gnn_units_list, sparse_inputs=sp_input_flag, act=tf.nn.relu, dropout=args.dropout, bias=False, l2_r=args.l2_r,
                 negative_num=args.negative_num, aug_gae_loss_w=args.aug_gae_loss_w, ins_loss_w=args.ins_loss_w, learning_rate=args.learning_rate, temperature=args.temperature, augment_num=args.augment_num,
                 norm_loss_w=args.norm_loss_w, hinge_loss_w=args.hinge_loss_w,
                 )
    else:
        assert False, "No such model"

    model.compile()


    # The number of epoch it has waited when loss is no longer minimum.
    wait = 0
    # The number of learning rate decreasing,
    lr_dec = 0
    # The epoch the training stops at.
    stopped_epoch = 0
    # Initialize the best as infinity.
    best = np.Inf
    # store best weights
    best_weights = None
    #initialize the link predict acc as zero(use to stop GAE based model)
    best_acc = 0.
    #flag of imporve loss or link prediction acc
    imporve = True

    patience = args.early_stop
    lr_patience = args.max_lr_dec

    last_gradint_dir = np.random.normal(size=(num_nodes, gnn_units_list[-1]))

    index_arr = np.array(range(num_nodes))

    #partial training model
    def batches(l, n):
        for ti in range(0, len(l), n):
            yield l[ti:ti+n]


    #debug
    print("start training")
    for i in range(args.epochs):
        star_time = time.time()

        iter_loss = 0.
        iter_gae_loss = 0.
        iter_aug_gae_loss = 0.
        iter_ins_loss = 0.
        iter_norm_loss = 0.

        batch_iter_num = 0.

        index_arr = shuffle(index_arr)
        for batch_index in batches(index_arr, args.batch_size):

            x = features[batch_index, :]

            nb_index = utils.sample_k_neighbors_from_adj(nb_sampled_adj, args.nb_size, batch_index, False)
            nb_index = np.reshape(nb_index, (-1))
            nb_x = features[nb_index, :]
            nb_x = np.reshape(nb_x, (x.shape[0], args.nb_size, x.shape[1]))

            nbnb_index = utils.sample_k_neighbors_from_adj(nb_sampled_adj, args.nb_size, nb_index, False)
            nbnb_index = np.reshape(nbnb_index, (-1))
            nbnb_x = features[nbnb_index, :]
            nbnb_x = np.reshape(nbnb_x, (x.shape[0]*args.nb_size, args.nb_size, x.shape[1]))

            gd = last_gradint_dir[batch_index, :]

            if args.model_name == 'AdvR':
                nb_gd = last_gradint_dir[nb_index, :]
                nb_gd = np.reshape(nb_gd, (x.shape[0], args.nb_size, -1))

                nb_index_tmp = np.reshape(nb_index, (x.shape[0], -1))
                batch_hyper_radius = []
                for advi, binx in enumerate(batch_index.tolist()):
                    batch_hyper_radius.append(hyper_radius[binx, :].toarray()[:, nb_index_tmp[advi]])
                batch_hyper_radius = np.squeeze(np.array(batch_hyper_radius))
            elif args.model_name == 'HyperR':
                batch_hyper_radius = hyper_radius[batch_index, :]


            with tf.GradientTape() as tape:
                if args.model_name == 'HyperR':
                    loss_list, _, _ = model((x, nb_x, nbnb_x, gd, batch_hyper_radius), training=True)
                elif args.model_name == 'AdvR':
                    loss_list, _, _ = model((x, nb_x, nbnb_x, gd, nb_gd, batch_hyper_radius), training=True)
                else:
                    loss_list, _, _ = model((x, nb_x, nbnb_x, gd), training=True)
                if args.model_name == 'AdvR':
                    batch_gradint_dir, batch_nb_gradint_dir = tf.keras.backend.eval(tape.gradient(loss_list[1], [model.h, model.nb_h]))
                    batch_nb_gradint_dir = tf.reshape(batch_nb_gradint_dir, shape=(nb_index.shape[0], -1))
                    last_gradint_dir[nb_index, :] = batch_nb_gradint_dir
                    last_gradint_dir[batch_index, :] = batch_gradint_dir
                else:
                    batch_gradint_dir = tf.keras.backend.eval(tape.gradient(loss_list[1], model.h))
                    last_gradint_dir[batch_index, :] = batch_gradint_dir
                #print(last_gradint_dir)

            with tf.GradientTape() as tape:
                if args.model_name == 'HyperR':
                    loss_list, _, _ = model((x, nb_x, nbnb_x, gd, batch_hyper_radius), training=True)
                elif args.model_name == 'AdvR':
                    loss_list, _, _ = model((x, nb_x, nbnb_x, gd, nb_gd, batch_hyper_radius), training=True)
                else:
                    loss_list, _, _ = model((x, nb_x, nbnb_x, gd), training=True)

                loss = loss_list[0]
                trainable_vars = model.trainable_variables
                gradients = tape.gradient(loss, trainable_vars)
                #gradients, last_gradint_dir = tape.gradient([loss, loss_list[3]], [trainable_vars, model.h])
                #print(last_gradint_dir)

                model.optimizer.apply_gradients(zip(gradients, trainable_vars))

            if args.model_name == 'AdvR':
                print("Epoch %d, Batch %d: Loss %f, GAE %f, AUG GAE %f"%(i, batch_iter_num, loss_list[0], loss_list[1], loss_list[2]))
                iter_loss += loss_list[0]
                iter_gae_loss += loss_list[1]
                iter_aug_gae_loss += loss_list[2]

            else:
                print("Epoch %d, Batch %d: Loss %f, GAE %f, AUG GAE %f, Ins %f, Norm %f"%(i, batch_iter_num, loss_list[0], loss_list[1], loss_list[2], loss_list[3], loss_list[5]))
                iter_loss += loss_list[0]
                iter_gae_loss += loss_list[1]
                iter_aug_gae_loss += loss_list[2]
                iter_ins_loss += loss_list[3]
                iter_norm_loss += loss_list[5]

            batch_iter_num += 1

        if args.model_name == 'AdvR':
            print("Epoch %d Cost %.2fs: Loss %f, GAE %f, AUG GAE %f"%(i, time.time() - star_time, iter_loss, iter_gae_loss, iter_aug_gae_loss))
        else:
            print("Epoch %d Cost %.2fs: Loss %f, GAE %f, AUG GAE %f, Ins %f, Norm %f"%(i, time.time() - star_time, iter_loss, iter_gae_loss, iter_aug_gae_loss, iter_ins_loss, iter_norm_loss))


        if True:
            if iter_loss < best:
                best = iter_loss
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


    star_time = time.time()
    index_arr = np.array(range(num_nodes))
    embeddings = np.zeros(shape=(num_nodes, gnn_units_list[-1]))
    k = 5
    for batch_index in batches(index_arr, args.batch_size):
        x = features[batch_index, :]

        tmp_embeddings = np.zeros(shape=(x.shape[0], gnn_units_list[-1]))
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
    aug_emb = None
    return embeddings, aug_emb

def clf_function(embeddings, gdata, args):
    X, Y = gdata[0], gdata[1]


    trainX, testX, trainY, testY = train_test_split(X, Y, test_size=args.test_label_ratio, random_state=1)

    conditional_train_ratio = args.train_label_ratio / (1.0 - args.test_label_ratio)
    trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size=1.0 - conditional_train_ratio, random_state=1)


    print("load node classification data end")
    print("train data size " + str(trainX.shape))
    print("val data size " + str(valX.shape))
    print("test data size " + str(testX.shape))

    class_num = np.shape(trainY)[1]

    ##using tensorflow linear model
    #model = keras.Sequential(
    #[
    #    #keras.layers.Dense(3, activation="relu", name="layer1"),
    #    keras.layers.Dense(class_num, name="layer2"),
    #]
    #)
    #model.compile(
    #optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
    #loss=keras.losses.SparseCategoricalCrossentropy(),
    #metrics=[keras.metrics.SparseCategoricalAccuracy()],
    #)

    #model.fit(trainX, np.argmax(trainY, axis=1), epochs=1000, verbose=10,
    #         callbacks=[LossAndErrorPrintingCallback(), EarlyStoppingAtMinLoss()])
    #val_res = {}
    #result = {}
    #val_res = model.evaluate(testX, np.argmax(valY, axis=1))
    #val_acc = val_res[1]
    #test_res = model.evaluate(testX, np.argmax(testY, axis=1))
    #test_acc = test_res[1]
    #val_result['micro'] = val_acc
    #val_result['macro'] = 0.0
    #result['micro'] = test_acc
    #result['macro'] = 0.0


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

def enhance_clf_function(embeddings_list, gdata, args):
    trainX_indices, trainY = gdata.get_node_classification_data(name='train')
    valX_indices, valY = gdata.get_node_classification_data(name='val')
    testX_indices, testY = gdata.get_node_classification_data(name='test')

    if args.sample_node_label and valX_indices:
        testX_indices = np.concatenate((testX_indices, valX_indices), axis=0)
        testY = np.concatenate((testY, valY), axis=0)

    trainX = np.concatenate([embeddings[trainX_indices] for embeddings in embeddings_list], axis=0)
    trainY = np.concatenate([trainY, trainY], axis=0)
    valX = embeddings_list[0][valX_indices]
    testX = embeddings_list[0][testX_indices]

    print("load node classification data end")
    print("train data size " + str(trainX.shape))
    print("val data size " + str(valX.shape))
    print("test data size " + str(testX.shape))

    class_num = np.shape(trainY)[1]
    #sample_num = np.shape(trainY)[0]
    #idx = shuffle(range(sample_num))
    #trainX = trainX[idx]
    #trainY = trainY[idx]

    ##using tensorflow linear model
    #model = keras.Sequential(
    #[
    #    #keras.layers.Dense(3, activation="relu", name="layer1"),
    #    keras.layers.Dense(class_num, name="layer2"),
    #]
    #)
    #model.compile(
    #optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
    #loss=keras.losses.SparseCategoricalCrossentropy(),
    #metrics=[keras.metrics.SparseCategoricalAccuracy()],
    #)

    #model.fit(trainX, np.argmax(trainY, axis=1), epochs=1000, verbose=10,
    #         callbacks=[LossAndErrorPrintingCallback(), EarlyStoppingAtMinLoss()])
    #val_res = {}
    #result = {}
    #val_res = model.evaluate(testX, np.argmax(valY, axis=1))
    #val_acc = val_res[1]
    #test_res = model.evaluate(testX, np.argmax(testY, axis=1))
    #test_acc = test_res[1]
    #val_result['micro'] = val_acc
    #val_result['macro'] = 0.0
    #result['micro'] = test_acc
    #result['macro'] = 0.0

    #clf = Classifier('SVM', class_num)
    clf = Classifier('LR', class_num)
    clf.train(trainX, trainY)
    if not valX_indices is None:
        val_result = clf.evaluate(valX, valY)
        print("val result of %s-%d: micro F1 %.5f, macro F1 %.5f"%(args.model_name, args.dimension, val_result['micro'], val_result['macro']))
    else:
        val_result = {}
        val_result['micro'], val_result['macro'] = 0.0, 0.0
    result = clf.evaluate(testX, testY)
    print("test result of %s-%d: micro F1 %.5f, macro F1 %.5f"%(args.model_name, args.dimension, result['micro'], result['macro']))
    return val_result['micro'], val_result['macro'], result['micro'], result['macro']

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
    print("shape of input feature [%d, %d]"%(X.shape[0], X.shape[1]))
    print("shape of label [%d, %d]"%(genresY.shape[0], genresY.shape[1]))


    if args.embedding:
        embeddings, aug_emb = training_embedding([adj, X], args)
        node_num = X.shape[0]


        header = "%d %d"%(node_num, emb_dimension)
        if not os.path.exists('emb/%s'%dataset_str):
            os.makedirs('emb/%s'%dataset_str)
        femb = 'emb/%s/%s-%d.txt'%(dataset_str, model_str, emb_dimension)
        if embeddings is None:
            print("embedding load error")
        else:
            utils.save_embedding(femb, embeddings, delimiter=' ', header=header)
            #utils.save_embedding(femb, embeddings, delimiter=',', header=header)
    else:
        femb = 'emb/%s/%s-%d.txt'%(dataset_str, model_str, emb_dimension)
        embeddings = utils.load_embedding(femb, delimiter=' ')
        print("load embedding end, emb shape " + str(embeddings.shape))


    task_res = []
    print("*************test on genres labels***************")
    v_mi, v_ma, t_mi, t_ma = clf_function(embeddings, [embeddings, genresY], args)
    task_res.append([v_mi, v_ma, t_mi, t_ma])
    print("*************test on score labels****************")
    v_mi, v_ma, t_mi, t_ma = clf_function(embeddings, [embeddings, scoreY], args)
    task_res.append([v_mi, v_ma, t_mi, t_ma])
    print("*************test on short_comments labels************")
    v_mi, v_ma, t_mi, t_ma = clf_function(embeddings, [embeddings, short_commentsY], args)
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
        #k = 5
        k = 1
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
        print("Val mean micro f1 %.5f, std %.5f, macro f1 %.5f, std %.5f"%(mean[0], std[0], mean[1], std[1]))
        print("Test mean micro f1 %.5f, std %.5f, macro f1 %.5f, std %.5f"%(mean[2], std[2], mean[3], std[3]))
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
    parser.add_argument('--early_stop', type=int, default=5, help='number of step to decrease learning rate with the loss stops decreasing')
    parser.add_argument('--max_lr_dec', type=int, default=0, help='max number of decreasing learning rate')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
    parser.add_argument('--temperature', type=float, default=1., help='temperature parameters in instance classify')

    parser.add_argument('--aug_dgi_loss_w', type=float, default=0.0, help='Deep Graph infomax on augmentation node')
    parser.add_argument('--ins_loss_w', type=float, default=0.0, help='instance classify loss, used to train node augmentation operator')
    parser.add_argument('--l2_r', type=float, default=0., help='l2 regularization on parameters')
    parser.add_argument('--norm_loss_w', type=float, default=0., help='norm constraints weight on node embeddings and augmentation embedddings')
    parser.add_argument('--negative_num', type=int, default=10, help='number of negative samples used in instance classify loss')
    parser.add_argument('--siamese_loss_w', type=float, default=0., help='using siamese loss to classify real node and its augmentation node')
    parser.add_argument('--siamese_pos_w', type=float, default=10, help='weight of positive samples in siamese loss')
    parser.add_argument('--augment_num', type=int, default=1, help='number of augmentation node sampled per step of train per node')
    parser.add_argument('--neighbor_hop', type=int, default=1, help='hop of neighbors using to sample negative sample in instance loss')
    parser.add_argument("--nb_size", type=int, default=5, help='number of neighbor nodes using in gcn aggregation')
    parser.add_argument("--batch_size", type=int, default=5, help='number of nodes in a batch')

    parser.add_argument("--number_of_walks", type=int, default=10, help='number of randomwalk sampled per node')
    parser.add_argument("--walk_length", type=int, default=10, help='length of per randomwalk path')
    parser.add_argument("--window_size", type=int, default=5, help='window size of skipgram model')
    parser.add_argument("--dw_neg_num", type=int, default=10, help='number of negative node in training skipgram model')
    parser.add_argument("--dw_batch_size", type=int, default=50, help='batch size in training deepwalk model')
    parser.add_argument("--aug_dw_loss_w", type=float, default=0.0, help='weight of skipgram with augmentation nodes')
    parser.add_argument("--aug_gae_loss_w", type=float, default=0.0, help='weight of GAE loss with augmentation nodes')
    parser.add_argument("--hinge_loss_w", type=float, default=0.0, help='weight of hinge loss(to keep node indentity)')

    parser.add_argument("--train_label_ratio", type=float, default=0.7, help='ratio of labeled nodes in training')
    parser.add_argument("--test_label_ratio", type=float, default=0.2, help='ratio of labeled nodes in test')
    parser.add_argument("--set_label_dis", dest='set_label_dis', default=False, action='store_true', help='split dataset according to label sperately')
    parser.add_argument('--sample_node_label', dest='sample_node_label', default=False, action='store_true', help='random sample train/val/test label dataset')
    parser.add_argument("--hp", dest='hp', default=False, action='store_true', help='hyperparameters seclection flag')
    parser.add_argument("--enhance", dest='enhance', default=False, action='store_true', help='using augmentation node in downstream tasks.')

    parser.add_argument("--gpuid", type=int, default=2, help='index of gpu')

    args = parser.parse_args()

    # Train on CPU (hide GPU) due to memory constraints
    os.environ['CUDA_VISIBLE_DEVICES'] = "%d"%args.gpuid
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(gpu, file=sys.stderr)

    if args.hp:
        with RedirectStdStreams(stdout=sys.stderr):
            start_time = time.time()
            #mi_f1, ma_f1, _, _ = main()
            mi_f1, ma_f1, mi_f1_std, ma_f1_std = multi_run(args)
            if args.lp:
                print("mean auc %.5f, std %.5f, ap %.5f, std %.5f"%(mi_f1, mi_f1_std, ma_f1, ma_f1_std))
            else:
                print("mean micro f1 %.5f, std %.5f, macro f1 %.5f, std %.5f"%(mi_f1, mi_f1_std, ma_f1, ma_f1_std))
            print("train and evaluation this model totally costs %f seconds"%(time.time()-start_time))
        print('(%.4f, %.4f)' %(mi_f1, ma_f1))
    else:
        start_time = time.time()
        #mi_f1, ma_f1, _, _ = main()
        mi_f1, ma_f1, mi_f1_std, ma_f1_std = multi_run(args)
        if args.lp:
            print("mean auc %.5f, std %.5f, ap %.5f, std %.5f"%(mi_f1, mi_f1_std, ma_f1, ma_f1_std))
        else:
            print("mean micro f1 %.5f, std %.5f, macro f1 %.5f, std %.5f"%(mi_f1, mi_f1_std, ma_f1, ma_f1_std))
        print("train and evaluation this model totally costs %f seconds"%(time.time()-start_time))

