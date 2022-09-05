import tensorflow as tf
from tensorflow import keras
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

import time
import os
import shutil
import sys

from models import GraphCLR, GraphGaussianCLR, GraphSiameseCLR, LossAndErrorPrintingCallback, EarlyStoppingAtMinLoss
from models import Deepwalk, GAECLR, GAEGaussianCLR, GAESiameseCLR, DNNFeatureCLR, GAE
from models import GAEAdvID
from tasks import Classifier, Cluster, LinkPredictor
import utils
from utils import Graphdata
from utils import ConfigerLoader

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle
from sklearn.preprocessing import label_binarize, LabelBinarizer
from sklearn.multiclass import OneVsRestClassifier

import randomwalk
from randomwalk import RandomWalk

import networkx as nx
import argparse
import json

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"


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
    sperate_name = 'all'
    if args.lp:
        val_pos_edges, val_neg_edges = gdata.get_link_prediction_data(data_name='val')
        sperate_name = 'train'
        

        #debug
        #train_edges = gdata.g.graph['%s_edges'%sperate_name]
        ##print("train edges %d"%len(train_edges))
        #adj = gdata.get_adj('all').astype(np.float32)
        #print(adj.nnz)
        #adj = adj.toarray()

        #for edge in train_edges:
        #    if not adj[edge[0]][edge[1]]:
        #        print(edge)
        #        print("ERROR")
        #        sys.exit(0)
        #print("SUCC")
        #sys.exit(0)
        #print("pos num %d"%len(val_pos_edges))
        #print("neg num %d"%len(val_neg_edges))
        #adj = gdata.get_adj('all').astype(np.float32)
        #adj = adj.toarray()
        #for edge in val_pos_edges:
        #    if not adj[edge[0]][edge[1]]:
        #        print(edge)
        #        print("ERROR!")
        #        sys.exit(0)
        #print("SUCC")
        #for edge in val_neg_edges:
        #    if adj[edge[0]][edge[1]]:
        #        print(edge)
        #        print("Error")
        #        sys.exit(0)
        #print("SUCC")
        #sys.exit(0)


    trainG = gdata.get_graph(sperate_name)
    #cnc = nx.connected_components(trainG)
    cnc = sorted(nx.connected_components(trainG), key = len, reverse=True)[0]
    cnc = list(cnc)
    #print(cnc)
    #for c in cnc:
    #    print(c)
    #sys.exit(0)

    features = gdata.get_features().astype(np.float32)
    sta_information = gdata.get_static_information()
    num_features = sta_information[0]
    adj = gdata.get_adj(sperate_name).astype(np.float32)

    features = features.toarray()[cnc]
    adj = adj.toarray()[cnc][:,cnc]
    adj = sp.csr_matrix(adj)
    features = sp.csr_matrix(features)

    #adj = utils.tuple_to_sparse(utils.preprocess_graph(adj))
    #adj = utils.tuple_to_sparse(utils.preprocess_features(utils.laplacian_adj(adj)))
    num_nodes = adj.shape[0]

    f, s, vt = svds(features, k=50)
    
    t = 10000
    for i in range(t):
        f = adj.dot(f)
        std = np.std(f, axis=0)
        print(std)
    print(f[1250])
    print(f[1251])
    print(f[1252])

    f, s, vt = svds(features, k=50)
    print(f[1250])
    print(f[1251])
    print(f[1252])
    sys.exit(0)

    if 'GAE' in args.model_name:
        adj_pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

        adj_orig = adj.copy()
        adj_orig = utils.sparse_to_tuple(adj_orig)
        adj_orig = tf.sparse.SparseTensor(adj_orig[0], adj_orig[1], adj_orig[2])

    neg_sampled_adj = adj.copy()
    for i in range(args.neighbor_hop-1):
    	neg_sampled_adj = neg_sampled_adj.dot(adj)

    if args.model_name in ['Deepwalk', 'DeepwalkCLR', 'DeepwalkSiamese', 'DeepwalkGaussian']:
        G = nx.from_scipy_sparse_matrix(adj)
        rw = RandomWalk(G)
        path_corpus = rw.simulate_walk(args.number_of_walks, args.walk_length, list(range(num_nodes)))
        path_corpus = np.reshape(np.array(path_corpus), (num_nodes, args.number_of_walks, -1))
        training_idx = list(range(num_nodes))


    if 'GAE' in args.model_name:
        features = features
    else:
        features = utils.tuple_to_sparse(utils.preprocess_features(features))
    adj = utils.tuple_to_sparse(utils.preprocess_graph(adj))

    corp_features = features.copy()
    corp_adj = adj.copy()


    features = utils.sparse_to_tuple(features)
    adj = utils.sparse_to_tuple(adj)



    with open(args.network_config, 'r') as f:
        network_config = json.load(f)

    gnn_units_list = network_config['gnn_units_list']
    dnn_units_list = network_config['dnn_units_list']

    if args.model_name == 'GraphCLR':
        model = GraphCLR(
                 input_dim=num_features, gnn_units_list=gnn_units_list, dnn_units_list=dnn_units_list,
                 sparse_inputs=True, act=tf.nn.relu, dropout=args.dropout, bias=False, l2_r=args.l2_r,
                 dao_categories_num=5, dao_continue_maxval=1, dao_continue_minval=-1, noise_mean=0., noise_stddev=1.0, noise_dim=10,
                 negative_num=args.negative_num, aug_dgi_loss_w=args.aug_dgi_loss_w, ins_loss_w=args.ins_loss_w, learning_rate=args.learning_rate,
                 temperature=args.temperature, norm_loss_w=args.norm_loss_w, augment_num=args.augment_num, hinge_loss_w=args.hinge_loss_w,
            )
    elif args.model_name == 'GraphGaussianCLR':
        model = GraphGaussianCLR(
                 input_dim=num_features, gnn_units_list=gnn_units_list,
                 sparse_inputs=True, act=tf.nn.relu, dropout=args.dropout, bias=False, l2_r=args.l2_r,
                 negative_num=args.negative_num, aug_dgi_loss_w=args.aug_dgi_loss_w, ins_loss_w=args.ins_loss_w, learning_rate=args.learning_rate,
                 temperature=args.temperature, augment_num=args.augment_num, norm_loss_w=args.norm_loss_w, hinge_loss_w=args.hinge_loss_w,
            )
    elif args.model_name == 'GraphSiameseCLR':
        model = GraphSiameseCLR(
                 input_dim=num_features, gnn_units_list=gnn_units_list, dnn_units_list=dnn_units_list,
                 sparse_inputs=True, act=tf.nn.relu, dropout=args.dropout, bias=False, l2_r=args.l2_r,
                 dao_categories_num=5, dao_continue_maxval=1, dao_continue_minval=-1, noise_mean=0., noise_stddev=1.0, noise_dim=10,
                 negative_num=args.negative_num, aug_dgi_loss_w=args.aug_dgi_loss_w, siamese_loss_w=args.siamese_loss_w, learning_rate=args.learning_rate,
                 norm_loss_w=args.norm_loss_w, siamese_pos_w=args.siamese_pos_w, augment_num=args.augment_num
            )
    elif args.model_name in ['Deepwalk', 'DeepwalkCLR', 'DeepwalkSiamese', 'DeepwalkGaussian']:
        model = Deepwalk(
        node_num=num_nodes, emb_dim=args.dimension, dw_neg_num=args.dw_neg_num, dnn_units_list=dnn_units_list, learning_rate=args.learning_rate,
        ins_loss_w=args.ins_loss_w, siamese_loss_w=args.siamese_loss_w, model_name=args.model_name, siamese_pos_w=args.siamese_pos_w, l2_r=args.l2_r,
        temperature=args.temperature, negative_num=args.negative_num, dropout=args.dropout, norm_loss_w=args.norm_loss_w
        )
    elif args.model_name in ['DNNFeatureCLR', 'DNNFeatureSiameseCLR', 'DNNFeatureGaussianCLR']:
        model = DNNFeatureCLR(gnn_units_list=gnn_units_list, dnn_units_list=dnn_units_list, learning_rate=args.learning_rate, negative_num=args.negative_num, act=tf.nn.relu,
                ins_loss_w=args.ins_loss_w, siamese_loss_w=args.siamese_loss_w, model_name=args.model_name, dropout=args.dropout,
                dao_categories_num=5, dao_continue_maxval=1, dao_continue_minval=-1, noise_mean=0., noise_stddev=1.0, noise_dim=10,
                siamese_pos_w=args.siamese_pos_w, l2_r=args.l2_r, temperature=args.temperature)
    elif args.model_name == 'GAECLR':
        model = GAECLR(input_dim=num_features, gnn_units_list=gnn_units_list, dnn_units_list=dnn_units_list, sparse_inputs=True, act=tf.nn.relu, dropout=args.dropout, bias=False, l2_r=args.l2_r,
                 negative_num=args.negative_num, aug_gae_loss_w=args.aug_gae_loss_w, ins_loss_w=args.ins_loss_w, learning_rate=args.learning_rate, temperature=args.temperature, augment_num=args.augment_num,
                 dao_categories_num=5, dao_continue_maxval=1, dao_continue_minval=-1, noise_mean=0., noise_stddev=1.0, noise_dim=10, hinge_loss_w=args.hinge_loss_w,
                 norm=norm, adj_pos_weight=adj_pos_weight)
    elif args.model_name == 'GAE':
        model = GAE(input_dim=num_features, gnn_units_list=gnn_units_list, sparse_inputs=True, act=tf.nn.relu, dropout=args.dropout, bias=False, l2_r=args.l2_r,
                learning_rate=args.learning_rate, adj_pos_weight=adj_pos_weight, norm=norm)
    elif args.model_name == 'GAEGaussianCLR':
        model = GAEGaussianCLR(input_dim=num_features, gnn_units_list=gnn_units_list, sparse_inputs=True, act=tf.nn.relu, dropout=args.dropout, bias=False, l2_r=args.l2_r,
                 negative_num=args.negative_num, aug_gae_loss_w=args.aug_gae_loss_w, ins_loss_w=args.ins_loss_w, learning_rate=args.learning_rate, temperature=args.temperature, augment_num=args.augment_num,
                 norm=norm, adj_pos_weight=adj_pos_weight, norm_loss_w=args.norm_loss_w, hinge_loss_w=args.hinge_loss_w,
            )
    elif args.model_name == 'GAEAdvID':
        model = GAEAdvID(input_dim=num_features, gnn_units_list=gnn_units_list, sparse_inputs=True, act=tf.nn.relu, dropout=args.dropout, bias=False, l2_r=args.l2_r,
                 negative_num=args.negative_num, aug_gae_loss_w=args.aug_gae_loss_w, ins_loss_w=args.ins_loss_w, learning_rate=args.learning_rate, temperature=args.temperature, augment_num=args.augment_num,
                 norm=norm, adj_pos_weight=adj_pos_weight, norm_loss_w=args.norm_loss_w, hinge_loss_w=args.hinge_loss_w,
                 )
        last_gradint_dir = np.random.normal(size=(num_nodes, gnn_units_list[-1]))
    elif args.model_name == 'GAESiameseCLR':
        model = GAESiameseCLR(input_dim=num_features, gnn_units_list=gnn_units_list, dnn_units_list=dnn_units_list, sparse_inputs=True, act=tf.nn.relu, dropout=args.dropout, bias=False, l2_r=args.l2_r,
                 negative_num=args.negative_num, aug_gae_loss_w=args.aug_gae_loss_w, siamese_loss_w=args.siamese_loss_w, learning_rate=args.learning_rate, temperature=args.temperature, augment_num=args.augment_num,
                 dao_categories_num=5, dao_continue_maxval=1, dao_continue_minval=-1, noise_mean=0., noise_stddev=1.0, noise_dim=10,
                 norm=norm, adj_pos_weight=adj_pos_weight, siamese_pos_w=args.siamese_pos_w
            )
    else:
        assert False, "No such model"

    model.compile()

    features = tf.sparse.SparseTensor(features[0], features[1], features[2])
    adj = tf.sparse.SparseTensor(adj[0], adj[1], adj[2])

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


    #debug
    print("start training")
    for i in range(args.epochs):
        if args.model_name in ['Deepwalk', 'DeepwalkGaussian', 'DeepwalkCLR', 'DeepwalkSiamese']:
            loss = 0.
            tmp_loss = np.array([0.0, 0.0, 0.0, 0.0])
            training_idx = shuffle(training_idx)
            bsize = args.dw_batch_size
            for k in range(int(num_nodes/bsize)):
                idx = training_idx[k*bsize:(k+1)*bsize]
                negative_index = utils.sample_k_neighbors_from_adj(neg_sampled_adj, args.negative_num, idx, True)
                paths = np.reshape(path_corpus[idx], (-1, args.walk_length))
                X, Y = randomwalk.path_to_input(paths.tolist(), args.window_size)
                with tf.GradientTape() as tape:
                    batch_loss, _ = model((X, Y, negative_index, idx), training=True)
                    trainable_vars = model.trainable_variables
                    gradients = tape.gradient(batch_loss[0], trainable_vars)
                    model.optimizer.apply_gradients(zip(gradients, trainable_vars))
                    loss += batch_loss[0]
                    tmp_loss = tmp_loss + np.array(batch_loss[1:])
            rest = num_nodes - int(num_nodes/bsize)*bsize
            if rest:
                idx = training_idx[-rest:]
                negative_index = utils.sample_k_neighbors_from_adj(neg_sampled_adj, args.negative_num, idx, True)
                paths = np.reshape(path_corpus[idx], (-1, args.walk_length))
                X, Y = randomwalk.path_to_input(paths.tolist(), args.window_size)
                with tf.GradientTape() as tape:
                    batch_loss, _ = model((X, Y, negative_index, idx), training=True)
                    trainable_vars = model.trainable_variables
                    gradients = tape.gradient(batch_loss[0], trainable_vars)
                    model.optimizer.apply_gradients(zip(gradients, trainable_vars))
                    loss += batch_loss[0]
                    tmp_loss = tmp_loss + np.array(batch_loss[1:])
            print("%s Epoch %d: training loss %f, DW loss %f, aug DW loss %f, ins loss %f, siamese loss %f"%(args.model_name, i, loss, tmp_loss[0], tmp_loss[1], tmp_loss[2], tmp_loss[3]))
        elif args.model_name == 'GAE':

            with tf.GradientTape() as tape:
                loss, _ = model((features, adj, adj_orig), training=True)
                #print("std value %f"%norm)
                trainable_vars = model.trainable_variables
                gradients = tape.gradient(loss, trainable_vars)
                #print(tf.keras.backend.eval(gradients))
                #print(tf.keras.backend.eval(trainable_vars))
                #sys.exit(0)
                model.optimizer.apply_gradients(zip(gradients, trainable_vars))

            ap_score = 0.
            if args.lp:
            #aoc on val data    
                logits = model.run_link_logits((features, adj), val_pos_edges, val_neg_edges)
                logits = tf.keras.backend.eval(logits)
                ones = np.ones(shape=(len(val_pos_edges)), dtype=np.float32)
                zeros = np.zeros(shape=(len(val_neg_edges)), dtype=np.float32)
                labels = np.concatenate((ones, zeros), axis=0)
                #ap_score = average_precision_score(labels, logits)
                ap_score = roc_auc_score(labels, logits)

            print("Epoch %d:val link pred acc %f, trainning loss %f"%(i, ap_score, loss))
        elif args.model_name == 'GAEAdvID':
            negative_index = utils.sample_k_neighbors_from_adj(neg_sampled_adj, args.negative_num, list(range(num_nodes)), True)

            with tf.GradientTape() as tape:
                loss_list, _, _ = model((features, adj, adj_orig, negative_index, last_gradint_dir), training=True)
                last_gradint_dir = tf.keras.backend.eval(tape.gradient(loss_list[1], model.h))
                #print(last_gradint_dir)
            

            with tf.GradientTape() as tape:
                loss_list, _, _ = model((features, adj, adj_orig, negative_index, last_gradint_dir), training=True)

                loss = loss_list[0]
                trainable_vars = model.trainable_variables
                gradients = tape.gradient(loss, trainable_vars)
                #gradients, last_gradint_dir = tape.gradient([loss, loss_list[3]], [trainable_vars, model.h])
                #print(last_gradint_dir)

                model.optimizer.apply_gradients(zip(gradients, trainable_vars))

            
            
            ap_score = 0.
            if args.lp:
            #aoc on val data    
                logits = model.run_link_logits((features, adj), val_pos_edges, val_neg_edges)
                logits = tf.keras.backend.eval(logits)
                ones = np.ones(shape=(len(val_pos_edges)), dtype=np.float32)
                zeros = np.zeros(shape=(len(val_neg_edges)), dtype=np.float32)
                labels = np.concatenate((ones, zeros), axis=0)
                #ap_score = average_precision_score(labels, logits)
                ap_score = roc_auc_score(labels, logits)
   
            print("Epoch %d:val LP acc %f, Loss %f, GAE %f, AUG GAE %f, Ins %f, Hinge %f, Norm %f"%(i, ap_score, loss_list[0], loss_list[1], loss_list[2], loss_list[3], loss_list[4], loss_list[5]))

        elif 'GAE' in args.model_name:
            negative_index = utils.sample_k_neighbors_from_adj(neg_sampled_adj, args.negative_num, list(range(num_nodes)), True)

            with tf.GradientTape() as tape:
                loss_list, _, _ = model((features, adj, adj_orig, negative_index), training=True)

                loss = loss_list[0]
                trainable_vars = model.trainable_variables
                gradients = tape.gradient(loss, trainable_vars)

                model.optimizer.apply_gradients(zip(gradients, trainable_vars))

            ap_score = .0
            if args.lp:
                #run link prediction to early stop
                logits = model.run_link_logits((features, adj), val_pos_edges, val_neg_edges)
                logits = tf.keras.backend.eval(logits)
                ones = np.ones(shape=(len(val_pos_edges)), dtype=np.float32)
                zeros = np.zeros(shape=(len(val_neg_edges)), dtype=np.float32)
                labels = np.concatenate((ones, zeros), axis=0)
                #ap_score = average_precision_score(labels, logits)
                ap_score = roc_auc_score(labels, logits)

            #other ap calculate
            #_, embeddings, aug_emb = model((features, adj, adj_orig, negative_index), training=False)
            #embeddings = tf.keras.backend.eval(embeddings)
            #roc, ap = utils.get_roc_score(edges_pos=val_pos_edges, edges_neg=val_neg_edges, emb=embeddings)
            #print("roc v1 %f, v2 %f"%(ap_score, roc))

            if args.model_name == 'GAECLR':
                print("Epoch %d:val link pred acc %f, trainning loss %f, GAE loss %f, aug GAE loss %f, instance loss %f, hinge loss %f"%(i, ap_score, loss_list[0], loss_list[1], loss_list[2], loss_list[3], loss_list[4]))
            elif args.model_name == 'GAEGaussianCLR':
                #print("Epoch %d:val link pred acc %f,  trainning loss %f, GAE loss %f, aug GAE loss %f, instance loss %f, norm loss %f"%(i, ap_score, loss_list[0], loss_list[1], loss_list[2], loss_list[3], loss_list[4]))
                print("Epoch %d:val LP acc %f, Loss %f, GAE %f, AUG GAE %f, Ins %f, Hinge %f, Norm %f"%(i, ap_score, loss_list[0], loss_list[1], loss_list[2], loss_list[3], loss_list[4], loss_list[5]))
            elif args.model_name == 'GAESiameseCLR':
                print("Epoch %d:val link pred acc %f,  trainning loss %f, GAE loss %f, aug GAE loss %f, siamese loss %f"%(i, ap_score, loss_list[0], loss_list[1], loss_list[2], loss_list[3]))
        elif args.model_name in ['DNNFeatureCLR', 'DNNFeatureSiameseCLR', 'DNNFeatureGaussianCLR']:
            #negative_index = utils.sample_k_neighbors_from_adj(neg_sampled_adj, args.negative_num, list(range(num_nodes)))
            with tf.GradientTape() as tape:
                loss_list, _, _ = model((features), training=True)
                #print("std value %f"%norm)
                loss = loss_list[0]
                trainable_vars = model.trainable_variables
                gradients = tape.gradient(loss, trainable_vars)
                model.optimizer.apply_gradients(zip(gradients, trainable_vars))
                print("Epoch %d: training loss %f, instance loss %f, siamese loss %f"%(i, loss, loss_list[1], loss_list[2]))
        else:
            #add corrption in feature and adj
            corp_features_inputs = shuffle(corp_features)
            corp_features_inputs = utils.sparse_to_tuple(corp_features_inputs)
            corp_features_inputs = tf.sparse.SparseTensor(corp_features_inputs[0], corp_features_inputs[1], corp_features_inputs[2])
            corp_adj_inputs = adj

            negative_index = utils.sample_k_neighbors_from_adj(neg_sampled_adj, args.negative_num, list(range(num_nodes)), True)

            with tf.GradientTape() as tape:
                loss_list, _, _ = model((features, adj, corp_features_inputs, corp_adj_inputs, negative_index), training=True)

                loss = loss_list[0]
                trainable_vars = model.trainable_variables
                gradients = tape.gradient(loss, trainable_vars)
                model.optimizer.apply_gradients(zip(gradients, trainable_vars))

            ap_score = .0
            if args.lp:
                #run link prediction to early stop
                logits = model.run_link_logits((features, adj), val_pos_edges, val_neg_edges)
                logits = tf.keras.backend.eval(logits)
                ones = np.ones(shape=(len(val_pos_edges)), dtype=np.float32)
                zeros = np.zeros(shape=(len(val_neg_edges)), dtype=np.float32)
                labels = np.concatenate((ones, zeros), axis=0)
                #ap_score = average_precision_score(labels, logits)
                ap_score = roc_auc_score(labels, logits)

            if args.model_name == 'GraphCLR':
                print("Epoch %d: trainning loss %f, DGI loss %f, aug DGI loss %f, instance loss %f, hinge loss %f"%(i, loss_list[0], loss_list[1], loss_list[2], loss_list[3], loss_list[4]))
            elif args.model_name == 'GraphGaussianCLR':
                print("Epoch %d: Loss %f, DGI %f, AUG DGI %f, Ins %f, Hinge %f, Norm %f"%(i, loss_list[0], loss_list[1], loss_list[2], loss_list[3], loss_list[4], loss_list[5]))
            elif args.model_name == 'GraphSiameseCLR':
                print("Epoch %d: trainning loss %f, DGI loss %f, aug DGI loss %f, siamese loss %f, norm constraints loss %f"%(i, loss_list[0], loss_list[1], loss_list[2], loss_list[3], loss_list[4]))

        #if 'GAE' in args.model_name and args.lp:
        if args.lp:
            if ap_score > best_acc:
                best_acc = ap_score
                wait = 0
                best_weights = model.get_weights()
                imporve = True
            else:
                imporve = False
        else:
            if loss < best:
                best = loss
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

    if args.model_name in ['Deepwalk', 'DeepwalkCLR', 'DeepwalkSiamese', 'DeepwalkGaussian']:
        #embeddings = model.embedding
        #aug_emb = None
        embeddings, aug_emb = model.run_emb()
    elif args.model_name == 'GAEAdvID':
        _, embeddings, aug_emb = model((features, adj, adj_orig, negative_index, last_gradint_dir), training=False)
        std = model.run_std((features, adj))
        #std = tf.keras.backend(std)
        print("max std %f, min std %f, mean std %f"%(np.max(std), np.min(std), np.mean(std)))
    elif args.model_name in ['GAECLR', 'GAEGaussianCLR', 'GAESiameseCLR']:
        if args.model_name == 'GAEGaussianCLR':
            std = model.run_std((features, adj))
            #print("mean norm of std value is %f"%(np.mean(np.linalg.norm(std, axis=-1))))
            print("mean of std value is %f"%(np.mean(std)))
        _, embeddings, aug_emb = model((features, adj, adj_orig, negative_index), training=False)
        #if args.model_name == 'GAEGaussianCLR':
        #    std = model.run_std((features, adj))
        #    print("mean norm of std value is %f"%(np.mean(np.linalg.norm(std, axis=-1))))
    elif args.model_name in ['DNNFeatureCLR', 'DNNFeatureSiameseCLR', 'DNNFeatureGaussianCLR']:
        _, embeddings, aug_emb = model((features), training=False)
    elif args.model_name == 'GAE':
        _, embeddings = model((features, adj, adj_orig), training=False)
        aug_emb = embeddings
    else:
        if args.model_name == 'GraphGaussianCLR':
            std = model.run_std((features, adj))
            #print("mean norm of std value is %f"%(np.mean(np.linalg.norm(std, axis=-1))))
            print("mean of std value is %f"%(np.mean(std)))
        _, embeddings, aug_emb = model((features, adj, corp_features_inputs, corp_adj_inputs, negative_index), training=False)

    embeddings = tf.keras.backend.eval(embeddings)
    aug_emb = tf.keras.backend.eval(aug_emb)

    

    #run augmentation operator 10 times to get augmentation nodes
    #avg_aug_emb = 0.
    #aug_sample_num = 10
    #aug_emb_list = []
    #for i in range(aug_sample_num):
    #    avg_aug_emb += tf.keras.backend.eval(aug_emb)
    #    aug_emb_list.append(tf.keras.backend.eval(aug_emb))
    #avg_aug_emb = avg_aug_emb / aug_sample_num
    #return embeddings, avg_aug_emb, aug_emb_list
    return embeddings, aug_emb

def clf_function(embeddings, gdata, args):
    trainX_indices, trainY = gdata.get_node_classification_data(name='train')
    valX_indices, valY = gdata.get_node_classification_data(name='val')
    testX_indices, testY = gdata.get_node_classification_data(name='test')

    if args.sample_node_label and valX_indices:
        testX_indices = np.concatenate((testX_indices, valX_indices), axis=0)
        testY = np.concatenate((testY, valY), axis=0)

    trainX = embeddings[trainX_indices]
    valX = embeddings[valX_indices]
    testX = embeddings[testX_indices]

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


    gdata = Graphdata()
    if dataset_str in ['cora', 'citeseer', 'pubmed', 'wiki']:
        if args.sample_node_label:
            gdata.load_gcn_data(dataset_str, fix_node_test=False, node_label=True)
            gdata.random_split_train_test(node_split=True, node_train_ratio=args.train_label_ratio, node_test_ratio=args.test_label_ratio, set_label_dis=args.set_label_dis)

        else:
            gdata.load_gcn_data(dataset_str, fix_node_test=True)
    elif dataset_str in ['reddit', 'ppi']:
        gdata = InductiveData('data', dataset_str)
    else:
        print("no such dataset")
        sys.exit(0)
        pass

    if args.lp:
        gdata.random_split_train_test(edge_split=True, edge_train_ratio=0.85, edge_test_ratio=0.10)

    if args.embedding:
        #embeddings, avg_aug_emb, aug_emb_list = training_embedding(gdata, args)
        embeddings, aug_emb = training_embedding(gdata, args)
        node_num = len(gdata.get_nodes('train'))
        #print(embeddings.shape)
        #print(aug_emb.shape)
        #sys.exit(0)


        #print(avg_aug_emb)

        #debug compare norm of embedding and augmentation node embedding
        norm_e =  np.linalg.norm(embeddings, axis=-1)
        #norm_e =  np.linalg.norm(np.tile(embeddings, [args.augment_num, 1]), axis=-1)
        print("mean norm in embedding %f"%np.mean(norm_e))
        norm_ae = np.linalg.norm(aug_emb, axis=-1)
        print("mean norm in augmentation embedding %f"%np.mean(norm_ae))
        diff = norm_e - norm_ae
        print("mean difference in norm %f"%np.mean(np.abs(diff)))





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

    if args.lp:
        test_pos_edges, test_neg_edges = gdata.get_link_prediction_data(data_name='test')
        roc_score, ap_score = utils.get_roc_score(test_pos_edges, test_neg_edges, embeddings)
        print("Test roc scoreL:%.5f, average precision:%.5f"%(roc_score, ap_score))
        #return None, None, roc_score, ap_score

    if args.clf:
        _, _, mi, ma = clf_function(aug_emb, gdata, args)
        #aug_emb = np.reshape(aug_emb, newshape=(embeddings.shape[0], args.augment_num, -1))
        #avg_aug_emb = np.mean(aug_emb, axis=1)
        #_, _, mi, ma = clf_function(avg_aug_emb, gdata, args)
        #aug_emb_list = np.split(np.reshape(aug_emb, newshape=(norm_e.shape[0]*args.augment_num, -1)), args.augment_num)
        #print("avg augmentation nodes micro f1 %.5f, macro f1 %.5f"%(mi, ma))
        #_, _, mi, ma = enhance_clf_function(aug_emb_list, gdata, args)
        #print("append all augmentation nodes micro f1 %.5f, macro f1 %.5f"%(mi, ma))
        if args.enhance:
            #_, _, mi, ma = enhance_clf_function([embeddings, avg_aug_emb], gdata, args)
            _, _, mi, ma = enhance_clf_function([embeddings, aug_emb], gdata, args)
            print("avgerage enhance version micro f1 %.5f, macro f1 %.5f"%(mi, ma))
        #    _, _, mi, ma = enhance_clf_function([embeddings] + aug_emb_list, gdata, args)
        #    print("append all augmentation nodes enhance version micro f1 %.5f, macro f1 %.5f"%(mi, ma))
        return clf_function(embeddings, gdata, args)




def multi_run(args):
    micro_f1 = []
    macro_f1 = []
    k = 1
    if args.hp:
        k = 5
    for i in range(k):
        #with tf.variable_scope("main", reuse=tf.AUTO_REUSE):
        #with tf.variable_scope("main_%d"%i):
        _, _, mi, ma = main(args)
        micro_f1.append(mi)
        macro_f1.append(ma)
    return np.mean(micro_f1), np.mean(macro_f1), np.std(micro_f1), np.std(macro_f1)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', type=str, default='exam', help='数据集名称')
    parser.add_argument("--model_name", type=str, default='GraphCLR', help='训练节点表示使用的模型')
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
    parser.add_argument('--temperature', type=float, default=0.1, help='temperature parameters in instance classify')

    parser.add_argument('--aug_dgi_loss_w', type=float, default=0.0, help='Deep Graph infomax on augmentation node')
    parser.add_argument('--ins_loss_w', type=float, default=0.0, help='instance classify loss, used to train node augmentation operator')
    parser.add_argument('--l2_r', type=float, default=0., help='l2 regularization on parameters')
    parser.add_argument('--norm_loss_w', type=float, default=0., help='norm constraints weight on node embeddings and augmentation embedddings')
    parser.add_argument('--negative_num', type=int, default=10, help='number of negative samples used in instance classify loss')
    parser.add_argument('--siamese_loss_w', type=float, default=0., help='using siamese loss to classify real node and its augmentation node')
    parser.add_argument('--siamese_pos_w', type=float, default=10, help='weight of positive samples in siamese loss')
    parser.add_argument('--augment_num', type=int, default=1, help='number of augmentation node sampled per step of train per node')
    parser.add_argument('--neighbor_hop', type=int, default=2, help='hop of neighbors using to sample negative sample in instance loss')

    parser.add_argument("--number_of_walks", type=int, default=10, help='number of randomwalk sampled per node')
    parser.add_argument("--walk_length", type=int, default=10, help='length of per randomwalk path')
    parser.add_argument("--window_size", type=int, default=5, help='window size of skipgram model')
    parser.add_argument("--dw_neg_num", type=int, default=10, help='number of negative node in training skipgram model')
    parser.add_argument("--dw_batch_size", type=int, default=50, help='batch size in training deepwalk model')
    parser.add_argument("--aug_dw_loss_w", type=float, default=0.0, help='weight of skipgram with augmentation nodes')
    parser.add_argument("--aug_gae_loss_w", type=float, default=0.0, help='weight of GAE loss with augmentation nodes')
    parser.add_argument("--hinge_loss_w", type=float, default=0.0, help='weight of hinge loss(to keep node indentity)')

    parser.add_argument("--train_label_ratio", type=float, default=0.8, help='ratio of labeled nodes in training')
    parser.add_argument("--test_label_ratio", type=float, default=0.2, help='ratio of labeled nodes in test')
    parser.add_argument("--set_label_dis", dest='set_label_dis', default=False, action='store_true', help='split dataset according to label sperately')
    parser.add_argument('--sample_node_label', dest='sample_node_label', default=False, action='store_true', help='random sample train/val/test label dataset')
    parser.add_argument("--hp", dest='hp', default=False, action='store_true', help='hyperparameters seclection flag')
    parser.add_argument("--enhance", dest='enhance', default=False, action='store_true', help='using augmentation node in downstream tasks.')

    args = parser.parse_args()

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

