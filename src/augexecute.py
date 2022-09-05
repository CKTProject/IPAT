import tensorflow as tf
from tensorflow import keras
import numpy as np
import scipy.sparse as sp

import time
import os
import shutil
import sys

from models import LossAndErrorPrintingCallback, EarlyStoppingAtMinLoss
from augmodel import FowardR, FixR, HyperR, LearnR, LearnSimpleR,AdvR, GAE
from tasks import Classifier, Cluster, LinkPredictor
import utils
from utils import Graphdata
from utils import ConfigerLoader

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle
from sklearn.preprocessing import label_binarize, LabelBinarizer
from sklearn.multiclass import OneVsRestClassifier

from ogb.nodeproppred import Evaluator

#import matplotlib.pyplot as plt

import randomwalk
from randomwalk import RandomWalk

import networkx as nx
import argparse
import json

# Train on CPU (hide GPU) due to memory constraints
#os.environ['CUDA_VISIBLE_DEVICES'] = "3"
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

#gloabl var store emb attack result
attack_result = {}


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

def collect_new_index(row_index):
    cur_new_index = 0
    old_index_to_new_index = {}

    new_row_index = []

    for i in row_index:
        if i not in old_index_to_new_index:
            old_index_to_new_index[i] = cur_new_index
            cur_new_index += 1
        new_row_index.append(old_index_to_new_index[i])
    new_row_index = np.array(new_row_index)

    items = sorted(old_index_to_new_index.items(), key=lambda x:x[1])

    select_old_index = []
    for it in items:
        select_old_index.append(it[0])

    return select_old_index, old_index_to_new_index, new_row_index


def training_embedding(gdata, args, a_r=0.):

    tf.keras.backend.set_floatx('float64')
    #tf.keras.backend.set_floatx('float32')

    sperate_name = 'all'
    if args.lp:
        val_pos_edges, val_neg_edges = gdata.get_link_prediction_data(data_name='val')
        sperate_name = 'train'


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
        return ppmi.toarray()



    #trainG = gdata.get_graph(sperate_name)


    features = gdata.get_features().astype(np.float64)
    num_features = features.shape[1]
    #sta_information = gdata.get_static_information()
    #num_features = sta_information[0]

    adj = gdata.get_adj(sperate_name).astype(np.float64)
    num_nodes = adj.shape[0]
    node_num = adj.shape[0]




    #adj = utils.random_add_value_in_spmatrix(adj.copy(), a_r)


    if args.model_name == 'HyperR':
        hyper_radius = get_hyper_radius(adj, 2, num_nodes)
    if args.model_name == 'AdvR':
        hyper_radius = get_adv_hyper_radius(adj, 2, num_nodes)


    adj_pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)


    adj_orig = adj.copy()
    adj_orig = utils.sparse_to_tuple(adj_orig)
    adj_orig = tf.sparse.SparseTensor(adj_orig[0], adj_orig[1], adj_orig[2])


    #neg_sampled_adj = adj.copy()
    #for i in range(args.neighbor_hop-1):
   # 	neg_sampled_adj = neg_sampled_adj.dot(adj)

    #if args.dataset == 'ogbn-arxiv':
    #    features = utils.preprocess_features(features)
    #else:
    #    #features = utils.tuple_to_sparse(utils.preprocess_features(features))
    #    #features = utils.tuple_to_sparse(features)
    features = features
    adj = utils.tuple_to_sparse(utils.preprocess_graph(adj))

    corp_features = features.copy()
    corp_adj = adj.copy()


    if args.dataset == 'ogbn-arxiv':
        rows, cols = gdata.get_ogb_edges()
        ogb_batch_size = 10000
        ogb_max_size = len(rows)
        ogb_max_iter = int(len(rows) / 10000) + 1
    else:
        features = utils.sparse_to_tuple(features)
    adj = utils.sparse_to_tuple(adj)



    with open(args.network_config, 'r') as f:
        network_config = json.load(f)

    gnn_units_list = network_config['gnn_units_list']
    dnn_units_list = network_config['dnn_units_list']

    #act_func = tf.nn.relu
    act_func = tf.nn.elu

    sparse_flag = True 
    if args.dataset == 'ogbn-arxiv':
        sparse_flag = False

    if args.model_name == 'FowardR':
        model = FowardR(input_dim=num_features, gnn_units_list=gnn_units_list, sparse_inputs=sparse_flag, dropout=args.dropout, bias=False, l2_r=args.l2_r,
                 negative_num=args.negative_num, aug_gae_loss_w=args.aug_gae_loss_w, ins_loss_w=args.ins_loss_w, learning_rate=args.learning_rate, temperature=args.temperature, augment_num=args.augment_num,
                 norm=norm, adj_pos_weight=adj_pos_weight, norm_loss_w=args.norm_loss_w, hinge_loss_w=args.hinge_loss_w, act=act_func,
            )
    elif args.model_name == 'FixR':
        model = FixR(input_dim=num_features, gnn_units_list=gnn_units_list, sparse_inputs=sparse_flag, dropout=args.dropout, bias=False, l2_r=args.l2_r,
                 negative_num=args.negative_num, aug_gae_loss_w=args.aug_gae_loss_w, ins_loss_w=args.ins_loss_w, learning_rate=args.learning_rate, temperature=args.temperature, augment_num=args.augment_num,
                 norm=norm, adj_pos_weight=adj_pos_weight, norm_loss_w=args.norm_loss_w, hinge_loss_w=args.hinge_loss_w, act=act_func,
            )
                 
    elif args.model_name == 'HyperR':
        model = HyperR(input_dim=num_features, gnn_units_list=gnn_units_list, sparse_inputs=sparse_flag, dropout=args.dropout, bias=False, l2_r=args.l2_r,
                 negative_num=args.negative_num, aug_gae_loss_w=args.aug_gae_loss_w, ins_loss_w=args.ins_loss_w, learning_rate=args.learning_rate, temperature=args.temperature, augment_num=args.augment_num,
                 norm=norm, adj_pos_weight=adj_pos_weight, norm_loss_w=args.norm_loss_w, hinge_loss_w=args.hinge_loss_w, act=act_func,
            )
                 
    elif args.model_name == 'LearnR':
        model = LearnR(input_dim=num_features, gnn_units_list=gnn_units_list, sparse_inputs=sparse_flag, dropout=args.dropout, bias=False, l2_r=args.l2_r,
                 negative_num=args.negative_num, aug_gae_loss_w=args.aug_gae_loss_w, ins_loss_w=args.ins_loss_w, learning_rate=args.learning_rate, temperature=args.temperature, augment_num=args.augment_num,
                 norm=norm, adj_pos_weight=adj_pos_weight, norm_loss_w=args.norm_loss_w, hinge_loss_w=args.hinge_loss_w, act=act_func,
            )
                 
    elif args.model_name == 'LearnSimpleR':
        model = LearnSimpleR(input_dim=num_features, gnn_units_list=gnn_units_list, sparse_inputs=sparse_flag, dropout=args.dropout, bias=False, l2_r=args.l2_r,
                 negative_num=args.negative_num, aug_gae_loss_w=args.aug_gae_loss_w, ins_loss_w=args.ins_loss_w, learning_rate=args.learning_rate, temperature=args.temperature, augment_num=args.augment_num,
                 norm=norm, adj_pos_weight=adj_pos_weight, norm_loss_w=args.norm_loss_w, hinge_loss_w=args.hinge_loss_w, act=act_func,
            )
                 
    elif args.model_name == 'AdvR':
        model = AdvR(input_dim=num_features, gnn_units_list=gnn_units_list, sparse_inputs=sparse_flag, dropout=args.dropout, bias=False, l2_r=args.l2_r,
                 negative_num=args.negative_num, aug_gae_loss_w=args.aug_gae_loss_w, ins_loss_w=args.ins_loss_w, learning_rate=args.learning_rate, temperature=args.temperature, augment_num=args.augment_num,
                 norm=norm, adj_pos_weight=adj_pos_weight, norm_loss_w=args.norm_loss_w, hinge_loss_w=args.hinge_loss_w, act=act_func,
            )
                 
    elif args.model_name == 'GAE':
        model = GAE(input_dim=num_features, gnn_units_list=gnn_units_list, sparse_inputs=sparse_flag, dropout=args.dropout, bias=False, l2_r=args.l2_r,
                learning_rate=args.learning_rate, adj_pos_weight=adj_pos_weight, norm=norm, act=act_func,
            )
    else:
        assert False, "No such model"

    model.compile()
    
    if args.dataset == 'ogbn-arxiv':
        pass
    else:
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

    last_gradint_dir = None

    #debug
    print("start training")
    for i in range(args.epochs):
        if args.model_name == 'GAE':
            if args.dataset == 'ogbn-arxiv':
                loss = 0
                for j in range(ogb_max_iter):
                    with tf.GradientTape() as tape:
                        l = j * ogb_batch_size
                        r = min((j+1) * ogb_batch_size, ogb_max_size)
                        negs = np.random.randint(low=0, high=num_nodes, size=(r-l, args.dw_neg_num))
                        gae_loss, _ = model((features, adj, rows[l:r], cols[l:r], negs), training=True)
                        trainable_vars = model.trainable_variables
                        gradients = tape.gradient(gae_loss, trainable_vars)
                        model.optimizer.apply_gradients(zip(gradients, trainable_vars))
                        print("Epochs %d Iter %d: Loss %.5f"%(i, j, gae_loss))
                        loss += gae_loss
                        if r == ogb_max_size:
                            break
            else:
                with tf.GradientTape() as tape:
                    gae_loss, _= model((features, adj, adj_orig), training=True)
                    trainable_vars = model.trainable_variables
                    gradients = tape.gradient(gae_loss, trainable_vars)
                    model.optimizer.apply_gradients(zip(gradients, trainable_vars))
                    loss = gae_loss
        else:
            #negative_index = utils.sample_k_neighbors_from_adj(neg_sampled_adj, args.negative_num, list(range(num_nodes)), True)
            epoch_loss_list = [0, 0, 0, 0, 0, 0]
            if args.dataset == 'ogbn-arxiv':
                
                for j in range(ogb_max_iter):
                    l = j * ogb_batch_size
                    r = min((j+1) * ogb_batch_size, ogb_max_size)
                    negs = np.random.randint(low=0, high=num_nodes, size=(r-l, args.dw_neg_num))                    
                    select_index, id2id, new_rows = collect_new_index(rows[l:r].tolist())
                    id_negs = np.random.randint(low=0, high=num_nodes, size=(len(select_index), args.negative_num))

                    if last_gradint_dir is None:
                        last_gradint_dir = np.random.normal(size=(len(select_index), gnn_units_list[-1]))

                    with tf.GradientTape() as tape:     
                        if args.model_name == 'HyperR' or args.model_name == 'AdvR':
                            loss_list, _, _ = model((features, adj, adj_orig, last_gradint_dir, hyper_radius), training=True)
                        else:
                            loss_list, _, _ = model((features, adj, last_gradint_dir, select_index, id2id, new_rows, cols[l:r], negs, id_negs), training=True)

                        last_gradint_dir = tf.convert_to_tensor(tape.gradient(loss_list[1],model.gae_h))
                        last_gradint_dir = tf.keras.backend.eval(last_gradint_dir)

                    with tf.GradientTape() as tape:
                        if args.model_name == 'HyperR' or args.model_name == 'AdvR':
                            loss_list, _, _ = model((features, adj, adj_orig, last_gradint_dir, hyper_radius), training=True)
                        else:
                            loss_list, _, _ = model((features, adj, last_gradint_dir, select_index, id2id, new_rows, cols[l:r], negs, id_negs), training=True)
                        loss = loss_list[0]
                        trainable_vars = model.trainable_variables
                        gradients = tape.gradient(loss, trainable_vars)
                        model.optimizer.apply_gradients(zip(gradients, trainable_vars))
                        print("Epochs %d Iter %d: Loss %.5f GAE %f, AUG GAE %f, Ins %f, Norm %f"%(i, j, loss_list[0], loss_list[1], loss_list[2], loss_list[3], loss_list[5]))
                        for k, t_loss in enumerate(loss_list):
                            epoch_loss_list[k] += t_loss
                        if r == ogb_max_size:
                            break
            else:
                with tf.GradientTape() as tape:
                    if args.model_name == 'HyperR' or args.model_name == 'AdvR':
                        loss_list, _, _ = model((features, adj, adj_orig, last_gradint_dir, hyper_radius), training=True)
                    else:
                        loss_list, _, _ = model((features, adj, adj_orig, last_gradint_dir), training=True)
                    last_gradint_dir = tf.keras.backend.eval(tape.gradient(loss_list[1], model.h))
                    #print(last_gradint_dir)

                with tf.GradientTape() as tape:
                    if args.model_name == 'HyperR' or args.model_name == 'AdvR':
                        loss_list, _, _ = model((features, adj, adj_orig, last_gradint_dir, hyper_radius), training=True)
                    else:
                        loss_list, _, _ = model((features, adj, adj_orig, last_gradint_dir), training=True)

                    loss = loss_list[0]
                    trainable_vars = model.trainable_variables
                    gradients = tape.gradient(loss, trainable_vars)
                    #gradients, last_gradint_dir = tape.gradient([loss, loss_list[3]], [trainable_vars, model.h])
                    #print(last_gradint_dir)

                    model.optimizer.apply_gradients(zip(gradients, trainable_vars))
                    epoch_loss_list = loss_list

                    

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

            #print("Epoch %d:val link pred acc %f,  trainning loss %f, GAE loss %f, aug GAE loss %f, instance loss %f, norm loss %f"%(i, ap_score, loss_list[0], loss_list[1], loss_list[2], loss_list[3], loss_list[4]))
        if args.model_name == 'AdvR':
            print("Epoch %d:val LP acc %f, Loss %f, GAE %f, AUG GAE %f"%(i, ap_score, epoch_loss_list[0], epoch_loss_list[1], epoch_loss_list[2]))
        elif args.model_name == 'GAE':
            print("Epoch %d:val LP acc %f, GAE Loss %f"%(i, ap_score, loss))
        else:
            print("Epoch %d:val LP acc %f, Loss %f, GAE %f, AUG GAE %f, Ins %f, Hinge %f, Norm %f"%(i, ap_score, epoch_loss_list[0], epoch_loss_list[1], epoch_loss_list[2], epoch_loss_list[3], epoch_loss_list[4], epoch_loss_list[5]))
        
        verbose_step = 1
        if (i+1) % verbose_step == 0:
            temp_emb = model.run_embedding((features, adj))
            temp_emb = tf.keras.backend.eval(temp_emb)
            clf_function(temp_emb, gdata, args)


        
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

    if args.model_name in ['FowardR', 'FixR', 'HyperR', 'LearnR', 'LearnSimpleR', 'AdvR', 'GAE']:
        if args.model_name == 'GAEGaussianCLR':
            std = model.run_std((features, adj))
            #print("mean norm of std value is %f"%(np.mean(np.linalg.norm(std, axis=-1))))
            print("mean of std value is %f"%(np.mean(std)))
        embeddings = model.run_embedding((features, adj))


    embeddings = tf.keras.backend.eval(embeddings)
    #aug_emb = tf.keras.backend.eval(aug_emb)
    aug_emb = None

    #save radius of node embedding
    if args.model_name in ['FowardR', 'FixR', 'HyperR', 'LearnR', 'LearnSimpleR']:
        std = model.run_std((features, adj))
        std = tf.keras.backend.eval(std)
        outfile_name = 'temp/%s-%s.csv'%(args.model_name, args.dataset)
        std = np.reshape(std, (-1, 1))
        np.savetxt(outfile_name, std, delimiter=',')

    
    emb_dimension = embeddings.shape[1]
    dataset_str = args.dataset
    model_str = args.model_name
    header = "%d %d"%(node_num, emb_dimension)
    if not os.path.exists('emb/idvisual/%s'%dataset_str):
        os.makedirs('emb/idvisual/%s'%dataset_str)
    femb = 'emb/idvisual/%s/%s-%d.txt'%(dataset_str, model_str, emb_dimension)

    if True or args.model_name == 'GAE':
        return embeddings, None
    
    #Not Used
    #save embedding and augmentation embedding
    if args.model_name == 'HyperR' or args.model_name == 'AdvR':
        loss_list, temp_emb, temp_aug_emb = model((features, adj, adj_orig, last_gradint_dir, hyper_radius))
    else:
        loss_list, temp_emb, temp_aug_emb = model((features, adj, adj_orig, last_gradint_dir))

    
    if temp_emb is None:
        print("embedding load error")
    else:
        utils.save_embedding(femb, temp_emb, delimiter=' ', header=header)
    femb = 'emb/idvisual/%s/%s-%d-aug.txt'%(dataset_str, model_str, emb_dimension)
    if temp_aug_emb is None:
        print("embedding load error")
    else:
        utils.save_embedding(femb, temp_aug_emb, delimiter=' ', header=header)

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

    tf_linear_flag = True 
    svm_flag = False

    if tf_linear_flag:
        ##using tensorflow linear model
        model = keras.Sequential(
        [
            keras.layers.Dropout(0.3),
            keras.layers.Dense(128, activation='elu', name="layer1"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(128, activation='elu', name="layer2"),
            #keras.layers.Dense(16, activation="relu", name="layer1"),
            #keras.layers.BatchNormalization(),
            #keras.layers.Dropout(0.0),
            #keras.layers.Dense(64, activation=None, name="layer2"),
            #keras.layers.Dense(16, activation="relu", name="layer1"),
            #keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(class_num, name="layer3"),
        ]
        )

        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

        model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['categorical_accuracy', 'accuracy'],   
        )

        #model.fit(trainX, trainY, epochs=1000, verbose=10, validation_data=(valX, valY),
        #         callbacks=[LossAndErrorPrintingCallback(), EarlyStoppingAtMinLoss(0, 0)])
        epochs = 20
        #history = model.fit(trainX, trainY, epochs=epochs, validation_data=(valX, valY), callbacks=[callback])
        history = model.fit(trainX, trainY, epochs=epochs, validation_data=(testX, testY), callbacks=[callback])

        #acc = history.history['accuracy']
        #val_acc = history.history['val_accuracy']
        #loss = history.history['loss']
        #val_loss = history.history['val_loss']
        #epochs_range = range(epochs)
        #plt.figure(figsize=(8, 8))
        #plt.subplot(1, 2, 1)
        #plt.plot(epochs_range, acc, label='Training Accuracy')
        #plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        #plt.legend(loc='lower right')
        #plt.title('Training and Validation Accuracy')
        #plt.subplot(1, 2, 2)
        #plt.plot(epochs_range, loss, label='Training Loss')
        #plt.plot(epochs_range, val_loss, label='Validation Loss')
        #plt.legend(loc='upper right')
        #plt.title('Training and Validation Loss')
        #plt.show()

        val_result = {}
        result = {}
        val_res = model.evaluate(valX, valY)
        val_acc = val_res[1]
        test_res = model.evaluate(testX, testY)
        test_acc = test_res[1]
        val_result['micro'] = val_acc
        val_result['macro'] = 0.0
        result['micro'] = test_acc
        result['macro'] = 0.0
        print("Val acc %.5f, Test acc %.5f"%(val_acc, test_acc))


        return val_result['micro'], val_result['macro'], result['micro'], result['macro']

    if svm_flag:
        best_val_micro = 0.
        best_test_micro = 0.
        best_val_macro = 0.
        best_test_macro = 0.

    ##for c in [1e-3, 1e-2, 1e-1, 1.0, 10., 100., 1000.]:
        for c in [1.0]:
            clf = Classifier('SVM', class_num, c)
        #    #clf = Classifier('LR', class_num)
            clf.train(trainX, trainY)
            if not valX_indices is None:
                val_result = clf.evaluate(valX, valY)
                print("C:%f, val result of %s-%d: micro F1 %.5f, macro F1 %.5f"%(c, args.model_name, args.dimension, val_result['micro'], val_result['macro']))
            else:
                val_result = {}
                val_result['micro'], val_result['macro'] = 0.0, 0.0
            result = clf.evaluate(testX, testY)
            print("C:%f, test result of %s-%d: micro F1 %.5f, macro F1 %.5f"%(c, args.model_name, args.dimension, result['micro'], result['macro']))
            if best_val_micro < val_result['micro']:
                best_val_micro = val_result['micro']
                best_test_micro = result['micro']
                best_val_macro = val_result['macro']
                best_test_macro = result['macro']

            #debug save error index
        #    emb_dimension = embeddings.shape[1]
        #    dataset_str = args.dataset
        #    model_str = args.model_name
        #    if not os.path.exists('emb/idvisual/%s'%dataset_str):
        #        os.makedirs('emb/idvisual/%s'%dataset_str)
        #    femb = 'emb/idvisual/%s/%s-%d-error.txt'%(dataset_str, model_str, emb_dimension)
        #    error_indx = clf.collect_error_nodes(testX, testY, testX_indices)
        #    if error_indx is None:
        #        print("error_indx load error")
        #    else:
        #        utils.save_embedding(femb, error_indx, delimiter=' ')
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

    clf = Classifier('SVM', class_num)
    #clf = Classifier('LR', class_num)
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
    if dataset_str in ['cora', 'citeseer', 'pubmed', 'wiki','ogbn-arxiv']:
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

    if args.str_attack:
        attack_ratio = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        str_attack_result = {}
        for a_r in attack_ratio:
            if a_r not in str_attack_result:
                str_attack_result[a_r] = {}
                str_attack_result[a_r]["val_micro"]  = []
                str_attack_result[a_r]["val_macro"]  = []
                str_attack_result[a_r]["test_micro"]  = []
                str_attack_result[a_r]["test_macro"]  = []
        for a_r in attack_ratio:
            for k in range(3):
                embeddings, _ = training_embedding(gdata, args, a_r)
                best_val_micro, best_val_macro, best_test_micro, best_test_macro = clf_function(embeddings, gdata, args)
                print("Acctack raio %.5f: val_micro %.4f, val_macro %.4f, test_micro %.4f, test_macro %.4f"%(a_r, best_val_micro, best_val_macro, best_test_micro, best_test_macro))
                str_attack_result[a_r]["val_micro"].append(best_val_micro)
                str_attack_result[a_r]["val_macro"].append(best_val_macro)
                str_attack_result[a_r]["test_micro"].append(best_test_micro)
                str_attack_result[a_r]["test_macro"].append(best_test_macro)
        for a_r in attack_ratio:
            print("Acctack raio %.5f: mean val_micro %.4f, mean val_macro %.4f, mean test_micro %.4f, mean test_macro %.4f"%(a_r, np.mean(str_attack_result[a_r]["val_micro"]), np.mean(str_attack_result[a_r]["val_macro"]), np.mean(str_attack_result[a_r]["test_micro"]), np.mean(str_attack_result[a_r]["test_macro"]) ))


    if args.embedding:
        #embeddings, avg_aug_emb, aug_emb_list = training_embedding(gdata, args)
        embeddings, aug_emb = training_embedding(gdata, args)
        node_num = embeddings.shape[0]
        #print(embeddings.shape)
        #print(aug_emb.shape)
        #sys.exit(0)


        #print(avg_aug_emb)

        #debug compare norm of embedding and augmentation node embedding
        #norm_e =  np.linalg.norm(embeddings, axis=-1)
        ##norm_e =  np.linalg.norm(np.tile(embeddings, [args.augment_num, 1]), axis=-1)
        #print("mean norm in embedding %f"%np.mean(norm_e))
        #norm_ae = np.linalg.norm(aug_emb, axis=-1)
        #print("mean norm in augmentation embedding %f"%np.mean(norm_ae))
        #diff = norm_e - norm_ae
        #print("mean difference in norm %f"%np.mean(np.abs(diff)))





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

    if args.emb_attack:
        print("---------********* Start To Process Embedding Attack *********-----------")
        global attack_result
        attack_radius = [0.001, 0.01, 0.1, 1.0, 5.0, 10.]

        trainX_indices, trainY = gdata.get_node_classification_data(name='train')
        valX_indices, valY = gdata.get_node_classification_data(name='val')
        testX_indices, testY = gdata.get_node_classification_data(name='test')

        trainX = embeddings[trainX_indices]
        valX = embeddings[valX_indices]
        testX = embeddings[testX_indices]

        val_shape = valX.shape
        test_shape = testX.shape

        for a_r in attack_radius:
            if a_r not in attack_result:
                attack_result[a_r] = {}
                attack_result[a_r]["val_micro"]  = []
                attack_result[a_r]["val_macro"]  = []
                attack_result[a_r]["test_micro"]  = []
                attack_result[a_r]["test_macro"]  = []

            attack_emb = embeddings.copy()
            noise = np.random.normal(size=val_shape)
            #noise = sklearn.preprocessing.normalize(noise)
            noise = noise * a_r
            attack_emb[valX_indices, :] = valX + noise
            noise = np.random.normal(size=test_shape)
            #noise = sklearn.preprocessing.normalize(noise)
            noise = noise * a_r
            attack_emb[testX_indices, :] = testX + noise
            print("#######Start To Add Noise Wiht Radius %f ########"%a_r)
            best_val_micro, best_val_macro, best_test_micro, best_test_macro = clf_function(attack_emb, gdata, args)
            attack_result[a_r]['val_micro'].append(best_val_micro)
            attack_result[a_r]['val_macro'].append(best_val_macro)
            attack_result[a_r]['test_micro'].append(best_test_micro)
            attack_result[a_r]['test_macro'].append(best_test_macro)
        print("---------*********    Process Embedding Attack End   *********-----------")

    if args.clf:
        #_, _, mi, ma = clf_function(aug_emb, gdata, args)
        #aug_emb = np.reshape(aug_emb, newshape=(embeddings.shape[0], args.augment_num, -1))
        #avg_aug_emb = np.mean(aug_emb, axis=1)
        #_, _, mi, ma = clf_function(avg_aug_emb, gdata, args)
        #aug_emb_list = np.split(np.reshape(aug_emb, newshape=(norm_e.shape[0]*args.augment_num, -1)), args.augment_num)
        #print("avg augmentation nodes micro f1 %.5f, macro f1 %.5f"%(mi, ma))
        #_, _, mi, ma = enhance_clf_function(aug_emb_list, gdata, args)
        #print("append all augmentation nodes micro f1 %.5f, macro f1 %.5f"%(mi, ma))
        #if args.enhance:
            #_, _, mi, ma = enhance_clf_function([embeddings, avg_aug_emb], gdata, args)
        #    _, _, mi, ma = enhance_clf_function([embeddings, aug_emb], gdata, args)
        #    print("avgerage enhance version micro f1 %.5f, macro f1 %.5f"%(mi, ma))
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
    #print("-----------****** Show Attack Mean Result *****----------")
    #global attack_result
    #for attack_radius, result_dict in attack_result.items():
    #    for key_name, result_list in result_dict.items():
    #        res_mean = np.mean(result_list)
    #        res_std = np.std(result_list)
    #        print("Attack Radius %f: Mean of %s %.5f, %.5f"%(attack_radius, key_name, res_mean, res_std))
    #print("-----------****** End Attack  Mean Result *****----------")
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

    parser.add_argument("--gpuid", type=int, default=-1, help='index of gpu')

    parser.add_argument("--emb_attack", dest='emb_attack', default=False, action='store_true', help='attack node embedding in hidden space.')
    parser.add_argument("--str_attack", dest='str_attack', default=False, action='store_true', help='attack node embedding in input space.')

    args = parser.parse_args()

    if args.gpuid == -1:
        # Train on CPU (hide GPU) due to memory constraints
        print("Using CPU Only!!!", file=sys.stderr)
        os.environ['CUDA_VISIBLE_DEVICES'] = ""
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = "%d"%args.gpuid
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        #print(gpus)
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        #    print(gpu, file=sys.stderr)

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

