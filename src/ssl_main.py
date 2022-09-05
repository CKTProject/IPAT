import tensorflow as tf
import numpy as np
import scipy.sparse as sp

import time
import os
import shutil
import sys

import math

from models import GraphCLR
from tasks import Classifier, Cluster, LinkPredictor, Visualizer
import utils
from utils import Graphdata, NeiSampler, HierRankData, InductiveData, ClusterMiniBatchGenerator
from utils import ConfigerLoader
from initials import glorot, zeros

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.cluster import SpectralClustering
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import label_binarize, LabelBinarizer


from scipy.sparse.linalg import eigs
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF

from sklearn.multiclass import OneVsRestClassifier

import networkx as nx
import metis



# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

#debug
DEBUG_FLAG=False

#fix random seed when debug
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model', 'Gaussian', 'Model string.')#Gaussian:Gaussian distribution embedding;DC2V:DisentanglingContext2Vec;HiRank2Vec:HierarchicalRank2Vec
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')
flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 3000, 'Number of epochs to train.')
flags.DEFINE_integer('early_stop', 200, 'Number of early stop iteration')
flags.DEFINE_float('dropout', 0.0, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('in_drop', 0.0, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('coef_drop', 0.0, 'Dropout rate (1 - keep probability).')
flags.DEFINE_integer('batch_size', 1000, 'batch size of each iteration')

flags.DEFINE_integer("reverse_type", 1, 'axis of attention coef softmax')

#Masked ANAE setting
flags.DEFINE_float("masked_ratio", 0.1, 'ratio of node masked in training')
flags.DEFINE_boolean("mask_padding_flag", False, 'flag of using mask tags in mask strategy')


flags.DEFINE_string("configer_path", 'gcn_ae_1', 'filename of model architecture')


flags.DEFINE_float('l2_weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('s_weight_decay', 1., 'Weight for structural loss on embedding matrix.')
flags.DEFINE_float('c_weight_decay', 1., 'Weight for content reconstruction loss on embedding matrix.')
flags.DEFINE_float("m_weight_decay", 1., 'Weight for masked content reconstruction')

flags.DEFINE_string('re_con_type', 'clf', 'loss type of content reconstruction loss: clf(weighted cross entropy) or reg(mean square loss)')
#flags.DEFINE_string('multi_loss_type', 'weight_sum', 'multi tasks loss aggregation type: weighted sum(weight_sum) or learn from parameters(gaussian)')


flags.DEFINE_integer("n_cluster", 3, 'cluster of differ neighbors')


flags.DEFINE_string('logdir', 'log/', 'dir for save logging information')
flags.DEFINE_integer('log', 0, 'flag control whether to log')
flags.DEFINE_integer("hidden1", 64, 'dimension of hidden representation')

flags.DEFINE_integer('nb_size', 10, 'number of neighbors sampled for each node')
flags.DEFINE_boolean('include_self', False, 'if neighbors include self')

flags.DEFINE_boolean('embedding', False, 'retrain node representation')
flags.DEFINE_boolean('clf', False, 'use node classification as downstream tasks')
flags.DEFINE_boolean('lp', False, 'use link prediction as downstream tasks')
flags.DEFINE_boolean('cluster', False, 'use node classification as downstream tasks')
flags.DEFINE_boolean('visual', False, 'use node classification as downstream tasks')
flags.DEFINE_integer('verbose', 5, 'show clf result every K epochs')
flags.DEFINE_boolean('sample_node_label', False, 'random sample train/val/test label dataset')
flags.DEFINE_float("train_label_ratio", 0.8, 'ratio of labeled nodes in training')
flags.DEFINE_float("test_label_ratio", 0.2, 'ratio of labeled nodes in test')
flags.DEFINE_boolean("set_label_dis", False, 'split dataset according to label sperately')


#GraphBert settting
flags.DEFINE_string("att_key_type", 'identity', 'type attention stragtegy in GraphBert:sum, mean, sig, identity')

flags.DEFINE_integer("community_num", 7, 'number of community')

flags.DEFINE_boolean("kmeans_init", False, 'flag of using kmeans init')

#DGI model flags
flags.DEFINE_boolean("corp_flag_x", False, 'corrup feature matrix')
flags.DEFINE_boolean("corp_flag_adj", False, 'corrup adj matrix')
flags.DEFINE_integer("DGI_n_cluster", 50, 'number of pre-clusters')
flags.DEFINE_float("global_weight", 0.5, 'weight of global MI')
flags.DEFINE_float("local_weight", 1.0, 'weight of local MI')
flags.DEFINE_string("DGI_loss_type", 'JSD', 'objective function of DGI: KL, JSD, NCE')
flags.DEFINE_integer("neg_size", 5, 'num of negative nodes used in nce loss')

#PCDGI setting
flags.DEFINE_float("gb_temp", 1e-20, "hyperparameter in gumbel_softmax,non-negative scalar")
flags.DEFINE_float("pred_cluster_weight", 1.0, "regularize predict community close to pre-train community result")

#CLusterDGI setting
flags.DEFINE_integer("clusters_per_batch", 3, 'clusters used in each batch')
flags.DEFINE_boolean("minibatch_flag", False, 'flag of whether using batch version of clusterDGI')

flags.DEFINE_boolean("DGI_kmeans", False, 'whether using kmeans during the training')
flags.DEFINE_integer("DGI_kmeans_step", 10, 'kmeans after x epochs of var updates')

#NewClusterDGI setting
flags.DEFINE_boolean("debug_type1", False, 'DGI version of loss')
flags.DEFINE_boolean("debug_type2", False, 'reverse version of DGI loss')
flags.DEFINE_boolean("debug_type3", False, 'loss of corp cluster labels')

#downstream setting
flags.DEFINE_boolean("clf_minibatch", False, 'flag of mini-batched node classification')

#personal pagerank DGI setting
flags.DEFINE_float("ppr_alpha", 0.85, 'Damping factor of pagerank')
flags.DEFINE_boolean("ppr_sparse", False, 'flag of sparse pagerank value')
flags.DEFINE_integer("ppr_topK", 10, 'perserve top k value of personal pagerank values')


_CLUSTERING_FN = {
    'label_prop': label_prop.label_propagation_communities,
    'modularity': modularity.greedy_modularity_communities,
    'connected_components': components.connected_components
}

flags.DEFINE_enum(
    'local_clustering_fn', 'label_prop', _CLUSTERING_FN.keys(),
    'The method used for clustering the egonets of the graph. The options are '
    '"label_prop", "modularity" or "connected_components".')
flags.DEFINE_enum(
    'global_clustering_fn', 'label_prop', _CLUSTERING_FN.keys(),
    'The method used for clustering the egonets of the graph. The options are '
    '"label_prop", "modularity" or "connected_components".')

flags.DEFINE_integer('min_component_size', 10,
                     'Minimum size for an overlapping cluster to be output.')


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


def show_hyper_parameters():
    print('Dataset: ' + FLAGS.dataset)
    print("Model: " + FLAGS.model)
    print('----- Opt. hyperparams -----')
    print('learning rate: ' + str(FLAGS.learning_rate))
    print('l2_coef: ' + str(FLAGS.l2_weight_decay))
    print('struct_coef: ' + str(FLAGS.s_weight_decay))
    print('content_coef: ' + str(FLAGS.c_weight_decay))
    print('dropout: ' + str(FLAGS.dropout))
    print('in_dropout: ' + str(FLAGS.in_drop))
    print('coef_dropout: ' + str(FLAGS.coef_drop))
    print('features: ' + str(FLAGS.features))
    print('epochs: ' + str(FLAGS.epochs))
    print('early_stop: ' + str(FLAGS.early_stop))
    print('batch_size: ' + str(FLAGS.batch_size))
    print("content reconstruction func: " + str(FLAGS.re_con_type))
    print("reverse type, axis of normalizatioin: " + str(FLAGS.reverse_type))
    print("ratio of nodes masked in training: " + str(FLAGS.masked_ratio))
    print("----- DGI. hyperparams -------")
    print("corrup features: " + str(FLAGS.corp_flag_x))
    print("corrup adj: " + str(FLAGS.corp_flag_adj))
    print("num of clusters: " + str(FLAGS.DGI_n_cluster))
    print("weight of global MI: " + str(FLAGS.global_weight))
    print("num of negative samples used in Nce loss: " + str(FLAGS.neg_size))
    print("----- MiniBatch DGI. hyperparams -------")
    print("MiniBatch Model FLAG: " + str(FLAGS.minibatch_flag))
    print("Num of clusters used in each batch: " + str(FLAGS.clusters_per_batch))
    print("Loss type of DGI: " + str(FLAGS.DGI_loss_type))
    #print("triple_sample_num: "+str(FLAGS.triple_sample_num))
    print('----- Archi. hyperparams -----')
    print("the model architecture file: " + str(FLAGS.configer_path))




def init_hyper_parameters(gdata):
    config = {}
    #set max degree manually or exactly calculate from input graph
    #max_degree = gdata.get_node_degree().max()
    config['learning_rate'] = FLAGS.learning_rate
    config['neg_num'] = FLAGS.neg_num
    config['l2_weight_decay'] = FLAGS.l2_weight_decay

    return config


def get_neighbors(node, gdata, max_degree):
    neighbors = list(gdata.get_neighbors(node))
    max_node_num = gdata.g.number_of_nodes()
    neighbors_num = len(neighbors)

    if len(neighbors) < max_degree:
        padding = [max_node_num]*(max_degree-len(neighbors))
        neighbors = neighbors + padding
    else:
        neighbors = np.random.permutation(neighbors)
        neighbors = neighbors[:max_degree]

    return neighbors, neighbors_num

def build_batch_context(batch_nodes, max_degree, gdata):
    batch_context = []
    batch_context_size = []
    for node in batch_nodes:
        neighbors, neighbors_num = get_neighbors(node, gdata, max_degree)
        batch_context.append(neighbors)
        batch_context_size.append(neighbors_num)
    batch_context = np.array(batch_context)
    batch_context_size = np.array(batch_context_size).reshape((-1,1))
    return batch_context, batch_context_size

def build_batch_neg_context(batch_nodes, node_list, neg_num, max_degree, gdata):
    batch_size = len(batch_nodes)
    batch_neg_context = []
    batch_neg_context_size = []
    for i in range(neg_num):
        node_list = np.random.permutation(node_list)
        nodes = node_list[:batch_size]
        patch_context, patch_context_size = build_batch_context(nodes, max_degree, gdata)
        patch_context = np.reshape(patch_context, newshape=(batch_size, 1, max_degree))
        patch_context_size = np.reshape(patch_context_size, newshape=(batch_size, 1, 1))
        batch_neg_context.append(patch_context)
        batch_neg_context_size.append(patch_context_size)
    batch_neg_context = np.concatenate(batch_neg_context, axis=1)
    batch_neg_context_size = np.concatenate(batch_neg_context_size, axis=1)
    return batch_neg_context, batch_neg_context_size

def build_feed_dict(placeholders, features, context, neg_context,
                    context_size, neg_context_size, dropout):
    feed_dict = {}
    feed_dict.update({placeholders['features']:features})
    feed_dict.update({placeholders['context']:context})
    feed_dict.update({placeholders['neg_context']:neg_context})
    feed_dict.update({placeholders['context_size']:context_size})
    feed_dict.update({placeholders['neg_context_size']:neg_context_size})
    feed_dict.update({placeholders['dropout']:dropout})
    return feed_dict

def train_embedding_graphAE(gdata):
    sperate_name = 'all'
    if FLAGS.lp:
        val_pos_edges, val_neg_edges = gdata.get_link_prediction_data(data_name='val')
        sperate_name = 'train'



    trainG = gdata.get_graph(sperate_name)
#build content reconstruction weight
    features = gdata.get_features()

    f_pos_weight = float(features.shape[0] * features.shape[0] - features.sum()) / features.sum()
    f_norm = features.shape[0] * features.shape[0] / float((features.shape[0] * features.shape[0] - features.sum()) * 2)

#build structure inputs
    adj = gdata.get_adj(sperate_name)
    sta_information = gdata.get_static_information()
    num_features = sta_information[0]
    features_nonzero = sta_information[1]
    num_nodes = adj.shape[0]
    #features = utils.sparse_to_tuple(features)



    gat_adj = adj+sp.identity(num_nodes)
    adj_mat = utils.sparse_to_tuple(gat_adj.tocoo())
    adj_orig = gdata.get_orig_adj(sperate_name)
    adj_orig = utils.sparse_to_tuple(adj_orig.tocoo())
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    print("pos weight of structure %.5f, norm %.5f"%(pos_weight, norm))
    print("pos weight of content %.5f, norm %.5f"%(f_pos_weight, f_norm))

    #sys.exit(0)

    #build placeholders and hyper parameters
    placeholders = {
                   'features': tf.sparse_placeholder(tf.float32),
                   'features_orig': tf.sparse_placeholder(tf.float32),
                   'adj_mat': tf.sparse_placeholder(tf.float32),
                   'adj_orig': tf.sparse_placeholder(tf.float32),
                   'dropout': tf.placeholder_with_default(0., shape=()),
                   'in_drop': tf.placeholder_with_default(0., shape=()),
                   'coef_drop': tf.placeholder_with_default(0., shape=()),
                   'masked_nodes': tf.placeholder(tf.int32)
           }

    feed_dict = {}
    feed_dict.update({placeholders['dropout']:FLAGS.dropout})
    feed_dict.update({placeholders['in_drop']:FLAGS.in_drop})
    feed_dict.update({placeholders['coef_drop']:FLAGS.coef_drop})
    feed_dict.update({placeholders['adj_mat']:adj_mat})
    feed_dict.update({placeholders['adj_orig']:adj_orig})


    config = {}
    config['learning_rate'] = FLAGS.learning_rate
    config['l2_weight_decay'] = FLAGS.l2_weight_decay
    config['s_weight_decay'] = FLAGS.s_weight_decay
    config['c_weight_decay'] = FLAGS.c_weight_decay
    config['m_weight_decay'] = FLAGS.m_weight_decay
    config['act'] = tf.nn.elu
    config['pos_weight'] = pos_weight
    config['norm'] = norm
    config['f_pos_weight'] = f_pos_weight
    config['f_norm'] = f_norm

    config['re_con_type'] = FLAGS.re_con_type #clf or recon
    config['reverse_type'] = FLAGS.reverse_type #axis of normalizatoin

    config['mask_padding_flag'] = FLAGS.mask_padding_flag

    config['att_key_type'] = FLAGS.att_key_type

    CL = ConfigerLoader()
    config['n_heads_list'], config['hidden_dim_list'], config['layer_type_list'] = CL.load_graphAE_from_json('configer/'+FLAGS.configer_path+'.json')


    if FLAGS.model == 'MaskedGraphAE':
        model = MaskedGraphAE(placeholders=placeholders, num_features=num_features, config=config)
    elif FLAGS.model == 'GraphAE':
        model = GraphAE(placeholders=placeholders, num_features=num_features, config=config)
    elif FLAGS.model == 'GraphBert':
        placeholders['masked_index'] = tf.placeholder(tf.int32)
        model = GraphBert(placeholders=placeholders, num_features=num_features, config=config)

    else:
        print("No such model!!")
        sys.exit(-1)

#init training
    sess = tf.Session()
    sess.run(model.init_op)

    #early stop strategy 1
    train_loss_lastK = []
    #early stop strategy 2
    best = np.inf
    wait = 0
    saver = tf.train.Saver()
    saver.save(sess, "tmp/model_%s_%s_%s_%.2f_%.2f_%.2f_%.2f_%.2f_%s_%s.ckpt"%(FLAGS.model, FLAGS.dataset, FLAGS.configer_path, FLAGS.masked_ratio, FLAGS.in_drop, FLAGS.dropout, FLAGS.s_weight_decay, FLAGS.c_weight_decay, FLAGS.re_con_type, FLAGS.att_key_type))


    masked_nodes = list(range(num_nodes))
    num_of_masked_nodes = int(num_nodes * FLAGS.masked_ratio)

    remask_set = 1


    for k in range(FLAGS.epochs):
        if k % remask_set == 0:
            feed_masked_nodes = np.array(shuffle(masked_nodes)[:num_of_masked_nodes], dtype=np.int32)
            #feed_features_orig = utils.sparse_to_tuple(features[feed_masked_nodes, :])
            feed_features_orig = utils.sparse_to_tuple(features)

        if FLAGS.model == 'MaskedGraphAE':
            if k % remask_set == 0:
                if FLAGS.mask_padding_flag:
                    mask_tag_num = int(0.8 * num_of_masked_nodes)
                    change_tag_num  = int(0.1 * num_of_masked_nodes)
                    keep_tag_num = num_of_masked_nodes - mask_tag_num - change_tag_num

                    feed_features = features.copy()

                    feed_features = utils.enlarge_sparse_matrix_along_dim(feed_features, axis=1, dim_num=1)

                    masked_padding_value = [1.0]
                    masked_padding_index = [int(num_features)]
                    for masked_node in feed_masked_nodes[:mask_tag_num]:
                        feed_features = utils.reset_value_of_spmatrix_by_row(spmatrix=feed_features, row_indx=masked_node, values=masked_padding_value, reset_indx=masked_padding_index)

                    change_mased_nodes = np.array(shuffle(masked_nodes)[:change_tag_num], dtype=np.int32)
                    for i in range(change_tag_num):
                        indx = i + mask_tag_num
                        masked_node = feed_masked_nodes[indx]
                        change_node = change_mased_nodes[i]

                        change_values, change_index = utils.get_value_and_index_of_spmatrxi_by_row(features, change_node)
                        feed_features = utils.reset_value_of_spmatrix_by_row(spmatrix=feed_features, row_indx=masked_node, values=change_values, reset_indx=change_index)
                else:
                    #feed_masked_nodes = np.array(shuffle(masked_nodes)[:num_of_masked_nodes], dtype=np.int32)
                    #feed_features_orig = utils.sparse_to_tuple(features[feed_masked_nodes, :])

                    feed_features = features.copy()
                    feed_features = utils.set_row_of_spmatrix_to_val(feed_features, feed_masked_nodes, 0.0)

                feed_features = utils.sparse_to_tuple(feed_features)
                feed_dict.update({placeholders['masked_nodes']:feed_masked_nodes})
                feed_dict.update({placeholders['features']:feed_features})
                feed_dict.update({placeholders['features_orig']:feed_features_orig})
        elif FLAGS.model == 'GraphBert':
            feed_features = utils.sparse_to_tuple(features)
            feed_dict.update({placeholders['masked_nodes']:feed_masked_nodes})
            feed_dict.update({placeholders['features']:feed_features})
            feed_dict.update({placeholders['features_orig']:feed_features_orig})

            mask_tag_num = int(0.8 * num_of_masked_nodes)
            change_tag_num  = int(0.1 * num_of_masked_nodes)
            keep_tag_num = num_of_masked_nodes - mask_tag_num - change_tag_num
            masked_index = np.array(range(num_nodes), dtype=np.int)
            masked_index[feed_masked_nodes[:mask_tag_num]] = num_nodes
            change_mased_nodes = np.array(shuffle(masked_nodes)[:change_tag_num], dtype=np.int32)
            masked_index[feed_masked_nodes[mask_tag_num:mask_tag_num+change_tag_num]] = change_mased_nodes
            feed_dict.update({placeholders['masked_index']:masked_index})

        elif FLAGS.model == 'GraphAE':
            feed_features = utils.sparse_to_tuple(features)
            feed_dict.update({placeholders['features']:feed_features})

    #debug
        #embeddings, p, debug_emb, inputs = sess.run([model.embeddings, model.p, model.debug_emb, model.inputs], feed_dict=feed_dict)
        #print(embeddings)
        #print(debug_emb)
        #print(inputs)
        #print(p)
        #update parameters
        _ = sess.run(model.opt_op, feed_dict=feed_dict)

        if FLAGS.model == 'MaskedGraphAE' or FLAGS.model == 'GraphBert':
            loss, s_loss, c_loss, l2_loss, m_loss = sess.run([model.loss, model.s_loss, model.c_loss, model.l2_loss, model.m_loss],
                                                      feed_dict=feed_dict)
        else:
            loss, s_loss, c_loss, l2_loss = sess.run([model.loss, model.s_loss, model.c_loss, model.l2_loss],
                                                      feed_dict=feed_dict)

        #embeddings, p, debug_emb, inputs = sess.run([model.embeddings, model.p, model.debug_emb, model.inputs], feed_dict=feed_dict)
        #print(embeddings)
        #print(debug_emb)
        #print(p[1, :2, :])
        if np.isnan(loss):
        #debug
            #embeddings, p, debug_emb, inputs = sess.run([model.embeddings, model.p, model.debug_emb, model.inputs], feed_dict=feed_dict)
            #print(embeddings)
            #print(debug_emb)
            #print(inputs)
            #print(p)
            print("nan loss error!!!!")
            return None
        else:
            if FLAGS.model in ['MaskedGraphAE', 'GraphBert']:
                print("Epoch %d: loss %.5f, structural loss %.5f, content recon loss %.5f, masksed content recon loss %.5f, l2 %.5f"%(k, loss, s_loss, c_loss, m_loss, l2_loss))
            else:
                print("Epoch %d: loss %.5f, structural loss %.5f, content recon loss %.5f, l2 %.5f"%(k, loss, s_loss, c_loss, l2_loss))


        #early stop strategy 1
        #if k > FLAGS.early_stop:
        #    avg_loss_last_k = np.mean(train_loss_lastK[-FLAGS.early_stop:])
        #    if avg_loss_last_k < epoch_metric_loss:
        #        break
        #train_loss_lastK.append(epoch_metric_loss)

        #early stop strategy 2
        if loss < best:
            best = loss
            wait = 0
            save_path = saver.save(sess, "tmp/model_%s_%s_%s_%.2f_%.2f_%.2f_%.2f_%.2f_%s_%s.ckpt"%(FLAGS.model, FLAGS.dataset, FLAGS.configer_path, FLAGS.masked_ratio, FLAGS.in_drop, FLAGS.dropout, FLAGS.s_weight_decay, FLAGS.c_weight_decay, FLAGS.re_con_type, FLAGS.att_key_type))
            #print("Model saved in path: %s" % save_path)
        else:
            wait += 1

        if wait >= FLAGS.early_stop:
            break

        if (k % FLAGS.verbose == 0 and FLAGS.lp and DEBUG_FLAG):
            feed_dict.update({placeholders['dropout']:0.})
            feed_dict.update({placeholders['in_drop']:0.})
            feed_dict.update({placeholders['coef_drop']:0.})
            embeddings = sess.run(model.embeddings, feed_dict=feed_dict)
            roc_score, ap_score = utils.get_roc_score(val_pos_edges, val_neg_edges, embeddings)
            print("result of link prediction: Val roc score:%.5f, AP score:%.5f"%(roc_score, ap_score))

        if (k % FLAGS.verbose == 0 and DEBUG_FLAG):
            feed_dict.update({placeholders['dropout']:0.})
            feed_dict.update({placeholders['in_drop']:0.})
            feed_dict.update({placeholders['coef_drop']:0.})
            embeddings = sess.run(model.embeddings, feed_dict=feed_dict)

            trainX_indices, trainY = gdata.get_node_classification_data(name='train')
            valX_indices, valY = gdata.get_node_classification_data(name='val')
            testX_indices, testY = gdata.get_node_classification_data(name='test')

            trainX = embeddings[trainX_indices]
            valX = embeddings[valX_indices]
            testX = embeddings[testX_indices]

            class_num = np.shape(trainY)[1]
            clf = Classifier('LR', class_num)

            clf.train(trainX, trainY)
            result = clf.evaluate(valX, valY)
            print("val result of %s-%d at epoch %d: micro F1 %.5f, macro F1 %.5f"%(FLAGS.model, FLAGS.hidden1, k, result['micro'], result['macro']))
            result = clf.evaluate(testX, testY)
            print("test result of %s-%d at epoch %d: micro F1 %.5f, macro F1 %.5f"%(FLAGS.model, FLAGS.hidden1, k, result['micro'], result['macro']))

    if FLAGS.model == 'MaskedGraphAE':
        feed_features = features.copy()
        feed_features = utils.enlarge_sparse_matrix_along_dim(feed_features, axis=1, dim_num=1)
        feed_features = utils.sparse_to_tuple(feed_features)
        feed_dict.update({placeholders['features']:feed_features})


    #early stop strategy 2
    saver.restore(sess, "tmp/model_%s_%s_%s_%.2f_%.2f_%.2f_%.2f_%.2f_%s_%s.ckpt"%(FLAGS.model, FLAGS.dataset, FLAGS.configer_path, FLAGS.masked_ratio, FLAGS.in_drop, FLAGS.dropout, FLAGS.s_weight_decay, FLAGS.c_weight_decay, FLAGS.re_con_type, FLAGS.att_key_type))
    #saver.restore(sess, "tmp/model_%s_%s_%d_%d.ckpt"%(FLAGS.model, FLAGS.dataset, config['hidden_dim_list'][-1], FLAGS.DGI_n_cluster))
    print("Model restored.")

    feed_dict.update({placeholders['dropout']:0.})
    feed_dict.update({placeholders['in_drop']:0.})
    feed_dict.update({placeholders['coef_drop']:0.})
    embeddings = sess.run(model.embeddings, feed_dict=feed_dict)
    return embeddings



def train_embedding(gdata):
    #init hyper parameters
    config = init_hyper_parameters(gdata)

    #init input variables of Model
    placeholders = {
                   'features': tf.sparse_placeholder(tf.float32),
                   'context': tf.sparse_placeholder(tf.float32),
                   'neg_context': tf.sparse_placeholder(tf.float32),

                   'context_size': tf.placeholder(tf.float32, shape=(None, 1)),
                   'neg_context_size': tf.placeholder(tf.float32, shape=(None, config['neg_num'], 1)),
                   'dropout': tf.placeholder_with_default(0., shape=())
           }


    sperate_name = 'all'
    if FLAGS.lp:
        val_pos_edges, val_neg_edges = gdata.get_link_prediction_data(data_name='val')
        sperate_name = 'train'

    features = gdata.get_features()
    node_num = len(gdata.get_nodes(sperate_name))
    num_features, features_nonzero = gdata.get_static_information()
    #add zero feature to padding node
    features.resize((node_num+1, num_features))


    model = GaussianContext2Vec(placeholders, num_features, config)

    max_degree = config['max_degree']
    node_list = list(gdata.get_nodes(sperate_name))
    batch_size = FLAGS.batch_size

#create session and init all variables
    sess = tf.Session()
    sess.run(model.init_op)

    train_loss_lastK = []

    for k in range(FLAGS.epochs):
        epoch_loss = 0.
        epoch_metric_loss = 0.
        epoch_l2_loss = 0.
        for i in range(int(len(node_list)/batch_size)+1):
            batch_nodes = node_list[batch_size*i:batch_size*(i+1)]
            batch_context, batch_context_size = build_batch_context(batch_nodes, max_degree, gdata)
            batch_neg_context, batch_neg_context_size = build_batch_neg_context(batch_nodes,
                                                                                 node_list,
                                                                                 config['neg_num'],
                                                                                 max_degree,
                                                                                 gdata)
            batch_context = np.reshape(batch_context, newshape=(-1))
            batch_neg_context = np.reshape(batch_neg_context, newshape=(-1, max_degree))
            batch_neg_context = np.reshape(batch_neg_context, newshape=(-1))

            batch_features = utils.sparse_to_tuple(features[batch_nodes, :])
            batch_context = utils.sparse_to_tuple(features[batch_context, :])
            batch_neg_context = utils.sparse_to_tuple(features[batch_neg_context, :])

            #debug
            #print(batch_features)
            #print(batch_context)
            #print(batch_neg_context)
            #print(np.shape(batch_context_size))
            #print(np.shape(batch_neg_context_size))
            #sys.exit(-1)

            feed_dict = build_feed_dict(placeholders=placeholders,
                                        features=batch_features,
                                        context=batch_context,
                                        neg_context=batch_neg_context,
                                        context_size=batch_context_size,
                                        neg_context_size=batch_neg_context_size,
                                        dropout=FLAGS.dropout)

            #bug code
            #node_mu, node_sigma = sess.run([model.node_mu, model.node_sigma], feed_dict=feed_dict)
            #print(node_mu)
            #print(node_sigma)
            #print("********")
            #pos_v, neg_v = sess.run([model.pos_energy, model.neg_energy], feed_dict=feed_dict)
            #print(pos_v)
            #print(neg_v)
            #print("pos energy:%.5f"%np.sum(pos_v))
            #print("neg energy:%.5f"%np.sum(neg_v))
            #sys.exit(0)

            loss, metrics_loss, l2_loss, _ = sess.run([model.loss, model.metrics_loss, model.l2_loss, model.opt_op],
                                                      feed_dict=feed_dict)

            #debug
            #if math.isnan(loss):
            #    print(node_mu)
            #    print(node_sigma)
            #    sys.exit(0)
            #print("Iter result: loss %.5f, metrics loss %.5f, l2 %.5f"%(loss, metrics_loss, l2_loss))
            epoch_loss += loss
            epoch_metric_loss += metrics_loss
            epoch_l2_loss += l2_loss

            if (k % FLAGS.verbose == 0 and FLAGS.lp):
                feed_dict.update({placeholders['dropout']:0.})
                embeddings = sess.run(model.embeddings, feed_dict=feed_dict)
                roc_score, ap_score = utils.get_roc_score(val_pos_edges, val_neg_edges, embeddings)
                print("result of link prediction: Val roc score:%.5f, AP score:%.5f"%(roc_score, ap_score))

        print("Epoch %d: loss %.5f, metrics loss %.5f, l2 %.5f"%(k, epoch_loss, epoch_metric_loss, epoch_l2_loss))

        #early stop strategy 1
        if k > FLAGS.early_stop:
            avg_loss_last_k = np.mean(train_loss_lastK[-FLAGS.early_stop:])
            if avg_loss_last_k < epoch_metric_loss:
                break
        train_loss_lastK.append(epoch_metric_loss)

        #early stop strategy 2
        if loss < best:
            best = loss
            wait = 0
            save_path = saver.save(sess, "tmp/model_%s_%s_%s_%.2f_%.2f_%.2f_%.2f_%s.ckpt"%(FLAGS.model, FLAGS.dataset, FLAGS.configer_path, FLAGS.masked_ratio, FLAGS.in_drop, FLAGS.dropout, FLAGS.s_weight_decay, FLAGS.re_con_type))
            #print("Model saved in path: %s" % save_path)
        else:
            wait += 1

        if wait >= FLAGS.early_stop:
            break



        if (i % FLAGS.verbose == 0):
            feed_dict.update({placeholders['dropout']:0.})
            mu_embeddings, sigma_embeddings = sess.run([model.node_mu, model.node_sigma], feed_dict=feed_dict)
            embeddings = np.concatenate((mu_embeddings, sigma_embeddings), axis=1)
            trainX_indices, trainY = gdata.get_node_classification_data(name='train')
            valX_indices, valY = gdata.get_node_classification_data(name='val')
            testX_indices, testY = gdata.get_node_classification_data(name='test')

            trainX = embeddings[trainX_indices]
            valX = embeddings[valX_indices]
            testX = embeddings[testX_indices]

            class_num = np.shape(trainY)[1]
            #clf = Classifier('LR', class_num)
            clf = Classifier('SVM', class_num)

            clf.train(trainX, trainY)
            result = clf.evaluate(valX, valY)
            print("val result of %s-%d at epoch %d: micro F1 %.5f, macro F1 %.5f"%(FLAGS.model, FLAGS.hidden1, k, result['micro'], result['macro']))
            result = clf.evaluate(testX, testY)
            print("test result of %s-%d at epoch %d: micro F1 %.5f, macro F1 %.5f"%(FLAGS.model, FLAGS.hidden1, k, result['micro'], result['macro']))


    #early stop strategy 2
    saver.restore(sess, "tmp/model_%s_%s_%s_%.2f_%.2f_%.2f_%.2f_%s.ckpt"%(FLAGS.model, FLAGS.dataset, FLAGS.configer_path, FLAGS.masked_ratio, FLAGS.in_drop, FLAGS.dropout, FLAGS.s_weight_decay, FLAGS.re_con_type))
    #saver.restore(sess, "tmp/model_%s_%s_%d_%d.ckpt"%(FLAGS.model, FLAGS.dataset, config['hidden_dim_list'][-1], FLAGS.DGI_n_cluster))
    print("Model restored.")

    feed_dict = {}
    feed_dict[placeholders['features']] = utils.sparse_to_tuple(features)
    feed_dict[placeholders['dropout']] = 0.
    mu_embeddings, sigma_embeddings = sess.run([model.node_mu, model.node_sigma], feed_dict=feed_dict)
    embeddings = np.concatenate((mu_embeddings, sigma_embeddings), axis=1)

    #debug
    #node_mu, node_sigma = sess.run([model.node_mu, model.node_sigma], feed_dict=feed_dict)
    #print(node_mu)
    #print(node_sigma)

    return embeddings

def train_embedding_MinibatchDGI(gdata):
    #num_data, train_adj, adj, feats, train_feats, test_feats, labels, train_data, val_data, test_data = gdata.get_data()
    num_data, adj, feats, train_feats, test_feats, labels, train_data, val_data, test_data = gdata.get_data()
    print("adj shape (%d, %d)"%(adj.shape[0], adj.shape[1]))

    train_adj = adj[train_data, :][:, train_data]
    print("train adj shape (%d, %d)"%(train_adj.shape[0], train_adj.shape[1]))

    #
    num_features = feats.shape[1]

    if True or FLAGS.model == 'ClusterDGI':
        print("start to partition graph")
        start_time = time.time()
        (edgecuts, cluster_label_list) = metis.part_graph(nx.from_scipy_sparse_matrix(train_adj), FLAGS.DGI_n_cluster)
        print("process graph partition end, costs %f seconds"%(time.time()-start_time))

        cluster_contain_list = []
        for i in range(FLAGS.DGI_n_cluster):
            cluster_contain_list.append([])
        for idx, label in enumerate(cluster_label_list):
            cluster_contain_list[label].append(idx)

        cluster_label_list = np.array(cluster_label_list)
        node2cluster = utils.binarize_label(cluster_label_list).astype(np.float64)


    config = {}
    config['learning_rate'] = FLAGS.learning_rate

    config['node_num'] = num_data
    config['feature_dim'] = num_features
    config['global_weight'] = FLAGS.global_weight
    #config['act'] = tf.keras.layers.PReLU
    config['act'] = tf.nn.elu
    #config['act'] = tf.nn.relu
    config['feature_dim'] = num_features

    CL = ConfigerLoader()
    config['n_heads_list'], config['hidden_dim_list'], config['layer_type_list'] = CL.load_graphAE_from_json('configer/'+FLAGS.configer_path+'.json')

    placeholders = {
                   'inputs': tf.sparse_placeholder(tf.float32),
                   'adj': tf.sparse_placeholder(tf.float32),
                   'corp_inputs': tf.sparse_placeholder(tf.float32),
                   'corp_adj': tf.sparse_placeholder(tf.float32),
                   'dropout': tf.placeholder_with_default(0., shape=()),
                   'labels': tf.placeholder(tf.float32)
           }


    feed_dict = {}
    feed_dict.update({placeholders['dropout']:FLAGS.dropout})

    placeholders['cluster_labels'] = tf.placeholder(shape=[None, FLAGS.DGI_n_cluster], dtype=tf.float32)


    #feed_dict.update({placeholders['cluster_labels']:cluster_labels})
    #config['select_eigen_value'] = select_eigen_value

    #for i in range(select_eigen_value):
    #    placeholders['eigen_vector_%d'%i] = tf.sparse_placeholder(tf.float32)
    #    feed_dict.update({placeholders['eigen_vector_%d'%i]:utils.sparse_to_tuple(pooling_matrices[i])})

    model = ClusterDGI(placeholders=placeholders, config=config)

    sess = tf.Session()
    sess.run(model.init_op)

    best = np.inf
    wait = 0
    saver = tf.train.Saver()

    re_corp_step = 1

    cmbg = ClusterMiniBatchGenerator(adj=train_adj, features=train_feats, cluster_contain_list=cluster_contain_list, node2cluster=node2cluster, batch_size=FLAGS.clusters_per_batch)

    print("start training embedding")
    for i in range(FLAGS.epochs):
        loss = 0.
        local_loss = 0.
        global_loss = 0.
        while not cmbg.end:
            batch_adj, batch_features, batch_node2cluster, _ = cmbg.generate()
            node_num = batch_adj.shape[0]
            if i % re_corp_step == 0:
                if FLAGS.corp_flag_x:
                    corp_inputs = utils.corruption_function(batch_features)
                else:
                    corp_inputs = batch_features

                if FLAGS.corp_flag_adj:
                    corp_adj = corruption_function(batch_adj)
                else:
                    corp_adj = batch_adj

            #build sparse tensor
            feed_features = utils.preprocess_features(batch_features)
            feed_corp_inputs = utils.preprocess_features(corp_inputs)
            #features = utils.sparse_to_tuple(features)
            #corp_inputs = utils.sparse_to_tuple(corp_inputs)
            feed_adj = utils.preprocess_graph(batch_adj)
            feed_corp_adj = utils.preprocess_graph(corp_adj)

            #adj = utils.sparse_to_tuple(adj)
            #corp_adj = utils.sparse_to_tuple(corp_adj)

            #build self-supervised labels
            ones = np.ones(shape=[node_num, 1], dtype=np.float32)
            zeros = np.zeros(shape=[node_num, 1], dtype=np.float32)
            batch_labels = np.concatenate((ones, zeros), axis=0)


            feed_dict.update({placeholders['inputs']:feed_features})
            feed_dict.update({placeholders['adj']:feed_adj})
            feed_dict.update({placeholders['corp_inputs']:feed_corp_inputs})
            feed_dict.update({placeholders['corp_adj']:feed_corp_adj})
            feed_dict.update({placeholders['cluster_labels']:batch_node2cluster})
            feed_dict.update({placeholders['labels']:batch_labels})


            if FLAGS.model == 'ClusterDGI':
                batch_loss, batch_local_loss, batch_global_loss, _ = sess.run([model.loss, model.local_loss, model.global_loss, model.opt_op], feed_dict=feed_dict)
                loss += batch_loss
                local_loss += batch_local_loss
                global_loss += batch_global_loss
            else:
                batch_loss, _ = sess.run([model.loss, model.opt_op], feed_dict=feed_dict)

        if FLAGS.model == 'ClusterDGI':
            print("Epoch %d, loss %.10f, local %.8f, global %.8f"%(i, loss, local_loss, global_loss))
        else:
            print("Epoch %d, loss %.10f"%(i, loss))
        cmbg.refresh()

        if loss < best:
            best = loss
            wait = 0
            #save_path = saver.save(sess, "tmp/model_%s_%s_%d_%d.ckpt"%(FLAGS.model, FLAGS.dataset, FLAGS.hidden1, FLAGS.DGI_n_cluster))
            save_path = saver.save(sess, "tmp/model_%s_%s_%d_%s_%s_%d_%.3f_mini.ckpt"%(FLAGS.model, FLAGS.dataset, FLAGS.DGI_n_cluster, FLAGS.configer_path, FLAGS.DGI_loss_type, FLAGS.neg_size, FLAGS.global_weight))
            #print("Model saved in path: %s" % save_path)
        else:
            wait += 1

        if wait >= FLAGS.early_stop:
            break

    saver.restore(sess, "tmp/model_%s_%s_%d_%s_%s_%d_%.3f_mini.ckpt"%(FLAGS.model, FLAGS.dataset, FLAGS.DGI_n_cluster, FLAGS.configer_path, FLAGS.DGI_loss_type, FLAGS.neg_size, FLAGS.global_weight))
    #saver.restore(sess, "tmp/model_%s_%s_%d_%d.ckpt"%(FLAGS.model, FLAGS.dataset, config['hidden_dim_list'][-1], FLAGS.DGI_n_cluster))
    print("Model restored.")

    #build inductive test node embedding
    print("start to process embeddings !!")
    start_time = time.time()
    (edgecuts, cluster_label_list) = metis.part_graph(nx.from_scipy_sparse_matrix(adj), FLAGS.DGI_n_cluster)
    print("process graph partition end, costs %f seconds"%(time.time()-start_time))

    cluster_contain_list = []
    for i in range(FLAGS.DGI_n_cluster):
        cluster_contain_list.append([])
    for idx, label in enumerate(cluster_label_list):
        cluster_contain_list[label].append(idx)

    cluster_label_list = np.array(cluster_label_list)
    node2cluster = utils.binarize_label(cluster_label_list).astype(np.float64)

    test_cmbg = ClusterMiniBatchGenerator(adj=adj, features=feats, cluster_contain_list=cluster_contain_list, node2cluster=node2cluster, batch_size=FLAGS.clusters_per_batch)
    embeddings = np.zeros(shape=(adj.shape[0], config['hidden_dim_list'][-1]))

    start_time = time.time()
    repeat = 5
    for i in range(repeat):
        test_cmbg.refresh()
        while not test_cmbg.end:
            #print("debug 1")
            batch_adj, batch_features, batch_node2cluster, batch_nodes = test_cmbg.generate()
            feed_features = utils.preprocess_features(batch_features)
            feed_adj = utils.preprocess_graph(batch_adj)
            feed_dict.update({placeholders['inputs']:feed_features})
            feed_dict.update({placeholders['adj']:feed_adj})

            embeds = sess.run(model.embeddings, feed_dict=feed_dict)
            embeddings[batch_nodes] += embeds
    print("process test embedding end, costs %f seconds"%(time.time()-start_time))

    embeddings = embeddings / float(repeat)
    return embeddings

def training_embedding_PPRDGI(gdata):
    sperate_name = 'all'
    if FLAGS.lp:
        val_pos_edges, val_neg_edges = gdata.get_link_prediction_data(data_name='val')
        sperate_name = 'train'

    trainG = gdata.get_graph(sperate_name)

    features = gdata.get_features()
    sta_information = gdata.get_static_information()
    num_features = sta_information[0]

    adj = gdata.get_adj(sperate_name)
    num_nodes = adj.shape[0]


    #calculate personal pagerank
    ppr_alpha = FLAGS.ppr_alpha
    ppr_values = utils.get_personal_pagerank_values(adj, dataset=FLAGS.dataset, alpha=ppr_alpha, sparse=FLAGS.ppr_sparse, topK=FLAGS.ppr_topK)


    #print(ppr_values.shape)
    #print(corp_ppr_values.shape)
    #print(ppr_values.nnz)
    #print(corp_ppr_values.nnz)
    print("get personal pagerank value end.....")



    config = {}
    config['learning_rate'] = FLAGS.learning_rate
    config['act'] = tf.nn.relu
    #config['act'] = tf.nn.elu
    #config['act'] = tf.keras.layers.PReLU

    #personal pagerank readout activation function
    config['ppr_act'] = tf.sigmoid
    #config['ppr_act'] = tf.nn.relu

    config['node_num'] = num_nodes
    config['feature_dim'] = num_features


    #load network architecture
    CL = ConfigerLoader()
    config['n_heads_list'], config['hidden_dim_list'], config['layer_type_list'] = CL.load_graphAE_from_json('configer/'+FLAGS.configer_path+'.json')

    placeholders = {
                   'inputs': tf.sparse_placeholder(tf.float32),
                   'adj': tf.sparse_placeholder(tf.float32),
                   'corp_inputs': tf.sparse_placeholder(tf.float32),
                   'corp_adj': tf.sparse_placeholder(tf.float32),
                   'dropout': tf.placeholder_with_default(0., shape=()),
                   'in_drop': tf.placeholder_with_default(0., shape=()),
                   'coef_drop': tf.placeholder_with_default(0., shape=()),
                   'labels': tf.placeholder(tf.float32),
                   'ppr_values': tf.sparse_placeholder(tf.float32),
                   'corp_ppr_values': tf.sparse_placeholder(tf.float32)
           }

    feed_dict = {}
    feed_dict.update({placeholders['dropout']:FLAGS.dropout})
    feed_dict.update({placeholders['in_drop']:FLAGS.in_drop})
    feed_dict.update({placeholders['coef_drop']:FLAGS.coef_drop})

    ones = np.ones(shape=[num_nodes, 1], dtype=np.float32)
    zeros = np.zeros(shape=[num_nodes, 1], dtype=np.float32)
    labels = np.concatenate((ones, zeros), axis=0)
    feed_dict.update({placeholders['labels']:labels})



    model = PPRDGI(placeholders=placeholders, config=config)

    sess = tf.Session()
    sess.run(model.init_op)

    best = np.inf
    wait = 0
    saver = tf.train.Saver()
    #init save
    save_path = saver.save(sess, "tmp/model_%s_%s_%d_%s_%s_%d_%.3f_%.2f.ckpt"%(FLAGS.model, FLAGS.dataset, FLAGS.DGI_n_cluster, FLAGS.configer_path, FLAGS.DGI_loss_type, FLAGS.neg_size, FLAGS.global_weight, FLAGS.train_label_ratio))

    re_corp_step = 1


    for i in range(FLAGS.epochs):
        if FLAGS.corp_flag_x:
            corp_inputs = utils.corruption_function(features)
        else:
            corp_inputs = features

        if FLAGS.corp_flag_adj:
            corp_adj = utils.corruption_function(adj)
        else:
            corp_adj = adj





        #build sparse tensor
        feed_features = utils.preprocess_features(features)
        feed_corp_inputs = utils.preprocess_features(corp_inputs)
        #features = utils.sparse_to_tuple(features)
        #corp_inputs = utils.sparse_to_tuple(corp_inputs)

        feed_adj = utils.preprocess_graph(adj)
        feed_corp_adj = utils.preprocess_graph(corp_adj)
        #adj = utils.sparse_to_tuple(adj)
        #corp_adj = utils.sparse_to_tuple(corp_adj)

        feed_dict.update({placeholders['inputs']:feed_features})
        feed_dict.update({placeholders['adj']:feed_adj})
        feed_dict.update({placeholders['corp_inputs']:feed_corp_inputs})
        feed_dict.update({placeholders['corp_adj']:feed_corp_adj})

        corp_ppr_values = utils.corruption_function(ppr_values)
        feed_ppr_values = utils.preprocess_features(ppr_values)
        feed_corp_ppr_values = utils.preprocess_features(corp_ppr_values)

        feed_dict.update({placeholders['ppr_values']:feed_ppr_values})
        feed_dict.update({placeholders['corp_ppr_values']:feed_corp_ppr_values})

        #feed_dict.update({placeholders['dropout']:FLAGS.dropout})
        #feed_dict.update({placeholders['in_drop']:FLAGS.in_drop})
        #feed_dict.update({placeholders['coef_drop']:FLAGS.coef_drop})




        loss, _ = sess.run([model.loss, model.opt_op], feed_dict=feed_dict)
        print("Epoch %d, loss %.10f"%(i, loss))


        if loss < best:
            best = loss
            wait = 0
            #save_path = saver.save(sess, "tmp/model_%s_%s_%d_%d.ckpt"%(FLAGS.model, FLAGS.dataset, config['hidden_dim_list'][-1], FLAGS.DGI_n_cluster))
            save_path = saver.save(sess, "tmp/model_%s_%s_%d_%s_%s_%d_%.3f_%.2f.ckpt"%(FLAGS.model, FLAGS.dataset, FLAGS.DGI_n_cluster, FLAGS.configer_path, FLAGS.DGI_loss_type, FLAGS.neg_size, FLAGS.global_weight, FLAGS.train_label_ratio))
            #print("Model saved in path: %s" % save_path)
        else:
            wait += 1

        if wait >= FLAGS.early_stop:
            break

    saver.restore(sess, "tmp/model_%s_%s_%d_%s_%s_%d_%.3f_%.2f.ckpt"%(FLAGS.model, FLAGS.dataset, FLAGS.DGI_n_cluster, FLAGS.configer_path, FLAGS.DGI_loss_type, FLAGS.neg_size, FLAGS.global_weight, FLAGS.train_label_ratio))

    print("Model restored.")
    feed_dict.update({placeholders['dropout']:0.})
    feed_dict.update({placeholders['in_drop']:0.})
    feed_dict.update({placeholders['coef_drop']:0.})
    embeddings = sess.run(model.embeddings, feed_dict=feed_dict)


    return embeddings






def training_embedding_DGI(gdata):
    sperate_name = 'all'
    if FLAGS.lp:
        val_pos_edges, val_neg_edges = gdata.get_link_prediction_data(data_name='val')
        sperate_name = 'train'


    trainG = gdata.get_graph(sperate_name)

    features = gdata.get_features()
    sta_information = gdata.get_static_information()
    num_features = sta_information[0]

    adj = gdata.get_adj(sperate_name)
    num_nodes = adj.shape[0]

    if FLAGS.model in['ClusterDGI', 'PCDGI', 'NewClusterDGI', 'CommunityDGI']:

        #spectral cluster
        #pre_cluster = SpectralClustering(n_clusters=FLAGS.DGI_n_cluster, assign_labels='discretize', affinity='precomputed')
        #pre_cluster.fit(adj)
        #build label matrix Y
        #cluster_label_list = pre_cluster.labels_
        #print(cluster_labels)

        #metis
        (edgecuts, cluster_label_list) = metis.part_graph(nx.from_scipy_sparse_matrix(adj), FLAGS.DGI_n_cluster)
        cluster_label_list = np.array(cluster_label_list)


        cluster_labels = utils.binarize_label(cluster_label_list).astype(np.float32)

        #debug ture distribution
        #_, cluster_labels = gdata.get_node_classification_data('all')
        print("cluster shape" + str(cluster_labels.shape))


        #record node cluster label distribute used for build subgraph
        cluster = {}
        for inx, label in enumerate(cluster_label_list):
            if label not in cluster:
                cluster[label] = []
            cluster[label].append(inx)


        #sub_adj_list = []
        #for i in range(FLAGS.DGI_n_cluster):
        #    sub_adj_list.append(adj[cluster[i],:][:,cluster[i]])

        ##eigen decomposition
        #eigen_value_num = 10
        #normalized = True
        ##normalized = False
        #for k in cluster:
        #    eigen_value_num = min(len(cluster[k]), eigen_value_num)

        #select_eigen_value = int(eigen_value_num/2)
        #print("min cluster size %d, select eigen value %d"%(eigen_value_num, select_eigen_value))

        #eigen_vectors = []
        #for i in range(FLAGS.DGI_n_cluster):
        #    L = utils.laplacian_adj(sub_adj_list[i], normalized)
        #    e, ev = eigs(A=L, k=select_eigen_value, which='SM')
        #    eigen_vectors.append(ev)

        ##build feature pooling operator
        #pooling_matrices = [sp.lil_matrix((num_nodes, FLAGS.DGI_n_cluster)) for i in range(select_eigen_value)]
        #for i in range(select_eigen_value):
        #    for j in range(FLAGS.DGI_n_cluster):
        #        pooling_matrices[i][cluster[j],j] = eigen_vectors[j][:,i].reshape(-1,1)


    config = {}
    config['learning_rate'] = FLAGS.learning_rate
    config['act'] = tf.nn.relu
    #config['act'] = tf.nn.elu
    #config['act'] = tf.keras.layers.PReLU
    config['node_num'] = num_nodes
    config['feature_dim'] = num_features
    config['global_weight'] = FLAGS.global_weight
    config['local_weight'] = FLAGS.local_weight
    config['loss_type'] = FLAGS.DGI_loss_type
    config['neg_size'] = FLAGS.neg_size
    config['community_num'] = FLAGS.DGI_n_cluster

    if FLAGS.model == 'PCDGI':
        config['gb_temp'] = FLAGS.gb_temp
        config['pred_cluster_weight'] = FLAGS.pred_cluster_weight

    #load network architecture
    CL = ConfigerLoader()
    config['n_heads_list'], config['hidden_dim_list'], config['layer_type_list'] = CL.load_graphAE_from_json('configer/'+FLAGS.configer_path+'.json')

    placeholders = {
                   'inputs': tf.sparse_placeholder(tf.float32),
                   'adj': tf.sparse_placeholder(tf.float32),
                   'corp_inputs': tf.sparse_placeholder(tf.float32),
                   'corp_adj': tf.sparse_placeholder(tf.float32),
                   'dropout': tf.placeholder_with_default(0., shape=()),
                   'in_drop': tf.placeholder_with_default(0., shape=()),
                   'coef_drop': tf.placeholder_with_default(0., shape=()),
                   'labels': tf.placeholder(tf.float32),
                   'pos_weights': tf.placeholder(tf.float32)
           }


    feed_dict = {}
    feed_dict.update({placeholders['dropout']:FLAGS.dropout})
    feed_dict.update({placeholders['in_drop']:FLAGS.in_drop})
    feed_dict.update({placeholders['coef_drop']:FLAGS.coef_drop})

    if FLAGS.DGI_loss_type == 'JSD' or FLAGS.DGI_loss_type == 'KL':
        ones = np.ones(shape=[num_nodes, 1], dtype=np.float32)
        zeros = np.zeros(shape=[num_nodes, 1], dtype=np.float32)
        labels = np.concatenate((ones, zeros), axis=0)
        feed_dict.update({placeholders['labels']:labels})
    elif FLAGS.DGI_loss_type == 'NCE':
        config['neg_size'] = FLAGS.neg_size
        placeholders['neg_context'] = tf.placeholder(tf.int32)
    else:
        print("no such loss! ERROR")
        sys.exit(-1)


    if FLAGS.model == 'DGI':
        model = DGI(placeholders=placeholders, config=config)
    elif FLAGS.model == 'EgoDGI':
        model = EgoDGI(placeholders=placeholders, config=config)

    elif FLAGS.model == 'PCDGI':
        config['cluster_labels'] = cluster_labels
        config['n_cluster'] = FLAGS.DGI_n_cluster
        #placeholders['corp_cluster_labels'] = tf.placeholder(shape=[None, FLAGS.DGI_n_cluster], dtype=tf.float32)
        #pos_weights = np.ones(shape=(num_nodes*2, 1))
        model = PCDGI(placeholders=placeholders, config=config)

    elif FLAGS.model in ['ClusterDGI', 'NewClusterDGI', 'CommunityDGI']:
        placeholders['cluster_labels'] = tf.placeholder(shape=[None, FLAGS.DGI_n_cluster], dtype=tf.float32)
        feed_dict.update({placeholders['cluster_labels']:cluster_labels})
        if FLAGS.model == 'NewClusterDGI':
            placeholders['corp_cluster_labels'] = tf.placeholder(shape=[None, FLAGS.DGI_n_cluster], dtype=tf.float32)
            pos_weights = np.ones(shape=(num_nodes*2, 1))
        #debug
        #cluster_labels = np.random.rand(cluster_labels.shape[0], cluster_labels.shape[1])
        #cluster_labels = np.argmax(cluster_labels, axis=1)
        #lb = LabelBinarizer()
        #lb.fit(list(range(FLAGS.DGI_n_cluster)))
        #cluster_labels = lb.transform(cluster_labels)
        #feed_dict.update({placeholders['cluster_labels']:cluster_labels})

        print("show cluster distribution" + str(np.sum(cluster_labels, axis=0)))

        trainX_indices, trainY = gdata.get_node_classification_data(name='train')
        valX_indices, valY = gdata.get_node_classification_data(name='val')
        testX_indices, testY = gdata.get_node_classification_data(name='test')
        true_labels = np.sum(trainY, axis=0) + np.sum(valY, axis=0) + np.sum(testY, axis=0)
        print("true label distribution" + str(true_labels))


        #config['select_eigen_value'] = select_eigen_value
        #for i in range(select_eigen_value):
        #    placeholders['eigen_vector_%d'%i] = tf.sparse_placeholder(tf.float32)
        #    feed_dict.update({placeholders['eigen_vector_%d'%i]:utils.sparse_to_tuple(pooling_matrices[i])})

        if FLAGS.model == 'ClusterDGI':
            model = ClusterDGI(placeholders=placeholders, config=config)
        elif FLAGS.model == 'CommunityDGI':
            placeholders['corp_cluster_labels'] = tf.placeholder(shape=[None, FLAGS.DGI_n_cluster], dtype=tf.float32)
            model = CommunityDGI(placeholders=placeholders, config=config)
            community_init = False
        else:
            config['debug_type1'] = FLAGS.debug_type1
            config['debug_type2'] = FLAGS.debug_type2
            config['debug_type3'] = FLAGS.debug_type3
            model = NewClusterDGI(placeholders=placeholders, config=config)

    sess = tf.Session()
    sess.run(model.init_op)

    best = np.inf
    wait = 0
    saver = tf.train.Saver()
    #init save
    save_path = saver.save(sess, "tmp/model_%s_%s_%d_%s_%s_%d_%.3f_%.2f.ckpt"%(FLAGS.model, FLAGS.dataset, FLAGS.DGI_n_cluster, FLAGS.configer_path, FLAGS.DGI_loss_type, FLAGS.neg_size, FLAGS.global_weight, FLAGS.train_label_ratio))

    re_corp_step = 1


    for i in range(FLAGS.epochs):

        if i % re_corp_step == 0:
            if FLAGS.corp_flag_x:
                corp_inputs = utils.corruption_function(features)
            else:
                corp_inputs = features

            if FLAGS.corp_flag_adj:
                corp_adj = utils.corruption_function(adj)
            else:
                corp_adj = adj

            #build sparse tensor
            feed_features = utils.preprocess_features(features)
            feed_corp_inputs = utils.preprocess_features(corp_inputs)
            #features = utils.sparse_to_tuple(features)
            #corp_inputs = utils.sparse_to_tuple(corp_inputs)
            feed_adj = utils.preprocess_graph(adj)
            feed_corp_adj = utils.preprocess_graph(corp_adj)

            #adj = utils.sparse_to_tuple(adj)
            #corp_adj = utils.sparse_to_tuple(corp_adj)


            feed_dict.update({placeholders['inputs']:feed_features})
            feed_dict.update({placeholders['adj']:feed_adj})
            feed_dict.update({placeholders['corp_inputs']:feed_corp_inputs})
            feed_dict.update({placeholders['corp_adj']:feed_corp_adj})
            feed_dict.update({placeholders['dropout']:FLAGS.dropout})
            feed_dict.update({placeholders['in_drop']:FLAGS.in_drop})
            feed_dict.update({placeholders['coef_drop']:FLAGS.coef_drop})

        if FLAGS.DGI_loss_type == 'NCE':
            config['neg_size'] = FLAGS.neg_size
            neg_context = np.random.randint(low=0, high=num_nodes, size=(num_nodes, FLAGS.neg_size), dtype=np.int32)
            feed_dict.update({placeholders['neg_context']:neg_context})


        if FLAGS.model == 'ClusterDGI' or FLAGS.model == 'NewClusterDGI':
            if FLAGS.model == 'NewClusterDGI':
                if i % re_corp_step == 0:
                    corp_cluster_labels = utils.cluster_corruption_function(cluster_labels, 'new')
                feed_dict.update({placeholders['pos_weights']:pos_weights})
                feed_dict.update({placeholders['corp_cluster_labels']:corp_cluster_labels})
            loss, local_loss, global_loss, _ = sess.run([model.loss, model.local_loss, model.global_loss, model.opt_op], feed_dict=feed_dict)
            print("Epoch %d, loss %.10f, local %.8f, global %.8f"%(i, loss, local_loss, global_loss))
        elif FLAGS.model == 'EgoDGI':
            loss, _ = sess.run([model.loss, model.opt_op], feed_dict=feed_dict)
            print("Epoch %d, loss %.10f"%(i, loss))
        elif FLAGS.model == 'DGI':
            loss, _ = sess.run([model.loss, model.opt_op], feed_dict=feed_dict)
            print("Epoch %d, loss %.10f"%(i, loss))
        elif FLAGS.model == 'PCDGI':
            '''     
step by step training version       
            emb_run_step = 10
            cluster_run_step = 10
            if i % re_corp_step == 0:
                #cluster_labels = np.array(cluster_labels)
                corp_cluster_labels = utils.cluster_corruption_function(cluster_labels, 'new')
            feed_dict.update({placeholders['pos_weights']:pos_weights})
            feed_dict.update({placeholders['corp_cluster_labels']:corp_cluster_labels})
            feed_dict.update({placeholders['cluster_labels']:cluster_labels})

            for k in range(emb_run_step):
                if i == 0:
                    loss, local_loss, global_loss, _ = sess.run([model.loss_init, model.local_loss_init, model.global_loss, model.opt_op_e_init], feed_dict=feed_dict)
                else:
                    loss, local_loss, global_loss, _ = sess.run([model.loss, model.local_loss, model.global_loss, model.opt_op_e], feed_dict=feed_dict)
                print("Update encoder, loss %.10f, local %.8f, global %.8f"%(loss, local_loss, global_loss))
            for k in range(cluster_run_step):
                loss, local_loss, global_loss, _ = sess.run([model.loss, model.local_loss, model.global_loss, model.opt_op_c], feed_dict=feed_dict)
                print("Update cluster, loss %.10f, local %.8f, global %.8f"%(loss, local_loss, global_loss))
            cluster_labels = sess.run(model.cluster_var, feed_dict=feed_dict)
            print("Epoch %d, loss %.10f, local %.8f, global %.8f"%(i, loss, local_loss, global_loss))
'''
            _, loss, IM_loss, kl_loss = sess.run([model.opt_op, model.loss, model.IM_loss, model.loss_label_prob_kl], feed_dict=feed_dict)
            print("Epoch %d, loss %.10f, Info Loss %.10f, KL Loss %.10f"%(i, loss, IM_loss, kl_loss))

            #debug
            pos_logits, neg_logits, prob, prob_var = sess.run([model.debug_pos, model.debug_neg, model.cluster_prob, model.cluster_var], feed_dict=feed_dict)
            #print(pos_logits)
            #print(neg_logits)
            #print(prob[1:10])
            print(prob_var[1])
            #print("Debug %.10f, %.10f"%(pos_logits, neg_logits))
        elif FLAGS.model == 'CommunityDGI':
            if not community_init:
                for i in range(FLAGS.epochs):
                    loss, _ = sess.run([model.dgi_loss, model.dgi_opt_op], feed_dict=feed_dict)
                    print("Epoch %d, init encoder loss %.10f"%(i, loss))
                community_init = True
                embeddings = sess.run(model.embeddings, feed_dict=feed_dict)
                print("############# Res After Init ##############")
                clf_function(embeddings, gdata)
                print("###########################################")
                init_community_h = cluster_labels.T.dot(embeddings) / np.expand_dims(np.sum(cluster_labels, axis=0), axis=1)
                sess.run(tf.assign(model.mu, init_community_h))

                P = list(range(10000))#debug
            
            #corp_cluster_labels = utils.cluster_corruption_function(cluster_labels, 'new')

            corp_cluster_labels = utils.cluster_corruption_function(cluster_labels, 'raw')
            print("....................")
            print(cluster_labels[10])
            print(P[10])
            print(corp_cluster_labels[10])
            print(cluster_labels[200])
            print(P[200])
            print(corp_cluster_labels[200])
            print(cluster_labels[1050])
            print(P[1050])
            print(corp_cluster_labels[1050])
            print("....................")
            feed_dict.update({placeholders['corp_cluster_labels']:corp_cluster_labels})
            feed_dict.update({placeholders['cluster_labels']:cluster_labels})
            #dgi_run_step = 10
            #dec_run_step = 10
            #for k in range(dgi_run_step):
            #    MI_loss, _ = sess.run([model.MI_loss, model.MI_op], feed_dict=feed_dict)
            #    print("Update DGI decoder, MI loss %.8f"%MI_loss)
            #for k in range(dec_run_step):
            #    DEC_loss, _ = sess.run([model.DEC_loss, model.DEC_op], feed_dict=feed_dict)
            #    print("Update DEC decoder, DEC loss %.8f"%DEC_loss)
            MI_loss, DEC_loss= sess.run([model.MI_loss, model.DEC_loss], feed_dict=feed_dict)
            print("MI loss %.8f, DEC loss %.8f"%(MI_loss, DEC_loss))
            MI_loss, DEC_loss, _ = sess.run([model.MI_loss, model.DEC_loss, model.opt_op], feed_dict=feed_dict)
            cluster_labels, P = sess.run([model.Q, model.P], feed_dict=feed_dict)
            #print(cluster_labels.shape)
            #sys.exit(0)
            loss =  MI_loss + DEC_loss
        else:
            print("no such model, error!!!")
            sys.exit(-1)



        if FLAGS.DGI_kmeans:
            if (i+1) % FLAGS.DGI_kmeans_step == 0:
                start_time = time.time()
                embeddings = sess.run(model.embeddings, feed_dict=feed_dict)
                kmeans_pred = KMeans(n_clusters=FLAGS.DGI_n_cluster, random_state=0).fit_transform(embeddings)

                #hard clustering
                new_cluster_labels = np.argmin(kmeans_pred, axis=1)
                lb = LabelBinarizer()
                lb.fit(list(range(FLAGS.DGI_n_cluster)))
                new_cluster_labels = lb.transform(new_cluster_labels)

                #soft clustering
                #kmeans_pred[kmeans_pred < 1e-10] = 1e-10
                #cluster_labels = utils.softmax(1.0 / kmeans_pred, axis=1)
                #cluster_labels = 1.0 - kmeans_pred
                #print("show cluster distribution" + str(cluster_labels[0]))

                print("show cluster distribution" + str(np.sum(new_cluster_labels, axis=0)))

                #cluster_labels = 0.5 * new_cluster_labels + 0.5 * cluster_labels
                cluster_labels = new_cluster_labels

                #build pos weights
                pos_weights = np.exp(-np.reshape(kmeans_pred[cluster_labels>0], newshape=(num_nodes, 1)))
                pos_weights = np.concatenate((pos_weights, np.ones(shape=(num_nodes, 1))))



                feed_dict.update({placeholders['cluster_labels']:cluster_labels})
                print("process kmeans end, costs %f seconds"%(time.time()-start_time))




        #logits, labels = sess.run([model.logits, model.labels], feed_dict=feed_dict)
        #print(labels)

        if loss < best:
            best = loss
            wait = 0
            #save_path = saver.save(sess, "tmp/model_%s_%s_%d_%d.ckpt"%(FLAGS.model, FLAGS.dataset, config['hidden_dim_list'][-1], FLAGS.DGI_n_cluster))
            save_path = saver.save(sess, "tmp/model_%s_%s_%d_%s_%s_%d_%.3f_%.2f.ckpt"%(FLAGS.model, FLAGS.dataset, FLAGS.DGI_n_cluster, FLAGS.configer_path, FLAGS.DGI_loss_type, FLAGS.neg_size, FLAGS.global_weight, FLAGS.train_label_ratio))
            #print("Model saved in path: %s" % save_path)
        else:
            wait += 1

        if wait >= FLAGS.early_stop:
            break
    saver.restore(sess, "tmp/model_%s_%s_%d_%s_%s_%d_%.3f_%.2f.ckpt"%(FLAGS.model, FLAGS.dataset, FLAGS.DGI_n_cluster, FLAGS.configer_path, FLAGS.DGI_loss_type, FLAGS.neg_size, FLAGS.global_weight, FLAGS.train_label_ratio))

    print("Model restored.")



    #DGI debug
    #logits, embeddings, corp_embeddings = sess.run([model.logits, model.embeddings, model.corp_embeddings], feed_dict=feed_dict)
    #embeddings = np.concatenate((embeddings, corp_embeddings), axis=0)
    #mean_e = np.mean(embeddings, axis=1)
    #print("mean of dimension" + str(np.mean(embeddings)))

    #f_dim = embeddings.shape[1]
    #node_pad_id = embeddings.shape[0]
    #padding = np.zeros(shape=[1, f_dim])
    #embeddings = np.concatenate((embeddings, padding), axis=0)


    #_, labels = gdata.get_node_classification_data(name='all')
    #labels = np.argmax(labels, axis=1)

    #logits = np.squeeze(logits)
    #sorted_idx = np.argsort(logits).tolist()

    #top_idx = list(reversed(sorted_idx[-500:]))
    #bottom_idx = sorted_idx[:50]

    #print(logits[top_idx[:10]])
    #print(logits[bottom_idx[:10]])

    #idx_label_pair = []
    #label_count = {}
    #for idx in top_idx:
    #    if idx >= num_nodes:
    #        continue
    #    l = labels[idx]
    #    idx_label_pair.append((idx, l))
    #    if l not in label_count:
    #        label_count[l] = 0
    #    label_count[l] += 1

    #print("label count" + str(label_count))
    #idx_label_pair = sorted(idx_label_pair, key=lambda x:x[1])
    #sorted_idx = []
    #last_label = idx_label_pair[0][1]
    #sorted_idx.append(idx_label_pair[0][0])
    #sorted_idx.append(node_pad_id)
    #for pair in idx_label_pair:
    #    l = pair[1]
    #    i = pair[0]
    #    sorted_idx.append(i)
    #    if not l == last_label:
    #        last_label = l
    #        sorted_idx.append(node_pad_id)
    #sorted_idx.append(node_pad_id)

    #idx = sorted_idx + bottom_idx

    #embeddings = embeddings[idx]

    #header = "%d %d"%(1, 1)
    #dimension = FLAGS.hidden1
    #if not os.path.exists('emb/%s'%FLAGS.dataset):
    #    os.makedirs('emb/%s'%FLAGS.dataset)
    #femb = 'emb/%s/%s-%d-debug.txt'%(FLAGS.dataset, FLAGS.model, dimension)
    #utils.save_embedding(femb, embeddings, delimiter=' ', header=header)



    #debug Kmeans classify
    #kmeans_pred = KMeans(n_clusters=FLAGS.DGI_n_cluster, random_state=0).fit_predict(embeddings)

    #trainX_indices, trainY = gdata.get_node_classification_data(name='train')
    #valX_indices, valY = gdata.get_node_classification_data(name='val')
    #testX_indices, testY = gdata.get_node_classification_data(name='test')

    #clusted_nodes = []
    #for i in range(FLAGS.DGI_n_cluster):
    #    clusted_nodes.append([])
    #for i, l in enumerate(kmeans_pred):
    #    clusted_nodes[l].append(i)

    #pred_y = np.zeros(shape=(num_nodes))

    #def max_label(nodes, Y, label_num, filters):
    #    f = {}
    #    for n in filters:
    #        f[n] = 1

    #    label_count = []
    #    for i in range(FLAGS.DGI_n_cluster):
    #        label_count.append(0)
    #    for n in nodes:
    #        if n not in f:
    #            continue
    #        l = np.argmax(Y[n])
    #        label_count[l] += 1
    #    return np.argmax(label_count)

    #for nodes in clusted_nodes:
    #    l = max_label(nodes, trainY, FLAGS.DGI_n_cluster, trainX_indices)
    #    pred_y[nodes] = l

    #val_result = {}
    #val_result['micro'] = f1_score(np.argmax(valY, axis=1), pred_y[valX_indices], average='micro')
    #val_result['macro'] = f1_score(np.argmax(valY, axis=1), pred_y[valX_indices], average='macro')
    #print("val result of %s-%d: micro F1 %.5f, macro F1 %.5f"%(FLAGS.model, FLAGS.hidden1, val_result['micro'], val_result['macro']))

    #result = {}
    #result['micro'] = f1_score(np.argmax(testY, axis=1), pred_y[testX_indices], average='micro')
    #result['macro'] = f1_score(np.argmax(testY, axis=1), pred_y[testX_indices], average='macro')
    #print("test result of %s-%d: micro F1 %.5f, macro F1 %.5f"%(FLAGS.model, FLAGS.hidden1, result['micro'], result['macro']))




    feed_dict.update({placeholders['dropout']:0.})
    feed_dict.update({placeholders['in_drop']:0.})
    feed_dict.update({placeholders['coef_drop']:0.})
    embeddings = sess.run(model.embeddings, feed_dict=feed_dict)


    return embeddings


def clf_function(embeddings, gdata):
    trainX_indices, trainY = gdata.get_node_classification_data(name='train')
    valX_indices, valY = gdata.get_node_classification_data(name='val')
    testX_indices, testY = gdata.get_node_classification_data(name='test')

    if FLAGS.sample_node_label and valX_indices:
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


    if not FLAGS.clf_minibatch:
    #whole data version
        #clf = Classifier('SVM', class_num)
        clf = Classifier('LR', class_num)
        clf.train(trainX, trainY)
        if not valX_indices is None:
            val_result = clf.evaluate(valX, valY)
            print("val result of %s-%d: micro F1 %.5f, macro F1 %.5f"%(FLAGS.model, FLAGS.hidden1, val_result['micro'], val_result['macro']))
        else:
            val_result = {}
            val_result['micro'], val_result['macro'] = 0.0, 0.0
        result = clf.evaluate(testX, testY)
        print("test result of %s-%d: micro F1 %.5f, macro F1 %.5f"%(FLAGS.model, FLAGS.hidden1, result['micro'], result['macro']))
    else:
        #mini batch version
        clf = SGDClassifier(loss="hinge", penalty="l2")
        if FLAGS.dataset == 'ppi':
            #clf = SGDClassifier(loss="modified_huber", penalty="l2")
            clf = SGDClassifier(loss="log", penalty="l2")
            clf = OneVsRestClassifier(clf)
        train_idx = list(range(np.shape(trainY)[0]))
        batch_num = 10
        batch_size = int(len(train_idx) / batch_num)
        print("start to train classifier, batch size %d, num of batch %d"%(batch_size, batch_num))
        clf_repeat = 10
        for k in range(clf_repeat):
            train_idx = shuffle(train_idx)
            for i in range(batch_num):
                lid = i*batch_size
                if i == batch_num - 1:
                    rid = len(train_idx)
                else:
                    rid = lid + batch_size
                batchX = trainX[train_idx[lid:rid]]
                batchY = np.argmax(trainY[train_idx[lid:rid]], axis=1)
                clf.partial_fit(batchX, batchY, classes=list(range(class_num)))
                print("epoch %d, batch %d end"%(k, i))

        if valX:
            val_result = {}
            #val_predY = clf.predict_proba(valX) > 0.5
            #val_result['micro'] = f1_score(valY, val_predY, average='micro')
            #val_result['macro'] = f1_score(valY, val_predY, average='macro')
            val_predY = clf.predict(valX)
            val_result['micro'] = f1_score(np.argmax(valY, axis=1), val_predY, average='micro')
            val_result['macro'] = f1_score(np.argmax(valY, axis=1), val_predY, average='macro')
            print("val result of %s-%d: micro F1 %.5f, macro F1 %.5f"%(FLAGS.model, FLAGS.hidden1, val_result['micro'], val_result['macro']))
        result = {}
        #test_predY = clf.predict_proba(testX) > 0.5
        #result['micro'] = f1_score(testY, test_predY, average='micro')
        #result['macro'] = f1_score(testY, test_predY, average='macro')
        test_predY = clf.predict(testX)
        result['micro'] = f1_score(np.argmax(testY, axis=1), test_predY, average='micro')
        result['macro'] = f1_score(np.argmax(testY, axis=1), test_predY, average='macro')
        print("test result of %s-%d: micro F1 %.5f, macro F1 %.5f"%(FLAGS.model, FLAGS.hidden1, result['micro'], result['macro']))

        #print(test_predY)

    return val_result['micro'], val_result['macro'], result['micro'], result['macro']


def main():
    gdata = Graphdata()
    if FLAGS.dataset in ['cora', 'citeseer', 'pubmed', 'wiki']:
        if FLAGS.sample_node_label:
            gdata.load_gcn_data(FLAGS.dataset, fix_node_test=False, node_label=True)
            gdata.random_split_train_test(node_split=True, node_train_ratio=FLAGS.train_label_ratio, node_test_ratio=FLAGS.test_label_ratio, set_label_dis=FLAGS.set_label_dis)

        else:
            gdata.load_gcn_data(FLAGS.dataset, fix_node_test=True)
    elif FLAGS.dataset in ['reddit', 'ppi']:
        gdata = InductiveData('data', FLAGS.dataset)
    else:
        print("no such dataset")
        sys.exit(0)
        pass


    #load data label test
    #features = gdata.get_features()
    #trainX_indices, trainY = gdata.get_node_classification_data(name='train')
    #valX_indices, valY = gdata.get_node_classification_data(name='val')
    #testX_indices, testY = gdata.get_node_classification_data(name='test')

    #print(trainX_indices)
    #print(trainY)
    #print(trainY[:10].tolist())
    #sys.exit()

    show_hyper_parameters()

    if FLAGS.lp:
        gdata.random_split_train_test(edge_split=True, edge_train_ratio=0.8, edge_test_ratio=0.15)

    if FLAGS.embedding:
        if FLAGS.model in ['MaskedGraphAE', 'GraphAE', 'GraphBert']:
            embeddings = train_embedding_graphAE(gdata)
        elif FLAGS.model in ['DGI', 'ClusterDGI', 'PCDGI', 'EgoDGI', 'NewClusterDGI', 'CommunityDGI']:
            if FLAGS.minibatch_flag:
                embeddings = train_embedding_MinibatchDGI(gdata)
            else:
                embeddings = training_embedding_DGI(gdata)
        elif FLAGS.model in ['PPRDGI']:
            embeddings = training_embedding_PPRDGI(gdata)
        else:
            embeddings = train_embedding(gdata)
        node_num = len(gdata.get_nodes('train'))
        dimension = FLAGS.hidden1
        header = "%d %d"%(node_num, dimension)
        if not os.path.exists('emb/%s'%FLAGS.dataset):
            os.makedirs('emb/%s'%FLAGS.dataset)
        femb = 'emb/%s/%s-%d.txt'%(FLAGS.dataset, FLAGS.model, dimension)
        if embeddings is None:
            print("embedding load error")
        else:
            utils.save_embedding(femb, embeddings, delimiter=' ', header=header)
            #utils.save_embedding(femb, embeddings, delimiter=',', header=header)
    else:
        femb = 'emb/%s/%s-%d.txt'%(FLAGS.dataset, FLAGS.model, FLAGS.hidden1)
        embeddings = utils.load_embedding(femb, delimiter=' ')
        print("load embedding end, emb shape " + str(embeddings.shape))

    if embeddings is None:
        return 0.0, 0.0

    if FLAGS.visual:
        visualizer = Visualizer('tsne', 10)

    if FLAGS.cluster:
        cluster = Cluster('KMeans', 10)
        cluster.fit(embeddings)
        pred_y = cluster.predict()

    if FLAGS.lp:
        test_pos_edges, test_neg_edges = gdata.get_link_prediction_data(data_name='test')
        roc_score, ap_score = utils.get_roc_score(test_pos_edges, test_neg_edges, embeddings)
        print("Test roc scoreL:%.5f, average precision:%.5f"%(roc_score, ap_score))
        return None, None, roc_score, ap_score

    if FLAGS.clf:

        return clf_function(embeddings, gdata)


        trainX_indices, trainY = gdata.get_node_classification_data(name='train')
        valX_indices, valY = gdata.get_node_classification_data(name='val')
        testX_indices, testY = gdata.get_node_classification_data(name='test')

        if FLAGS.sample_node_label and valX_indices:
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


        if not FLAGS.clf_minibatch:
        #whole data version
            #clf = Classifier('SVM', class_num)
            clf = Classifier('LR', class_num)
            clf.train(trainX, trainY)
            if not valX_indices is None:
                val_result = clf.evaluate(valX, valY)
                print("val result of %s-%d: micro F1 %.5f, macro F1 %.5f"%(FLAGS.model, FLAGS.hidden1, val_result['micro'], val_result['macro']))
            else:
                val_result = {}
                val_result['micro'], val_result['macro'] = 0.0, 0.0
            result = clf.evaluate(testX, testY)
            print("test result of %s-%d: micro F1 %.5f, macro F1 %.5f"%(FLAGS.model, FLAGS.hidden1, result['micro'], result['macro']))
        else:
            #mini batch version
            clf = SGDClassifier(loss="hinge", penalty="l2")
            if FLAGS.dataset == 'ppi':
                #clf = SGDClassifier(loss="modified_huber", penalty="l2")
                clf = SGDClassifier(loss="log", penalty="l2")
                clf = OneVsRestClassifier(clf)
            train_idx = list(range(np.shape(trainY)[0]))
            batch_num = 10
            batch_size = int(len(train_idx) / batch_num)
            print("start to train classifier, batch size %d, num of batch %d"%(batch_size, batch_num))
            clf_repeat = 10
            for k in range(clf_repeat):
                train_idx = shuffle(train_idx)
                for i in range(batch_num):
                    lid = i*batch_size
                    if i == batch_num - 1:
                        rid = len(train_idx)
                    else:
                        rid = lid + batch_size
                    batchX = trainX[train_idx[lid:rid]]
                    batchY = np.argmax(trainY[train_idx[lid:rid]], axis=1)
                    clf.partial_fit(batchX, batchY, classes=list(range(class_num)))
                    print("epoch %d, batch %d end"%(k, i))

            if valX:
                val_result = {}
                #val_predY = clf.predict_proba(valX) > 0.5
                #val_result['micro'] = f1_score(valY, val_predY, average='micro')
                #val_result['macro'] = f1_score(valY, val_predY, average='macro')
                val_predY = clf.predict(valX)
                val_result['micro'] = f1_score(np.argmax(valY, axis=1), val_predY, average='micro')
                val_result['macro'] = f1_score(np.argmax(valY, axis=1), val_predY, average='macro')
                print("val result of %s-%d: micro F1 %.5f, macro F1 %.5f"%(FLAGS.model, FLAGS.hidden1, val_result['micro'], val_result['macro']))
            result = {}
            #test_predY = clf.predict_proba(testX) > 0.5
            #result['micro'] = f1_score(testY, test_predY, average='micro')
            #result['macro'] = f1_score(testY, test_predY, average='macro')
            test_predY = clf.predict(testX)
            result['micro'] = f1_score(np.argmax(testY, axis=1), test_predY, average='micro')
            result['macro'] = f1_score(np.argmax(testY, axis=1), test_predY, average='macro')
            print("test result of %s-%d: micro F1 %.5f, macro F1 %.5f"%(FLAGS.model, FLAGS.hidden1, result['micro'], result['macro']))

            #print(test_predY)

        return val_result['micro'], val_result['macro'], result['micro'], result['macro']

def multi_run():
    micro_f1 = []
    macro_f1 = []
    k = 1
    for i in range(k):
        #with tf.variable_scope("main", reuse=tf.AUTO_REUSE):
        with tf.variable_scope("main_%d"%i):
            _, _, mi, ma = main()
        micro_f1.append(mi)
        macro_f1.append(ma)
    return np.mean(micro_f1), np.mean(macro_f1), np.std(micro_f1), np.std(macro_f1)

def test():
    return 1.0, 1.0

if __name__ == '__main__':
    with RedirectStdStreams(stdout=sys.stderr):
        start_time = time.time()
        #mi_f1, ma_f1, _, _ = main()
        mi_f1, ma_f1, mi_f1_std, ma_f1_std = multi_run()
        if FLAGS.lp:
            print("mean auc %.5f, std %.5f, ap %.5f, std %.5f"%(mi_f1, mi_f1_std, ma_f1, ma_f1_std))
        else:
            print("mean micro f1 %.5f, std %.5f, macro f1 %.5f, std %.5f"%(mi_f1, mi_f1_std, ma_f1, ma_f1_std))
        print("train and evaluation this model totally costs %f seconds"%(time.time()-start_time))
    print('(%.4f, %.4f)' %(mi_f1, ma_f1))
