import tensorflow as tf
import numpy as np
import scipy.sparse as sp

import time
import os
import shutil
import sys

import math
from model_backup import GaussianContext2Vec, DisentanglingContext2Vec, DisentanglingNeighborGraphAE, HierarchicalRank2Vec
from model_backup import MNMF, LF_NMF, ANMF, AHNMF, ANMFNEW, LF_ANMF
from model_backup import LinearREmb

from models import GraphAE, GraphVAE, MiniBatchGraphAE
from models import DGI, ClusterDGI
from tasks import Classifier, Cluster, LinkPredictor, Visualizer
import utils
from utils import Graphdata, NeiSampler, HierRankData, InductiveData, ClusterMiniBatchGenerator
from initials import glorot, zeros

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.cluster import SpectralClustering
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle
from sklearn.metrics import f1_score

import networkx.algorithms.community.label_propagation as label_prop
import networkx.algorithms.community.modularity_max as modularity
import networkx.algorithms.components.connected as components
from persona import PersonaOverlappingClustering

from scipy.sparse.linalg import eigs

from sklearn.cluster import KMeans


from sklearn.decomposition import NMF

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


flags.DEFINE_float('l2_weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('s_weight_decay', 1., 'Weight for structural loss on embedding matrix.')
flags.DEFINE_float('c_weight_decay', 1., 'Weight for content reconstruction loss on embedding matrix.')
flags.DEFINE_float('entropy_weight_decay', 1., 'Weight for entropy_weight_decay loss on embedding matrix.')
flags.DEFINE_float("tc_weight_decay", 1e-5, 'weight for tc term in loss')


flags.DEFINE_string('re_con_type', 'clf', 'loss type of content reconstruction loss: clf(weighted cross entropy) or reg(mean square loss)')
flags.DEFINE_string('multi_loss_type', 'weight_sum', 'multi tasks loss aggregation type: weighted sum(weight_sum) or learn from parameters(gaussian)')

flags.DEFINE_float('tc_lr', 0.01, 'learning rate of tc discriminator')
flags.DEFINE_boolean('tc_term_flag', False, 'flag of adding tc disentangle regularization')

flags.DEFINE_integer("neg_num", 5, 'number of negative context')
flags.DEFINE_integer("max_degree", 10, 'set max size of context')

flags.DEFINE_integer("hidden1", 32, 'dimension of first hidden layer')
flags.DEFINE_integer("hidden2", 16, 'dimension of second hidden layer')
flags.DEFINE_integer("n_layer", 2, 'num of neighbor disentangle aggregation layer')

flags.DEFINE_integer("n_cluster", 3, 'cluster of differ neighbors')


flags.DEFINE_string('logdir', 'log/', 'dir for save logging information')
flags.DEFINE_integer('log', 0, 'flag control whether to log')

flags.DEFINE_integer('nb_size', 10, 'number of neighbors sampled for each node')
flags.DEFINE_boolean('include_self', False, 'if neighbors include self')

flags.DEFINE_boolean('embedding', False, 'retrain node representation')
flags.DEFINE_boolean('clf', False, 'use node classification as downstream tasks')
flags.DEFINE_boolean('lp', False, 'use link prediction as downstream tasks')
flags.DEFINE_boolean('cluster', False, 'use node classification as downstream tasks')
flags.DEFINE_boolean('visual', False, 'use node classification as downstream tasks')
flags.DEFINE_integer('verbose', 5, 'show clf result every K epochs')

flags.DEFINE_boolean('hirank_encoder', False, 'whether get embedding from features')
flags.DEFINE_float('hinge_gamma', 0.8, 'margin used in hinge loss')
#flags.DEFINE_integer('triple_sample_num', 5, 'num of node sampling at each distance level')
flags.DEFINE_boolean('negative_triple_flag', False, 'whether add unconnect node pair')
flags.DEFINE_integer("negative_triple_num", 5, 'number of negative triple samples')
flags.DEFINE_enum('hi_structure_loss_type', 'hinge', ['hinge', 'energy'], 'loss type of strucuture reconstruction in HierarchicalRank2Vec model,[hinge, energy]')

flags.DEFINE_integer("community_num", 7, 'number of community')
flags.DEFINE_float('NMF_alpha', 1.0, 'weight of community indictor reconstruction loss')
flags.DEFINE_float("NMF_beta", 1.0, 'weight of max modularity loss')
flags.DEFINE_float("NMF_eta", 5.0, 'weight of second order proximity')
flags.DEFINE_float("NMF_sigma", 1.0, 'weight of feature reconstruction')
flags.DEFINE_float("NMF_lambda", 1e9, 'weight of bias of single community indictor')
flags.DEFINE_float("NMF_gamma", 1.0, 'weight of low filter constraints')
flags.DEFINE_float("lower_control", 1e-12, 'lower bound of very small number')

flags.DEFINE_integer("ANMF_s_community_num", 10, 'number of strucuture based community')
flags.DEFINE_integer("ANMF_c_community_num", 7, 'number of content based community')
flags.DEFINE_float('ANMF_alpha', 1.0, 'weight of content community indictor reconstruction loss')
flags.DEFINE_float("ANMF_beta", 1.0, 'weight of structure community indictor reconstruction loss')
flags.DEFINE_float("ANMF_phi", 1.0, 'weight of structure reconstruction loss')
flags.DEFINE_float("ANMF_gamma", 1.0, 'weight of feature reconstruction loss')
flags.DEFINE_float("ANMF_eta", 5.0, 'weight of second order proximity')
flags.DEFINE_float("ANMF_theta", 1.0, 'weight of low filter loss')
flags.DEFINE_float("ANMF_rho", 1.0, 'weight of modularity loss')
flags.DEFINE_boolean("ANMF_nonzero", False, 'flag of nonzero constraint')

flags.DEFINE_boolean("kmeans_init", False, 'flag of using kmeans init')

flags.DEFINE_float('R_alpha', 1.0, 'weight of link reconstruction loss')
flags.DEFINE_float("R_beta", 1.0, 'weight of structure reconstruction loss')
flags.DEFINE_float("R_gamma", 1.0, 'weight of feature reconstruction loss')
flags.DEFINE_float("R_gate_emb_size", 8, 'dim of gate matrix factorization')

#DGI model flags
flags.DEFINE_boolean("corp_flag_x", False, 'corrup feature matrix')
flags.DEFINE_boolean("corp_flag_adj", False, 'corrup adj matrix')
flags.DEFINE_integer("DGI_n_cluster", 50, 'number of pre-clusters')
flags.DEFINE_float("global_weight", 0.5, 'weight of global MI')
flags.DEFINE_integer("clusters_per_batch", 3, 'clusters used in each batch')
flags.DEFINE_boolean("minibatch_flag", False, 'flag of whether using batch version of clusterDGI')



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
    print('entropy_coef: ' + str(FLAGS.entropy_weight_decay))
    print('struct_coef: ' + str(FLAGS.s_weight_decay))
    print('content_coef: ' + str(FLAGS.c_weight_decay))
    print('tc_term_coef: ' + str(FLAGS.tc_weight_decay))
    print('dropout: ' + str(FLAGS.dropout))
    print('in_dropout: ' + str(FLAGS.in_drop))
    print('coef_dropout: ' + str(FLAGS.coef_drop))
    print('features: ' + str(FLAGS.features))
    print('epochs: ' + str(FLAGS.epochs))
    print('early_stop: ' + str(FLAGS.early_stop))
    print('batch_size: ' + str(FLAGS.batch_size))
    print('negative number: ' + str(FLAGS.neg_num))
    print('max neighbors of context: ' + str(FLAGS.max_degree))
    print("multi tasks loss type: " + str(FLAGS.multi_loss_type))
    print("tc term learning rate: " + str(FLAGS.tc_lr))
    print("tc term flag: " + str(FLAGS.tc_term_flag))
    print("num of different disentangle factor: " + str(FLAGS.n_cluster))
    print("content reconstruction func: " + str(FLAGS.re_con_type))
    print("global cluster function: " + str(FLAGS.global_clustering_fn))
    print("local clustering function: " + str(FLAGS.local_clustering_fn))
    print("number of sampled neighbors: " + str(FLAGS.nb_size))
    print("min size of community: " + str(FLAGS.min_component_size))
    print("include self in neighbor sampling: " + str(FLAGS.include_self))
    print("hirank2vec encoder flag: "+str(FLAGS.hirank_encoder))
    print("hinge gamma: "+str(FLAGS.hinge_gamma))
    #print("triple_sample_num: "+str(FLAGS.triple_sample_num))
    print("number of sampled neighbors: "+str(FLAGS.nb_size))
    print("whether add unconnect node pair: "+str(FLAGS.negative_triple_flag))
    print("number of negative triple samples: "+str(FLAGS.negative_triple_num))
    print('----- Archi. hyperparams -----')
    print('hidden1 dimension size: ' + str(FLAGS.hidden1))
    print('hidden2 dimension size: ' + str(FLAGS.hidden2))
    print("num of disentangle neighbors agg layer: " + str(FLAGS.n_layer))



def init_hyper_parameters(gdata):
    config = {}
    #set max degree manually or exactly calculate from input graph
    #max_degree = gdata.get_node_degree().max()
    max_degree = FLAGS.max_degree
    config['max_degree'] = max_degree
    config['learning_rate'] = FLAGS.learning_rate
    config['neg_num'] = FLAGS.neg_num
    config['l2_weight_decay'] = FLAGS.l2_weight_decay
    config['hidden1'] = FLAGS.hidden1

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
    #init neighbor sampler
    if FLAGS.model in ['GraphAE', 'GraphVAE']:
        pass
    else:
        nb_size = FLAGS.nb_size
        nb_sampler = NeiSampler(trainG, nb_size, FLAGS.include_self)
        if FLAGS.include_self:
            nb_size += 1

    #init cluster
    if FLAGS.model == 'HiRank2Vec':
        overlapping_clustering, _, _ = PersonaOverlappingClustering(trainG, _CLUSTERING_FN[FLAGS.local_clustering_fn],
                                _CLUSTERING_FN[FLAGS.global_clustering_fn], FLAGS.min_component_size)
        cluster_mask = utils.build_cluster_mask(trainG, overlapping_clustering)
        num_of_cluster = int(cluster_mask.shape[1])
        print("num of cluster:%d"%len(overlapping_clustering))
        print("avg %.3f num of clusters each node in, node %d join %d num of clusters(most)"
        %(np.mean(np.sum(cluster_mask,1)), np.argmax(np.sum(cluster_mask,1)), np.max(np.sum(cluster_mask,1))))

#build content reconstruction weight
    features = gdata.get_features()


    if FLAGS.model == 'DNBGraphAE':
        features = utils.expand_csr_matrix(features, 0)

    f_pos_weight = float(features.shape[0] * features.shape[0] - features.sum()) / features.sum()
    f_norm = features.shape[0] * features.shape[0] / float((features.shape[0] * features.shape[0] - features.sum()) * 2)

#build structure inputs
    adj = gdata.get_adj(sperate_name)
    sta_information = gdata.get_static_information()
    num_features = sta_information[0]
    features_nonzero = sta_information[1]
    num_nodes = adj.shape[0]
    features = utils.sparse_to_tuple(features)



    gat_adj = adj+sp.identity(num_nodes)
    adj_mat = utils.sparse_to_tuple(gat_adj.tocoo())
    adj_orig = gdata.get_orig_adj(sperate_name)
    #adj_orig = adj
    #adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    #adj_orig.eliminate_zeros()
    adj_orig = utils.sparse_to_tuple(adj_orig.tocoo())
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)



    print("pos weight of structure %.5f, norm %.5f"%(pos_weight, norm))
    print("pos weight of content %.5f, norm %.5f"%(f_pos_weight, f_norm))

    #sys.exit(0)

    #build placeholders and hyper parameters
    placeholders = {
                   'features': tf.sparse_placeholder(tf.float32),
                   'adj_mat': tf.sparse_placeholder(tf.float32),
                   'adj_orig': tf.sparse_placeholder(tf.float32),
                   'dropout': tf.placeholder_with_default(0., shape=()),
                   'in_drop': tf.placeholder_with_default(0., shape=()),
                   'coef_drop': tf.placeholder_with_default(0., shape=())
           }

    feed_dict = {}
    feed_dict.update({placeholders['features']:features})
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
    config['entropy_weight_decay'] = FLAGS.entropy_weight_decay
    config['n_cluster'] = FLAGS.n_cluster
    config['hidden1'] = FLAGS.hidden1
    config['hidden2'] = FLAGS.hidden2
    config['act'] = tf.nn.elu
    config['pos_weight'] = pos_weight
    config['norm'] = norm
    config['f_pos_weight'] = f_pos_weight
    config['f_norm'] = f_norm
    config['n_heads_list'] = [8, 8, 8, 8]
    config['re_con_type'] = 'clf' #clf or recon
    config['multi_loss_type'] = FLAGS.multi_loss_type
    config['num_of_nodes'] = num_nodes
    config['tc_term_flag'] = FLAGS.tc_term_flag
    config['tc_lr'] = FLAGS.tc_lr
    config['tc_weight_decay'] = FLAGS.tc_weight_decay

    config['reverse_type'] = FLAGS.reverse_type


    if FLAGS.model == 'GraphAE':
        model = GraphAE(placeholders=placeholders, num_features=num_features, config=config)
    elif FLAGS.model == 'GraphVAE':
        model = GraphVAE(placeholders=placeholders, num_features=num_features, config=config)
    elif FLAGS.model == 'DNBGraphAE':
        placeholders['neighbors_matrix'] = tf.placeholder(dtype=tf.int32, shape=[num_nodes, nb_size], name='neighbors_matrix')
        #add padding node
        placeholders['cluster_mask'] = tf.placeholder(dtype=tf.float32, shape=[num_nodes+1, num_of_cluster], name='cluster_mask')
        cluster_mask = np.vstack((cluster_mask, np.zeros(shape=(1, num_of_cluster))))
        config['nb_size'] = nb_size
        feed_dict.update({placeholders['cluster_mask']:cluster_mask})
        config['hidden1'] = FLAGS.hidden1 * num_of_cluster
        print("new hidden dim %d"%config['hidden1'])
        config['n_layer'] = FLAGS.n_layer
        model = DisentanglingNeighborGraphAE(placeholders=placeholders, num_features=num_features, node_num=num_nodes, num_of_cluster=num_of_cluster, config=config)
    elif FLAGS.model == 'DC2V':
        model = DisentanglingContext2Vec(placeholders=placeholders, num_features=num_features, config=config)
    else:
        print("No such model!!")
        sys.exit(-1)

#init training
    sess = tf.Session()
    sess.run(model.init_op)

    train_loss_lastK = []



    for k in range(FLAGS.epochs):
        feed_dict.update({placeholders['dropout']:FLAGS.dropout})
        feed_dict.update({placeholders['in_drop']:FLAGS.in_drop})
        feed_dict.update({placeholders['coef_drop']:FLAGS.coef_drop})
        if FLAGS.model == "DNBGraphAE":
            neighbors_matrix = nb_sampler.sample()
            #print(neighbors_matrix[1,:2])
            feed_dict.update({placeholders['neighbors_matrix']:neighbors_matrix})
    #debug
        #embeddings, p, debug_emb, inputs = sess.run([model.embeddings, model.p, model.debug_emb, model.inputs], feed_dict=feed_dict)
        #print(embeddings)
        #print(debug_emb)
        #print(inputs)
        #print(p)
        #update parameters
        _ = sess.run(model.opt_op, feed_dict=feed_dict)

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
        if FLAGS.model == 'GraphVAE':
            kl_loss = sess.run(model.kl_loss, feed_dict=feed_dict)
            print("Epoch %d: loss %.5f, structural loss %.5f, content recon loss %.5f, l2 %.5f, kl loss %.5f"%(k, loss, s_loss, c_loss, l2_loss, kl_loss))
        elif FLAGS.model == 'DC2V':
            e_loss = sess.run(model.entropy_loss, feed_dict=feed_dict)
            print("Epoch %d: loss %.5f, structural loss %.5f, content recon loss %.5f, l2 %.5f, entropy loss %.5f"%(k, loss, s_loss, c_loss, l2_loss, e_loss))
        else:
            print("Epoch %d: loss %.5f, structural loss %.5f, content recon loss %.5f, l2 %.5f"%(k, loss, s_loss, c_loss, l2_loss))


        #train_loss_lastK.append(loss)

        if FLAGS.tc_term_flag:
            T = 1
            for i in range(T):
                tc_loss, tc_dis_loss = sess.run([model.tc_loss, model.tc_dis_loss], feed_dict=feed_dict)
                print("tc loss  %.5f, tc discriminator loss %.5f"%(tc_loss, tc_dis_loss))
                sess.run(model.tc_opt_op, feed_dict=feed_dict)

        if k > FLAGS.early_stop:
            avg_loss_last_k = np.mean(train_loss_lastK[-FLAGS.early_stop:])
            if avg_loss_last_k < loss:
                break
        train_loss_lastK.append(loss)

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

    feed_dict.update({placeholders['dropout']:0.})
    feed_dict.update({placeholders['in_drop']:0.})
    feed_dict.update({placeholders['coef_drop']:0.})
    if FLAGS.model == "DNBGraphAE":
        #neighbors_matrix = nb_sampler.sample()
        neighbors_matrix = nb_sampler.sample()
        feed_dict.update({placeholders['neighbors_matrix']:neighbors_matrix})
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

        if k > FLAGS.early_stop:
            avg_loss_last_k = np.mean(train_loss_lastK[-FLAGS.early_stop:])
            if avg_loss_last_k < epoch_metric_loss:
                break
        train_loss_lastK.append(epoch_metric_loss)

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
    num_data, train_adj, adj, feats, train_feats, test_feats, labels, train_data, val_data, test_data = gdata.get_data()
    print("adj shape (%d, %d)"%(adj.shape[0], adj.shape[1]))
    print("train adj shape (%d, %d)"%(train_adj.shape[0], train_adj.shape[1]))

    #
    num_features = feats.shape[1]

    if FLAGS.model == 'ClusterDGI':
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
    config['hidden1'] = FLAGS.hidden1
    config['hidden2'] = FLAGS.hidden2
    config['act'] = tf.nn.relu
    config['node_num'] = num_data
    config['feature_dim'] = num_features
    config['global_weight'] = FLAGS.global_weight
    config['act'] = tf.keras.layers.PReLU
    config['feature_dim'] = num_features

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
            save_path = saver.save(sess, "tmp/model_%s_%s_%d_%d.ckpt"%(FLAGS.model, FLAGS.dataset, FLAGS.hidden1, FLAGS.DGI_n_cluster))
            #print("Model saved in path: %s" % save_path)
        else:
            wait += 1

        if wait >= FLAGS.early_stop:
            break

    saver.restore(sess, "tmp/model_%s_%s_%d_%d.ckpt"%(FLAGS.model, FLAGS.dataset, FLAGS.hidden1, FLAGS.DGI_n_cluster))
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
    embeddings = np.zeros(shape=(adj.shape[0], FLAGS.hidden1))

    repeat = 5
    for i in range(repeat):
        test_cmbg.refresh()
        while not test_cmbg.end:
            print("debug 1")
            batch_adj, batch_features, batch_node2cluster, batch_nodes = test_cmbg.generate()
            feed_features = utils.preprocess_features(batch_features)
            feed_adj = utils.preprocess_graph(batch_adj)
            feed_dict.update({placeholders['inputs']:feed_features})
            feed_dict.update({placeholders['adj']:feed_adj})

            embeds = sess.run(model.embeddings, feed_dict=feed_dict)
            embeddings[batch_nodes] += embeds

    embeddings = embeddings / float(repeat)
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

    if FLAGS.model == 'ClusterDGI':

        #spectral cluster
        #pre_cluster = SpectralClustering(n_clusters=FLAGS.DGI_n_cluster, assign_labels='discretize', affinity='precomputed')
        #pre_cluster.fit(adj)
        #build label matrix Y
        #cluster_label_list = pre_cluster.labels_
        #print(cluster_labels)

        #metis
        (edgecuts, cluster_label_list) = metis.part_graph(nx.from_scipy_sparse_matrix(adj), FLAGS.DGI_n_cluster)
        cluster_label_list = np.array(cluster_label_list)


        cluster_labels = utils.binarize_label(cluster_label_list).astype(np.float64)

        #record node cluster label distribute used for build subgraph
        cluster = {}
        for inx, label in enumerate(cluster_label_list):
            if label not in cluster:
                cluster[label] = []
            cluster[label].append(inx)


        sub_adj_list = []
        for i in range(FLAGS.DGI_n_cluster):
            sub_adj_list.append(adj[cluster[i],:][:,cluster[i]])

        #eigen decomposition
        eigen_value_num = 10
        normalized = True
        #normalized = False
        for k in cluster:
            eigen_value_num = min(len(cluster[k]), eigen_value_num)

        select_eigen_value = int(eigen_value_num/2)
        print("min cluster size %d, select eigen value %d"%(eigen_value_num, select_eigen_value))

        eigen_vectors = []
        for i in range(FLAGS.DGI_n_cluster):
            L = utils.laplacian_adj(sub_adj_list[i], normalized)
            e, ev = eigs(A=L, k=select_eigen_value, which='SM')
            eigen_vectors.append(ev)

        #build feature pooling operator
        pooling_matrices = [sp.lil_matrix((num_nodes, FLAGS.DGI_n_cluster)) for i in range(select_eigen_value)]
        for i in range(select_eigen_value):
            for j in range(FLAGS.DGI_n_cluster):
                pooling_matrices[i][cluster[j],j] = eigen_vectors[j][:,i].reshape(-1,1)


    config = {}
    config['learning_rate'] = FLAGS.learning_rate
    config['hidden1'] = FLAGS.hidden1
    config['hidden2'] = FLAGS.hidden2
    #config['act'] = tf.nn.relu
    config['act'] = tf.keras.layers.PReLU
    config['node_num'] = num_nodes
    config['feature_dim'] = num_features
    config['global_weight'] = FLAGS.global_weight

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

    ones = np.ones(shape=[num_nodes, 1], dtype=np.float32)
    zeros = np.zeros(shape=[num_nodes, 1], dtype=np.float32)
    labels = np.concatenate((ones, zeros), axis=0)
    feed_dict.update({placeholders['labels']:labels})


    if FLAGS.model == 'DGI':
        model = DGI(placeholders=placeholders, config=config)
    else:
        placeholders['cluster_labels'] = tf.placeholder(shape=[None, FLAGS.DGI_n_cluster], dtype=tf.float32)
        feed_dict.update({placeholders['cluster_labels']:cluster_labels})
        config['select_eigen_value'] = select_eigen_value

        for i in range(select_eigen_value):
            placeholders['eigen_vector_%d'%i] = tf.sparse_placeholder(tf.float32)
            feed_dict.update({placeholders['eigen_vector_%d'%i]:utils.sparse_to_tuple(pooling_matrices[i])})

        model = ClusterDGI(placeholders=placeholders, config=config)


    sess = tf.Session()
    sess.run(model.init_op)

    best = np.inf
    wait = 0
    saver = tf.train.Saver()

    re_corp_step = 1


    for i in range(FLAGS.epochs):

        if i % re_corp_step == 0:
            if FLAGS.corp_flag_x:
                corp_inputs = utils.corruption_function(features)
            else:
                corp_inputs = features

            if FLAGS.corp_flag_adj:
                corp_adj = corruption_function(adj)
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


        if FLAGS.model == 'ClusterDGI':
            loss, local_loss, global_loss, _ = sess.run([model.loss, model.local_loss, model.global_loss, model.opt_op], feed_dict=feed_dict)
            print("Epoch %d, loss %.10f, local %.8f, global %.8f"%(i, loss, local_loss, global_loss))
        else:
            loss, _ = sess.run([model.loss, model.opt_op], feed_dict=feed_dict)
            print("Epoch %d, loss %.10f"%(i, loss))
        #logits, labels = sess.run([model.logits, model.labels], feed_dict=feed_dict)
        #print(labels)

        if loss < best:
            best = loss
            wait = 0
            save_path = saver.save(sess, "tmp/model_%s_%s_%d_%d.ckpt"%(FLAGS.model, FLAGS.dataset, FLAGS.hidden1, FLAGS.DGI_n_cluster))
            #print("Model saved in path: %s" % save_path)
        else:
            wait += 1

        if wait >= FLAGS.early_stop:
            break
    saver.restore(sess, "tmp/model_%s_%s_%d_%d.ckpt"%(FLAGS.model, FLAGS.dataset, FLAGS.hidden1, FLAGS.DGI_n_cluster))
    print("Model restored.")
    feed_dict.update({placeholders['dropout']:0.})
    embeddings = sess.run(model.embeddings, feed_dict=feed_dict)
    return embeddings








def training_embedding_hirank2vec(gdata):
    sperate_name = 'all'
    if FLAGS.lp:
        val_pos_edges, val_neg_edges = gdata.get_link_prediction_data(data_name='val')
        sperate_name = 'train'



    trainG = gdata.get_graph(sperate_name)
    #init neighbor sampler
    nb_size = FLAGS.nb_size
    nb_sampler = NeiSampler(trainG, nb_size, FLAGS.include_self)

    #build content reconstruction weight
    features = gdata.get_features()
    f_pos_weight = float(features.shape[0] * features.shape[0] - features.sum()) / features.sum()
    f_norm = features.shape[0] * features.shape[0] / float((features.shape[0] * features.shape[0] - features.sum()) * 2)
    #padding zero features for padding node
    #features = utils.expand_csr_matrix(features, 0)

    sta_information = gdata.get_static_information()
    num_features = sta_information[0]
    features_nonzero = sta_information[1]
    features = utils.sparse_to_tuple(features)
    print("pos weight of content %.5f, norm %.5f"%(f_pos_weight, f_norm))


    adj = gdata.get_adj(sperate_name)
    num_nodes = adj.shape[0]


    #build placeholders and hyper parameters
    act_sampled_nb_size = FLAGS.nb_size
    if FLAGS.include_self:
        act_sampled_nb_size = FLAGS.nb_size + 1

    neg_num = FLAGS.neg_num

    if FLAGS.model == 'HiRank2Vec':
        placeholders = {
                   'features': tf.sparse_placeholder(tf.float32),
                   'dropout': tf.placeholder_with_default(0., shape=()),
                   'neighbor_nodes': tf.placeholder(shape=[None, act_sampled_nb_size], dtype=tf.int32),
                   'triple_pair': tf.placeholder(shape=[None, 3], dtype=tf.int32),
                   'triple_weight': tf.placeholder(shape=[None], dtype=tf.float32)
           }
    elif FLAGS.model == 'MiniBatchGraphAE':
        placeholders = {
                   'features': tf.sparse_placeholder(tf.float32),
                   'dropout': tf.placeholder_with_default(0., shape=()),
                   'input_nodes1':tf.placeholder(shape=[None], dtype=tf.int32),
                   'input_nodes2':tf.placeholder(shape=[None], dtype=tf.int32),
                   'input_nb1':tf.placeholder(shape=[None, act_sampled_nb_size], dtype=tf.int32),
                   'input_nb2':tf.placeholder(shape=[None, act_sampled_nb_size], dtype=tf.int32),
                   'neg_samples':tf.placeholder(shape=[None, neg_num], dtype=tf.int32),
                   'edge_a':tf.placeholder(shape=[None], dtype=tf.int32),
                   'edge_b':tf.placeholder(shape=[None], dtype=tf.int32),
        }

    config = {}
    config['learning_rate'] = FLAGS.learning_rate
    config['l2_weight_decay'] = FLAGS.l2_weight_decay
    config['s_weight_decay'] = FLAGS.s_weight_decay
    config['c_weight_decay'] = FLAGS.c_weight_decay
    config['hidden1'] = FLAGS.hidden1
    config['hidden2'] = FLAGS.hidden2
    config['act'] = tf.nn.elu
    config['f_pos_weight'] = f_pos_weight
    config['f_norm'] = f_norm
    config['re_con_type'] = FLAGS.re_con_type #clf or recon
    config['multi_loss_type'] = FLAGS.multi_loss_type
    config['num_of_nodes'] = num_nodes
    config['encoder'] = FLAGS.hirank_encoder
    config['hinge_gamma'] = FLAGS.hinge_gamma
    config['hi_structure_loss_type'] = FLAGS.hi_structure_loss_type
    config['neg_size'] = neg_num
    config['nb_size'] = act_sampled_nb_size
    config['reverse_type'] = FLAGS.reverse_type
    config['nn_att'] = False
    #
    config['k_head'] = [8, 1]

    if FLAGS.model == 'MiniBatchGraphAE':
        model = MiniBatchGraphAE(placeholders=placeholders, num_nodes=num_nodes, num_features=num_features, config=config)
        time1 = time.time()
        nb_samper = NeiSampler(nx.from_scipy_sparse_matrix(adj), nb_size=FLAGS.nb_size, include_self=FLAGS.include_self)
        neighbor_nodes = nb_samper.sample()
        minibatch_generator = MiniBatchGenerator(nb=neighbor_nodes, neg_size=neg_num, batch_size=FLAGS.batch_size, num_nodes=num_nodes)
        print("sample neighbors costs: %.5f"%(time.time()-time1))
        resample_flag = False

    elif FLAGS.model == 'HiRank2Vec':
        model = HierarchicalRank2Vec(placeholders=placeholders, node_num=num_nodes, num_features=num_features, config=config)
        #generate neighbor nodes
        time1 = time.time()
        nb_samper = NeiSampler(nx.from_scipy_sparse_matrix(adj), nb_size=FLAGS.nb_size, include_self=FLAGS.include_self)
        neighbor_nodes = nb_samper.sample()
        print("sample neighbors costs: %.5f"%(time.time()-time1))

        #generate distance triple pair
        time1 = time.time()
        #cluster_num_list = [30, 7]#TODO better strategy
        #1.scaled by 10 each layer
        cluster_num_list = []
        last_cluster = num_nodes
        while last_cluster > 100:
            last_cluster = math.floor(last_cluster/10)
            cluster_num_list.append(last_cluster)

        hr_generator = HierRankData(cluster_num_list=cluster_num_list, adj=adj, directed=False)
        print("spectral clustering costs: %.5f"%(time.time()-time1))




    #debug
    #print(triple_pair)
    #sys.exit(0)

    def build_feed_dict(features, triple_pair, triple_weight, neighbor_nodes=None, dropout=0.):
        feed_dict = {}
        feed_dict.update({placeholders['features']:features})
        feed_dict.update({placeholders['dropout']:FLAGS.dropout})
        feed_dict.update({placeholders['triple_pair']:triple_pair})
        feed_dict.update({placeholders['neighbor_nodes']:neighbor_nodes})
        feed_dict.update({placeholders['triple_weight']:triple_weight})
        return feed_dict

    def build_minibatch_feed_dict(features, dropout, input_nodes1, input_nodes2, input_nb1, input_nb2, edge_a, edge_b, neg_samples):
        feed_dict = {}
        feed_dict.update({placeholders['features']:features})
        feed_dict.update({placeholders['dropout']:dropout})
        feed_dict.update({placeholders['input_nodes1']:input_nodes1})
        feed_dict.update({placeholders['input_nodes2']:input_nodes2})
        feed_dict.update({placeholders['input_nb1']:input_nb1})
        feed_dict.update({placeholders['input_nb2']:input_nb2})
        feed_dict.update({placeholders['edge_a']:edge_a})
        feed_dict.update({placeholders['edge_b']:edge_b})
        feed_dict.update({placeholders['neg_samples']:neg_samples})
        return feed_dict



#training embedding
    sess = tf.Session()
    sess.run(model.init_op)

    train_loss_lastK = []

    for k in range(FLAGS.epochs):
        if FLAGS.model == 'HiRank2Vec':
            time1 = time.time()
            triple_pair, triple_weight = hr_generator.GenerateTripleData()
            if FLAGS.negative_triple_flag:
                negative_triple_pair, negative_triple_weight = hr_generator.GenerateNegativeTripleData(FLAGS.negative_triple_num)
                triple_pair = triple_pair + negative_triple_pair
                triple_weight = triple_weight + negative_triple_weight
            print("generate triple samples costs: %.5f"%(time.time()-time1))
            feed_dict = build_feed_dict(features, triple_pair, triple_weight, neighbor_nodes=neighbor_nodes)
            #feed_dict.update({placeholders['dropout']:FLAGS.dropout})
        elif FLAGS.model == 'MiniBatchGraphAE':
            if resample_flag:
                time1 = time.time()
                neighbor_nodes = nb_samper.sample()
                print("sample neighbors costs: %.5f"%(time.time()-time1))
                resample_flag = False
            minibatch_generator.refresh(neighbor_nodes)
            while not minibatch_generator.end:
                cur_nodes, cur_nb_nodes, one_hop_nodes, one_hop_nb_nodes, edge_a, edge_b, neg_samples = minibatch_generator.generate()
                feed_dict = build_minibatch_feed_dict(build_minibatch_feed_dict(features=features, dropout=FLAGS.dropout,
                                                                                input_nodes1=one_hop_nodes, input_nodes2=cur_nodes,
                                                                                input_nb1=one_hop_nb_nodes, input_nb2=cur_nb_nodes,
                                                                                edge_a=edge_a, edge_b=edge_b,
                                                                                neg_samples=neg_samples))
        if k % 5 == 0:
            resample_flag = True

        _, loss, s_loss, c_loss, l2_loss = sess.run([model.opt_op, model.loss, model.s_loss, model.c_loss, model.l2_loss],
                                                      feed_dict=feed_dict)

        print("Epoch %d: loss %.5f, structural loss %.5f, content recon loss %.5f, l2 %.5f"%(k, loss, s_loss, c_loss, l2_loss))
        if k > FLAGS.early_stop:
            avg_loss_last_k = np.mean(train_loss_lastK[-FLAGS.early_stop:])
            if avg_loss_last_k < loss:
                break
        train_loss_lastK.append(loss)

    feed_dict.update({placeholders['dropout']:0.})
    embeddings = sess.run(model.embeddings, feed_dict=feed_dict)
    return embeddings

def training_embedding_NMF(gdata):
    sperate_name = 'all'
    if FLAGS.lp:
        val_pos_edges, val_neg_edges = gdata.get_link_prediction_data(data_name='val')
        sperate_name = 'train'
    if FLAGS.features:
        featureless = False
    else:
        featureless = True
    trainG = gdata.get_graph(sperate_name)

    #build content reconstruction weight
    if not featureless:
        features = gdata.get_features()
        sta_information = gdata.get_static_information()
        num_features = sta_information[0]
        features = utils.sparse_to_tuple(features)
    #f_pos_weight = float(features.shape[0] * features.shape[0] - features.sum()) / features.sum()
    #f_norm = features.shape[0] * features.shape[0] / float((features.shape[0] * features.shape[0] - features.sum()) * 2)

    #adj = gdata.get_adj(sperate_name)
    adj = gdata.get_orig_adj(sperate_name)
    num_nodes = adj.shape[0]

    #build second-order proximity matrix
    proximity_matrix = sp.coo_matrix(gdata.second_order_proximity_matrix(sperate_name))
    s_matrix = adj + FLAGS.NMF_eta * proximity_matrix



    model = NMF(n_components=4, init='random', random_state=0)
    W = model.fit_transform(s_matrix)
    H = model.components_
    print(W)
    print(H)
    loss = np.sum(np.power(s_matrix - W.dot(H), 2))
    print(loss)
    all_zero_loss =  np.sum(np.power(s_matrix, 2))
    print(all_zero_loss)
    sys.exit(0)



    #print(s_matrix)
    #sys.exit(0)


    #build modularity matrix
    modularity_matrix = gdata.get_modularity_matrix(sperate_name)

    #convert to float64
    adj.astype(np.float64)
    modularity_matrix.astype(np.float64)

    #build sparse tensor turple
    adj = utils.sparse_to_tuple(adj)
    s_matrix = utils.sparse_to_tuple(s_matrix)
    #modularity_matrix = utils.sparse_to_tuple(modularity_matrix)


    used_dtype = tf.float64
    #used_dtype = tf.float32

    if featureless:
        placeholders = {'structure_mat':tf.sparse_placeholder(used_dtype),
                    'adj_mat':tf.sparse_placeholder(used_dtype),
                    'modularity_mat':tf.placeholder(shape=[num_nodes, num_nodes], dtype=used_dtype)
                    }
    else:
        placeholders = {'structure_mat':tf.sparse_placeholder(used_dtype),
                    'adj_mat':tf.sparse_placeholder(used_dtype),
                    'modularity_mat':tf.placeholder(shape=[num_nodes, num_nodes], dtype=used_dtype),
                    'feature_mat':tf.sparse_placeholder(used_dtype)
                    }

    #placeholders = {
    #               'structure_mat': tf.placeholder(shape=[num_nodes, num_nodes], dtype=used_dtype),
    #               'adj_mat': tf.placeholder(shape=[num_nodes, num_nodes], dtype=used_dtype),
    #               'modularity_mat': tf.placeholder(shape=[num_nodes, num_nodes], dtype=used_dtype)
    #       }

    config = {}
    config['embedding_size'] = FLAGS.hidden2
    config['community_num'] = FLAGS.community_num
    config['alpha'] = FLAGS.NMF_alpha
    config['beta'] = FLAGS.NMF_beta
    config['lambda'] = FLAGS.NMF_lambda
    config['lower_control'] = FLAGS.lower_control
    config['sigma'] = FLAGS.NMF_sigma
    config['gamma'] = FLAGS.NMF_gamma

    if not featureless:
        config['feature_dim'] = num_features


    if FLAGS.model == 'LF_NMF':
        model = LF_NMF(placeholders=placeholders, node_num=num_nodes, config=config, featureless=featureless)
    else:
        model = MNMF(placeholders=placeholders, node_num=num_nodes, config=config, featureless=featureless)

    feed_dict = {}
    feed_dict.update({placeholders['structure_mat']:s_matrix})
    feed_dict.update({placeholders['adj_mat']:adj})
    feed_dict.update({placeholders['modularity_mat']:modularity_matrix})
    if not featureless:
        feed_dict.update({placeholders['feature_mat']:features})

    #print(s_matrix)
    #print(adj)
    #print(modularity_matrix)
    #sys.exit(0)

    sess = tf.Session()
    sess.run(model.init_op)

    best_loss = np.inf
    wait_step = 0


    for i in range(FLAGS.epochs):
        if FLAGS.model == 'LF_NMF':

            if featureless:
                M, U, C, H = sess.run([model.M, model.U, model.C, model.H], feed_dict=feed_dict)
                loss, strucure_recon_loss, community_recon_loss, modularity_loss, constraint_loss, low_filter_loss = model.loss(featureless)
                loss, strucure_recon_loss, community_recon_loss, modularity_loss, constraint_loss, low_filter_loss = sess.run([loss, strucure_recon_loss, community_recon_loss, modularity_loss, constraint_loss, low_filter_loss], feed_dict=feed_dict)
                print("Epoch %d: Loss of MNMF %.5f, structural loss %.5f, community loss %.5f, modularity value(max) %.5f, constraint loss %.5f, low_filter_loss %.5f"%(i+1, loss, strucure_recon_loss, community_recon_loss, modularity_loss, constraint_loss, low_filter_loss))
            else:
                M, U, C, H, F = sess.run([model.M, model.U, model.C, model.H, model.F], feed_dict=feed_dict)
                loss, strucure_recon_loss, community_recon_loss, modularity_loss, constraint_loss, low_filter_loss, feature_loss = model.loss(featureless)
                loss, strucure_recon_loss, community_recon_loss, modularity_loss, constraint_loss, low_filter_loss, feature_loss = sess.run([loss, strucure_recon_loss, community_recon_loss, modularity_loss, constraint_loss, low_filter_loss, feature_loss], feed_dict=feed_dict)
                print("Epoch %d: Loss of MNMF %.5f, structural loss %.5f, community loss %.5f, modularity value(max) %.5f, constraint loss %.5f, low_filter_loss %.5f, feature loss %.5f"%(i+1, loss, strucure_recon_loss, community_recon_loss, modularity_loss, constraint_loss, low_filter_loss, feature_loss))
        else:
            if featureless:
                M, U, C, H = sess.run([model.M, model.U, model.C, model.H], feed_dict=feed_dict)
                loss, strucure_recon_loss, community_recon_loss, modularity_loss, constraint_loss = model.loss(featureless)
                loss, strucure_recon_loss, community_recon_loss, modularity_loss, constraint_loss = sess.run([loss, strucure_recon_loss, community_recon_loss, modularity_loss, constraint_loss], feed_dict=feed_dict)
                print("Epoch %d: Loss of MNMF %.5f, structural loss %.5f, community loss %.5f, modularity value(max) %.5f, constraint loss %.5f"%(i+1, loss, strucure_recon_loss, community_recon_loss, modularity_loss, constraint_loss))
            else:
                M, U, C, H, F = sess.run([model.M, model.U, model.C, model.H, model.F], feed_dict=feed_dict)
                loss, strucure_recon_loss, community_recon_loss, modularity_loss, constraint_loss, feature_loss = model.loss(featureless)
                loss, strucure_recon_loss, community_recon_loss, modularity_loss, constraint_loss, feature_loss = sess.run([loss, strucure_recon_loss, community_recon_loss, modularity_loss, constraint_loss, feature_loss], feed_dict=feed_dict)
                print("Epoch %d: Loss of MNMF %.5f, structural loss %.5f, community loss %.5f, modularity value(max) %.5f, constraint loss %.5f, feature loss %.5f"%(i+1, loss, strucure_recon_loss, community_recon_loss, modularity_loss, constraint_loss, feature_loss))

        if i > FLAGS.early_stop:
            if loss < best_loss:
                best_loss = loss
            else:
                wait_step += 1
        if wait_step > FLAGS.early_stop:
            break
    embeddings = sess.run(model.U, feed_dict=feed_dict)
    #debug
    M, U, C, H = sess.run([model.M, model.U, model.C, model.H], feed_dict=feed_dict)
    print("M")
    print(M)
    print("U")
    print(U)
    print("C")
    print(C)
    print("H")
    print(H)
    return embeddings


def training_embedding_ANMF(gdata):
    sperate_name = 'all'
    if FLAGS.lp:
        val_pos_edges, val_neg_edges = gdata.get_link_prediction_data(data_name='val')
        sperate_name = 'train'
    if FLAGS.features:
        featureless = False
    else:
        featureless = True
    trainG = gdata.get_graph(sperate_name)

    #build content reconstruction weight
    if not featureless:
        features = gdata.get_features()
        sta_information = gdata.get_static_information()
        num_features = sta_information[0]
        features = utils.sparse_to_tuple(features)
    #f_pos_weight = float(features.shape[0] * features.shape[0] - features.sum()) / features.sum()
    #f_norm = features.shape[0] * features.shape[0] / float((features.shape[0] * features.shape[0] - features.sum()) * 2)

    #adj = gdata.get_adj(sperate_name)
    adj = gdata.get_orig_adj(sperate_name)
    num_nodes = adj.shape[0]

    #build second-order proximity matrix
    proximity_matrix = sp.coo_matrix(gdata.second_order_proximity_matrix(sperate_name))
    s_matrix = adj + FLAGS.ANMF_eta * proximity_matrix


    #print(s_matrix)
    #sys.exit(0)



    #build modularity matrix
    modularity_matrix = gdata.get_modularity_matrix(sperate_name)

    #convert to float64
    adj.astype(np.float64)
    modularity_matrix.astype(np.float64)

    #build sparse tensor turple
    adj = utils.sparse_to_tuple(adj)
    s_matrix = utils.sparse_to_tuple(s_matrix)

    used_dtype = tf.float64
    #used_dtype = tf.float32



    if featureless:
        placeholders = {'structure_mat':tf.sparse_placeholder(used_dtype),
                    'adj_mat':tf.sparse_placeholder(used_dtype),
                    'modularity_mat':tf.placeholder(shape=[num_nodes, num_nodes], dtype=used_dtype)
                    }
    else:
        placeholders = {'structure_mat':tf.sparse_placeholder(used_dtype),
                    'adj_mat':tf.sparse_placeholder(used_dtype),
                    'modularity_mat':tf.placeholder(shape=[num_nodes, num_nodes], dtype=used_dtype),
                    'feature_mat':tf.sparse_placeholder(used_dtype)
                    }


    config = {}
    config['embedding_size'] = FLAGS.hidden2
    config['s_community_num'] = FLAGS.ANMF_s_community_num
    config['c_community_num'] = FLAGS.ANMF_c_community_num
    config['anmf_alpha'] = FLAGS.ANMF_alpha
    config['anmf_beta'] = FLAGS.ANMF_beta
    config['anmf_gamma'] = FLAGS.ANMF_gamma
    config['anmf_phi'] = FLAGS.ANMF_phi

    config['anmf_rho'] = FLAGS.ANMF_rho
    config['anmf_theta'] = FLAGS.ANMF_theta

    config['lower_control'] = FLAGS.lower_control
    config['lambda'] = FLAGS.NMF_lambda
    config['kmeans_init'] = FLAGS.kmeans_init
    config['Hc'] = None
    config['Hs'] = None
    #init H value by kmeans result
    if FLAGS.kmeans_init:
        if not featureless:
            Hc = KMeans(n_clusters=FLAGS.ANMF_c_community_num, random_state=0).fit_predict(utils.tuple_to_sparse(features))
            Hc = utils.binarize_label(Hc).astype(np.float64)
            #print(Hc)
            config['Hc'] = Hc
            #sess.run(tf.assign(model.Hc, Hc), feed_dict=feed_dict)
        Hs = KMeans(n_clusters=FLAGS.ANMF_s_community_num, random_state=0).fit_predict(utils.tuple_to_sparse(s_matrix))
        Hs = utils.binarize_label(Hs).astype(np.float64)
        config['Hs'] = Hs
        #print(Hs)
        #sess.run(tf.assign(model.Hs, Hs), feed_dict=feed_dict)

    if not featureless:
        config['feature_dim'] = num_features

    if FLAGS.model == 'ANMF':
        model = ANMF(placeholders=placeholders, node_num=num_nodes, config=config, featureless=featureless)
    elif FLAGS.model == 'ANMFNEW':
        model = ANMFNEW(placeholders=placeholders, node_num=num_nodes, config=config, featureless=featureless)
    elif FLAGS.model == 'LF_ANMF':
        model = LF_ANMF(placeholders=placeholders, node_num=num_nodes, config=config, featureless=featureless)
    else:
        model = AHNMF(placeholders=placeholders, node_num=num_nodes, config=config, featureless=featureless)

    feed_dict = {}
    feed_dict.update({placeholders['structure_mat']:s_matrix})
    feed_dict.update({placeholders['adj_mat']:adj})
    feed_dict.update({placeholders['modularity_mat']:modularity_matrix})
    if not featureless:
        feed_dict.update({placeholders['feature_mat']:features})

    #print(s_matrix)
    #print(adj)
    #print(modularity_matrix)
    #sys.exit(0)

    sess = tf.Session()
    sess.run(model.init_op)



    best_loss = np.inf
    wait_step = 0


    for i in range(FLAGS.epochs):
        if FLAGS.model == 'AHNMF':
            M, U, F, G, Cs, Cc, Hc, Hs, Ps, Pc = sess.run([model.M, model.U, model.F, model.G, model.Cs, model.Cc, model.Hc, model.Hs, model.Ps, model.Pc], feed_dict=feed_dict)
            #print(Hc)
            #sys.exit(0)
            loss, feature_loss, strucure_recon_loss, s_community_recon_loss, c_community_recon_loss, sb_model_loss, s_community_recon_sym_loss, c_community_recon_sym_loss = model.loss(featureless)
            loss, feature_loss, strucure_recon_loss, s_community_recon_loss, c_community_recon_loss, sb_model_loss, s_community_recon_sym_loss, c_community_recon_sym_loss = sess.run([loss, feature_loss, strucure_recon_loss, s_community_recon_loss, c_community_recon_loss, sb_model_loss, s_community_recon_sym_loss, c_community_recon_sym_loss], feed_dict=feed_dict)
            print("Epoch %d: Loss of ANMF %.5f, feature loss %.5f, strucure loss %.5f, s community %.5f, c community %.5f, sb model loss %.5f, s sym community %.5f, c sym community %.5f"%(i, loss, feature_loss, strucure_recon_loss, s_community_recon_loss, c_community_recon_loss, sb_model_loss, s_community_recon_sym_loss, c_community_recon_sym_loss))
        elif FLAGS.model == 'LF_ANMF':
            M, U, F, G, Cs, Cc, Hc, Hs = sess.run([model.M, model.U, model.F, model.G, model.Cs, model.Cc, model.Hc, model.Hs], feed_dict=feed_dict)
            #if i % 5 == 0:
            loss, feature_loss, strucure_recon_loss, s_community_recon_loss, c_community_recon_loss, sb_model_loss, low_filter_loss, modularity_loss = model.loss(featureless)
            loss, feature_loss, strucure_recon_loss, s_community_recon_loss, c_community_recon_loss, sb_model_loss, low_filter_loss, modularity_loss = sess.run([loss, feature_loss, strucure_recon_loss, s_community_recon_loss, c_community_recon_loss, sb_model_loss, low_filter_loss, modularity_loss], feed_dict=feed_dict)
            print("Epoch %d:Loss of ANMF %.5f, feature loss %.5f, strucure loss %.5f, s community %.5f, c community %.5f, sb model loss %.5f, low filter loss %.5f, modularity loss %.5f"%(i,loss, feature_loss, strucure_recon_loss, s_community_recon_loss, c_community_recon_loss, sb_model_loss, low_filter_loss, modularity_loss))

        else:
            M, U, F, G, Cs, Cc, Hc, Hs = sess.run([model.M, model.U, model.F, model.G, model.Cs, model.Cc, model.Hc, model.Hs], feed_dict=feed_dict)
            if i % 50 == 0:
                loss, feature_loss, strucure_recon_loss, s_community_recon_loss, c_community_recon_loss, sb_model_loss = model.loss(featureless)
                loss, feature_loss, strucure_recon_loss, s_community_recon_loss, c_community_recon_loss, sb_model_loss = sess.run([loss, feature_loss, strucure_recon_loss, s_community_recon_loss, c_community_recon_loss, sb_model_loss], feed_dict=feed_dict)
                print("Epoch %d: Loss of ANMF %.5f, feature loss %.5f, strucure loss %.5f, s community %.5f, c community %.5f, sb model loss %.5f"%(i, loss, feature_loss, strucure_recon_loss, s_community_recon_loss, c_community_recon_loss, sb_model_loss))
        #sess.run(model.opt_op, feed_dict=feed_dict)
        #if FLAGS.ANMF_nonzero:
        #    sess.run(model.clip, feed_dict=feed_dict)

        if i > FLAGS.early_stop:
            if loss < best_loss:
                best_loss = loss
                wait_step = 0
            else:
                wait_step += 1
        if wait_step > FLAGS.early_stop:
            break
        #print(Hc[1])
        #print(Hs[1])
        #print(U[1])
    embeddings = sess.run(model.U, feed_dict=feed_dict)
    #debug
    return embeddings


def training_embedding_R(gdata):
    sperate_name = 'all'
    if FLAGS.lp:
        val_pos_edges, val_neg_edges = gdata.get_link_prediction_data(data_name='val')
        sperate_name = 'train'
    if FLAGS.features:
        featureless = False
    else:
        featureless = True
    trainG = gdata.get_graph(sperate_name)

    #build content reconstruction weight
    if not featureless:
        features = gdata.get_features()
        sta_information = gdata.get_static_information()
        num_features = sta_information[0]
        c_pos_weight = float(features.shape[0] * features.shape[0] - features.sum()) / features.sum()
        c_norm = features.shape[0] * features.shape[0] / float((features.shape[0] * features.shape[0] - features.sum()) * 2)
        features = utils.sparse_to_tuple(features)
    #f_pos_weight = float(features.shape[0] * features.shape[0] - features.sum()) / features.sum()
    #f_norm = features.shape[0] * features.shape[0] / float((features.shape[0] * features.shape[0] - features.sum()) * 2)

    #adj = gdata.get_adj(sperate_name)
    adj = gdata.get_orig_adj(sperate_name)
    num_nodes = adj.shape[0]
    #convert to float64
    adj.astype(np.float64)

    #build second-order proximity matrix
    proximity_matrix = sp.coo_matrix(gdata.second_order_proximity_matrix(sperate_name))
    s_matrix = adj + FLAGS.ANMF_eta * proximity_matrix


    #print(s_matrix)
    #sys.exit(0)


    a_pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    a_norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    s_pos_weight = float(s_matrix.shape[0] * s_matrix.shape[0] - s_matrix.sum()) / s_matrix.sum()
    s_norm = s_matrix.shape[0] * s_matrix.shape[0] / float((s_matrix.shape[0] * s_matrix.shape[0] - s_matrix.sum()) * 2)


    #build modularity matrix
    #modularity_matrix = gdata.get_modularity_matrix(sperate_name)

    #
    #modularity_matrix.astype(np.float64)

    #build sparse tensor turple
    adj = utils.sparse_to_tuple(adj)
    s_matrix = utils.sparse_to_tuple(s_matrix)
    #modularity_matrix = utils.sparse_to_tuple(modularity_matrix)


    used_dtype = tf.float64
    #used_dtype = tf.float32

    if featureless:
        placeholders = {'structure_mat':tf.sparse_placeholder(used_dtype),
                    'adj_mat':tf.sparse_placeholder(used_dtype),
                    #'modularity_mat':tf.placeholder(shape=[num_nodes, num_nodes], dtype=used_dtype)
                    }
    else:
        placeholders = {'structure_mat':tf.sparse_placeholder(used_dtype),
                    'adj_mat':tf.sparse_placeholder(used_dtype),
                    #'modularity_mat':tf.placeholder(shape=[num_nodes, num_nodes], dtype=used_dtype),
                    'feature_mat':tf.sparse_placeholder(used_dtype)
                    }


    config = {}
    config['embedding_size'] = FLAGS.hidden2
    config['s_community_num'] = FLAGS.ANMF_s_community_num
    config['c_community_num'] = FLAGS.ANMF_c_community_num
    config['alpha'] = FLAGS.R_alpha
    config['beta'] = FLAGS.R_beta
    config['gamma'] = FLAGS.R_gamma
    config['gate_emb_size'] = FLAGS.R_gate_emb_size
    config['a_norm'] = a_norm
    config['a_pos_weight'] = a_pos_weight
    config['s_norm'] = s_norm
    config['s_pos_weight'] = s_pos_weight

    if not featureless:
        config['feature_dim'] = num_features
        config['c_norm'] = c_norm
        config['c_pos_weight'] = c_pos_weight


    model = LinearREmb(placeholders=placeholders, node_num=num_nodes, config=config, featureless=featureless)

    feed_dict = {}
    feed_dict.update({placeholders['structure_mat']:s_matrix})
    feed_dict.update({placeholders['adj_mat']:adj})
    #feed_dict.update({placeholders['modularity_mat']:modularity_matrix})
    if not featureless:
        feed_dict.update({placeholders['feature_mat']:features})

    sess = tf.Session()
    sess.run(model.init_op)

    best_loss = np.inf
    wait_step = 0


    for i in range(FLAGS.epochs):
        _, loss, a_loss, s_loss, c_loss = sess.run([model.opt_op, model.loss, model.link_recon_loss, model.strucure_recon_loss, model.feature_recon_loss], feed_dict=feed_dict)
        print("Epoch %d, loss %.5f, link loss %.5f, strutural loss %.5f, content loss %.5f"%(i, loss, a_loss, s_loss, c_loss))

    embeddings = sess.run(model.U, feed_dict=feed_dict)
    #debug
    return embeddings





def main():
    gdata = Graphdata()
    if FLAGS.dataset in ['cora', 'citeseer', 'pubmed', 'wiki']:
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
        if FLAGS.model in ['DC2V', 'GraphAE', 'GraphVAE', 'DNBGraphAE']:
            embeddings = train_embedding_graphAE(gdata)
        elif FLAGS.model in ['HiRank2Vec', 'MiniBatchGraphAE']:
            embeddings = training_embedding_hirank2vec(gdata)
        elif FLAGS.model in ['MNMF', 'FMNMF', 'LF_NMF']:
            embeddings = training_embedding_NMF(gdata)
        elif FLAGS.model in ['ANMF', "AHNMF", 'ANMFNEW', 'LF_ANMF']:
            embeddings = training_embedding_ANMF(gdata)
        elif FLAGS.model in ["LinearREmb"]:
            embeddings = training_embedding_R(gdata)
        elif FLAGS.model in ['DGI', 'ClusterDGI']:
            if FLAGS.minibatch_flag:
                embeddings = train_embedding_MinibatchDGI(gdata)
            else:
                embeddings = training_embedding_DGI(gdata)
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
    else:
        femb = 'emb/%s/%s-%d.txt'%(FLAGS.dataset, FLAGS.model, FLAGS.hidden1)
        embeddings = utils.load_embedding(femb, delimiter=' ')

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

    if FLAGS.clf:
        trainX_indices, trainY = gdata.get_node_classification_data(name='train')
        valX_indices, valY = gdata.get_node_classification_data(name='val')
        testX_indices, testY = gdata.get_node_classification_data(name='test')

        trainX = embeddings[trainX_indices]
        valX = embeddings[valX_indices]
        testX = embeddings[testX_indices]

        class_num = np.shape(trainY)[1]

        #whole data version
        #clf = Classifier('SVM', class_num)
        #clf.train(trainX, trainY)
        #val_result = clf.evaluate(valX, valY)
        #print("val result of %s-%d: micro F1 %.5f, macro F1 %.5f"%(FLAGS.model, FLAGS.hidden1, val_result['micro'], val_result['macro']))
        #result = clf.evaluate(testX, testY)
        #print("test result of %s-%d: micro F1 %.5f, macro F1 %.5f"%(FLAGS.model, FLAGS.hidden1, result['micro'], result['macro']))

        #mini batch version
        clf = SGDClassifier(loss="hinge", penalty="l2")
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

        val_result = {}
        val_predY = clf.predict(valX)
        val_result['micro'] = f1_score(np.argmax(valY, axis=1), val_predY, average='micro')
        val_result['macro'] = f1_score(np.argmax(valY, axis=1), val_predY, average='macro')
        print("val result of %s-%d: micro F1 %.5f, macro F1 %.5f"%(FLAGS.model, FLAGS.hidden1, val_result['micro'], val_result['macro']))
        result = {}
        test_predY = clf.predict(testX)
        result['micro'] = f1_score(np.argmax(testY, axis=1), test_predY, average='micro')
        result['macro'] = f1_score(np.argmax(testY, axis=1), test_predY, average='macro')
        print("test result of %s-%d: micro F1 %.5f, macro F1 %.5f"%(FLAGS.model, FLAGS.hidden1, result['micro'], result['macro']))

        return val_result['micro'], val_result['macro'], result['micro'], result['macro']

def multi_run():
    micro_f1 = []
    macro_f1 = []
    k = 50
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
        #mi_f1, ma_f1, _, _ = main()
        mi_f1, ma_f1, mi_f1_std, ma_f1_std = multi_run()
        print("mean micro f1 %.5f, std %.5f, macro f1 %.5f, std %.5f"%(mi_f1, mi_f1_std, ma_f1, ma_f1_std))
    print('(%.4f, %.4f)' %(mi_f1, ma_f1))
