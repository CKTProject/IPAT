import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import sys
from sklearn.preprocessing import label_binarize, LabelBinarizer
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.cluster import SpectralClustering
from sklearn.utils import shuffle
import random
import time
import json
import tensorflow as tf
import os

from ogb.nodeproppred import Evaluator, NodePropPredDataset

sys.setrecursionlimit(99999)


#sample point on the surface of unit sphere
def unit_sphere_sample(shape):
    noise = tf.random.normal(shape=shape)
    points = tf.math.l2_normalize(noise, axis=-1)
    return points

#gumbel softmax Implementation
def sample_gumbel(shape, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = tf.random_uniform(shape,minval=0,maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(tf.shape(logits))
    return tf.nn.softmax( y / temperature)

def gumbel_softmax(logits, temperature, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        k = tf.shape(logits)[-1]
        #y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
        y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y


def run_dfs(adj, msk, u, ind, nb_nodes):
    if msk[u] == -1:
        msk[u] = ind
        #for v in range(nb_nodes):
        for v in adj[u,:].nonzero()[1]:
            #if adj[u,v]== 1:
            run_dfs(adj, msk, v, ind, nb_nodes)

# Use depth-first search to split a graph into subgraphs
def dfs_split(adj):
    # Assume adj is of shape [nb_nodes, nb_nodes]
    nb_nodes = adj.shape[0]
    ret = np.full(nb_nodes, -1, dtype=np.int32)

    graph_id = 0

    for i in range(nb_nodes):
        if ret[i] == -1:
            run_dfs(adj, ret, i, graph_id, nb_nodes)
            graph_id += 1

    return ret

# Sample k neighbors from adjacency matrix(scipy.sparse.csr_matrix)
def sample_k_neighbors_from_adj(adj, k, seed_nodes, padding=False):
    #if paddding, sample random nodes from graph when no neighbors else padding self node
    adj = adj.tocsr()
    node_num = adj.shape[0]
    indices = adj.indices
    indptr = adj.indptr
    neighbors_corpus = []
    for node in seed_nodes:
        neighbors = indices[indptr[node]:indptr[node+1]]
        #print(neighbors)
        if not neighbors.size:
            if padding:
                neighbors = np.array(list(range(node_num)))
            else:
                neighbors = np.array([node])
        sampled_idx = np.random.randint(low=0, high=len(neighbors), size=(k,), dtype=np.int32)
        sampled_nodes = neighbors[sampled_idx]
        neighbors_corpus.append(np.array(sampled_nodes))
    return np.vstack(neighbors_corpus)



#calculate pagerank value of graph, and add self defined sparse operator
def dict_to_array(d):
    items = list(d.items())
    a = sorted(items, key=lambda x:int(x[0]))
    a = [v[1] for v in a]
    return np.array(a)

def get_personal_pagerank_values(adj, dataset='cora', alpha=0.85, sparse=False, topK=10):
    #check if pagerank value is exits
    ppr_values_path = 'temp/%s_%f_ppr.pkl'%(dataset, alpha)
    if os.path.exists(ppr_values_path):
        with open(ppr_values_path, 'rb') as f:
            pr_values = np.load(f)
    else:
        node_num = adj.shape[0]
        M = cal_A_hat(adj)
        A_inner = sp.eye(node_num) - (1 - alpha) * M
        pr_values = alpha * np.linalg.inv(A_inner.toarray())
    return sp.csr_matrix(pr_values)



def softmax(x, axis=1):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.expand_dims(np.max(x, axis=1), axis=axis))
    e_sum = np.expand_dims(e_x.sum(axis=axis), axis=axis)
    return e_x / e_sum # only difference


def set_row_of_spmatrix_to_val(spmatrix, rows, value=0):
    if not sp.isspmatrix_csr(spmatrix):
        spmatrix = spmatrix.tocsr()
    if isinstance(rows, int):
        rows = [rows]
    for r in rows:
        spmatrix.data[spmatrix.indptr[r]:spmatrix.indptr[r+1]] = value
    spmatrix.eliminate_zeros()
    return spmatrix

def reset_value_of_spmatrix_by_row(spmatrix, row_indx, values, reset_indx):
    #reset row value of spmatrix by a new array of values and corresponding index
    if not sp.isspmatrix_csr(spmatrix):
        spmatrix = spmatrix.tocsr()
    assert row_indx < spmatrix.shape[0], \
            'The row index ({0}) shall be smaller than the number of rows in A ({1})' \
            .format(row_indx, A.shape[0])
    assert isinstance(values, list) or isinstance(values, np.ndarray), 'new value should be list or array'
    assert len(values) == len(reset_indx), 'new value number should be equal with index '

    col_num = spmatrix.shape[1]
    assert np.sum(np.array(reset_indx)>=col_num) == 0, 'add index out of range'

    idx_start_row = spmatrix.indptr[row_indx]
    idx_end_row = spmatrix.indptr[row_indx+1]
    new_value_num = len(values)
    d_value = int(new_value_num - (idx_end_row - idx_start_row))

    spmatrix.data = np.r_[spmatrix.data[:idx_start_row], np.array(values, dtype=spmatrix.data.dtype), spmatrix.data[idx_end_row:]]
    spmatrix.indices = np.r_[spmatrix.indices[:idx_start_row], np.array(reset_indx, dtype=spmatrix.indices.dtype), spmatrix.indices[idx_end_row:]]
    spmatrix.indptr = np.r_[spmatrix.indptr[:row_indx+1], spmatrix.indptr[row_indx+1:] + d_value]

    spmatrix.eliminate_zeros()

    return spmatrix

def get_value_and_index_of_spmatrxi_by_row(spmatrix, row_indx):
    if not sp.isspmatrix_csr(spmatrix):
        spmatrix = spmatrix.tocsr()
    assert row_indx < spmatrix.shape[0], \
            'The row index ({0}) shall be smaller than the number of rows in A ({1})' \
            .format(row_indx, A.shape[0])
    idx_start_row = spmatrix.indptr[row_indx]
    idx_end_row = spmatrix.indptr[row_indx+1]
    data = spmatrix.data[idx_start_row:idx_end_row]
    index = spmatrix.indices[idx_start_row:idx_end_row]
    return data, index

def PPMI(A, k=2, flag=True): 
    N = A.shape[0]

    # symmetric normalization
    D = sp.diags(1./np.sqrt(np.array(np.sum(A, axis=1)).reshape((1, N))), [0])
    A = D * A * D

    As = []
    tmp = A
    for i in range(k-1):
        tmp = A * tmp
        As.append(tmp)
    for i in range(k-1):
        A = A + As[i]
    A = A/k
    
    D = sp.diags(1./np.sqrt(np.array(np.sum(A, axis=1)).reshape((1, N))), [0])
    A = D * A * D
    
    if flag:
        # delete zero elements from the sparse matrix
        A.data = np.log(A.data) - np.reshape(np.array([np.log(1.0/N)] * A.data.shape[0]), (A.data.shape[0], ))
        A.data = np.maximum(A.data, 0)
    
    # A = sp.csc.csc_matrix(A)
    return A

def random_add_value_in_spmatrix(spmatrix, rate=0.1):
    if not sp.isspmatrix_csr(spmatrix):
        spmatrix = spmatrix.tocsr()
    before = spmatrix.nnz
    shape = spmatrix.shape
    density = len(spmatrix.data)*rate / (shape[0] * shape[1])
    noise = sp.random(shape[0], shape[1], density=density)
    spmatrix = spmatrix + noise
    spmatrix.data[:] = 1.
    after = spmatrix.nnz
    #print("actual add values %f compare set rate %f"%(float((after-before)/before), rate))
    return spmatrix

def enlarge_sparse_matrix_along_dim(spmatrix, axis, dim_num=1):
    #enlarge the matrix along the axis
    raw_shape = spmatrix.shape
    new_shape = list(raw_shape)
    new_shape[axis] += dim_num
    spmatrix.resize(new_shape)
    return spmatrix


def cluster_corruption_function(x=None, f='raw'):
    if f == 'raw':
        return corruption_function(x)
    else:
        #return corruption_function(x.T).T
        node_num = x.shape[0]
        label_num = x.shape[1]
        res = np.zeros(shape=x.shape, dtype=np.int)
        labels = np.argmax(x, axis=1)
        for i in range(node_num):
            neg = random.randint(0, label_num-1)
            while labels[i] == neg:
                neg = random.randint(0, label_num-1)
            res[i][neg] = 1
        return res



def corruption_function(x=None):
    x_corp = None
    if not x is None:
        x_corp = shuffle(x)
    return x_corp


def binarize_label(labels):
    label_distribution = LabelBinarizer().fit_transform(labels.reshape(-1, 1))
    return label_distribution

def build_cluster_mask(graph, clustering, default_cluster=True):
    #build cluster mask matrix, if set default cluster = True, then a cluster contain all nodes
    n = graph.number_of_nodes()
    cluster_num = len(clustering)
    if default_cluster:
        cluster_num += 1
    cluster_mask = np.zeros(shape=(n, cluster_num), dtype=float)
    if default_cluster:
        cluster_mask[:, -1] = np.ones(n)
    for i, nodes in enumerate(clustering):
        for node in nodes:
            cluster_mask[node, i] = 1
    return cluster_mask



def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    if sp.issparse(features):
        return sparse_to_tuple(features)
    return features


def tuple_to_sparse(tuples):
    row, col = np.split(tuples[0], 2, axis=1)
    row = np.squeeze(row)
    col = np.squeeze(col)
    sp_matrix = sp.coo_matrix((tuples[1], (row, col)), shape=tuples[2])
    return sp_matrix

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def expand_csr_matrix(sparse_mx, axis=0):
    if not sp.isspmatrix_csr(sparse_mx):
        sparse_mx = sparse_mx.tocsr()
    ori_shape = sparse_mx.shape
    new_shape = list(ori_shape)
    new_shape[axis] = new_shape[axis] + 1
    sparse_mx._shape = (new_shape[0], new_shape[1])
    if axis == 0:
        sparse_mx.indptr = np.hstack((sparse_mx.indptr, sparse_mx.indptr[-1]))
    return sparse_mx


def cal_A_hat(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return adj_normalized


def preprocess_graph(adj):
    adj_normalized = cal_A_hat(adj)
    return sparse_to_tuple(adj_normalized)

def laplacian_adj(adj, normalized=True):
    if normalized:
        """Symmetrically normalize adjacency matrix."""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -1.0).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        adj_normalized = adj.dot(d_mat_inv_sqrt).transpose().tocoo()
        laplacian = sp.eye(adj.shape[0]) - adj_normalized
    else:
        '''L = D - A'''
        adj = sp.coo_matrix(adj)
        rowsum = np.squeeze(np.array(adj.sum(1)))
        D = sp.diags(rowsum)
        laplacian = D - adj
        laplacian = laplacian.astype(np.float32)
    return laplacian


def get_roc_score(edges_pos, edges_neg, emb, adj_orig=None):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        if adj_orig is None:
            pos.append(1.0)
        else:
            pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        if adj_orig is None:
            neg.append(0.0)
        else:
            neg.append(adj_orig[e[0], e[1]])


    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

def get_acc_score(edges_pos, edges_neg, emb, adj_orig=None):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        if adj_orig is None:
            pos.append(1.0)
        else:
            pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        if adj_orig is None:
            neg.append(0.0)
        else:
            neg.append(adj_orig[e[0], e[1]])


    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
    ap_score = average_precision_score(labels_all, preds_all)

    return ap_score


#load embedding file, baselines outputs format(LINE, Deepwalk, Node2vec)
#format:
#node_num emb_dim
#ID dim1 dim2 ...
def load_embedding_data_sparse(fname):
    with open(filename, 'r') as fin:
        content = fin.read().strip().split('\n')
        items = content[0].split(' ')
        nodes, size = int(items[0]), int(items[1])
        features = np.zeros((nodes, size))
        for i, line in enumerate(content[1:]):
            items = line.split(' ')
            ids = int(items[0]) - 1
            emb = np.array([float(it) for it in items[1:]])
            features[ids, :] = emb
            return features


def load_embedding(fname, delimiter=' ', skiprows=1):
    emb = np.loadtxt(fname, delimiter=delimiter, skiprows=skiprows)
    return emb

def save_embedding(fname, X, delimiter=' ', header=None):
    if not header:
        np.savetxt(fname, X, delimiter=delimiter)
    else:
        np.savetxt(fname, X, delimiter=delimiter, header=header)


class ConfigerLoader(object):
    """load config information from json or the other files"""
    def __init__(self):
        pass

    def load_graphAE_from_json(self, filename):
        config_dict = json.load(open(filename, 'r'))
        n_heads_list = [int(h) for h in config_dict['n_heads_list'].split('.')]
        hidden_dim_list = [int(h) for h in config_dict['hidden_dim_list'].split('.')]
        layer_type_list = config_dict['layer_type_list'].split('.')
        return n_heads_list, hidden_dim_list, layer_type_list







class ClusterMiniBatchGenerator(object):
    '''generare mini-batch data for cluster based NRL'''
    def __init__(self, adj, features, cluster_contain_list, node2cluster, batch_size):
        #cluster_contain_list:enumerate nodes that each cluster contains
        #batch_size:number of clusters each batch contains
        self.adj = adj
        self.features = features
        self.cluster_contain_list = cluster_contain_list
        self.node2cluster = node2cluster
        self.batch_size = batch_size

        self.cluster_num = len(cluster_contain_list)

        self.end = False
        self.c_idx = shuffle(list(range(self.cluster_num)))
        self.c_base = 0



    def refresh(self):
        self.end = False
        self.c_idx = shuffle(self.c_idx)
        self.c_base = 0


    def generate(self):
        batch_nodes = []
        for i in range(self.c_base, min(self.cluster_num, self.c_base+self.batch_size)):
            batch_nodes += self.cluster_contain_list[self.c_idx[i]]

        batch_adj = self.adj[batch_nodes, :][:, batch_nodes]
        batch_features = self.features[batch_nodes]
        batch_node2cluster = self.node2cluster[batch_nodes]

        self.c_base += self.batch_size
        if self.c_base >= self.cluster_num:
            self.end = True

        return batch_adj, batch_features, batch_node2cluster, batch_nodes


class NeiSampler(object):
    #sample fix number of neighbors for each node in graph. If the degree of node less than the sample number, padding with auxiliary node(node id = max node id + 1)
    def __init__(self, graph, nb_size, include_self=False):
        n = graph.number_of_nodes()
        if include_self:
            nb_all = np.zeros(shape=(n, nb_size+1), dtype=int)
            nb_all[:, 0] = np.arange(n)
        else:
            nb_all = np.zeros(shape=(n, nb_size), dtype=int)
        popnb = []
        for v in range(n):
            nb_v = sorted(graph.neighbors(v))
            if len(nb_v) <= nb_size:
                nb_v = nb_v + [n]*(nb_size-len(nb_v))
                if include_self:
                    nb_all[v, 1:] = np.array(nb_v)
                else:
                    nb_all[v] = np.array(nb_v)
            else:
                popnb.append(v)
        self.graph, self.nb_all, self.popnb = graph, nb_all, popnb
        self.include_self = include_self
        self.nb_size = nb_size

    def sample(self):
        nb = self.nb_all
        for v in self.popnb:
            sample_nb = np.random.choice(sorted(self.graph.neighbors(v)), self.nb_size)
            if self.include_self:
                nb[v, 1:] = np.array(sample_nb)
            else:
                nb[v,:] = np.array(sample_nb)
        return nb

class MiniBatchGenerator(object):
    #generate mini batch data for mini-batch version of GraphAE
    def __init__(self, nb, neg_size, batch_size, num_nodes):
        self.nb = nb
        self.neg_size = neg_size
        self.batch_size = batch_size
        self.num_nodes = num_nodes
        self.k = 0
        self.end = False

    def _filter_repeat_nodes(self, nodes, nb_nodes):
        used_nodes = {}
        for n in nodes.tolist():
            used_nodes[n] = 1
        nb_nodes = np.squeeze(np.reshape(nb_nodes, shape=(1, -1))).tolist()
        for n in nb_nodes:
            used_nodes[n] = 1
        used_nodes = sorted(used_nodes.keys())
        if used_nodes[-1] == self.num_nodes:
            used_nodes = used_nodes[:-1]
        return used_nodes

    def refresh(self, nb):
        self.nb = nb
        self.k = 0
        self.end = False


    def generate(self):
        left_id = (self.k * self.batch_size) % self.num_nodes
        right_id = min(left_id + self.batch_size, self.num_nodes)
        cur_nodes = np.array(list(range(left_id, right_id)))
        cur_nb_nodes = nb[cur_nodes, :]
        one_hop_nodes = self._filter_repeat_nodes(cur_nodes, cur_nb_nodes)
        one_hop_nb_nodes = nb[one_hop_nodes, :]
        edge_a, edge_b, neg_samples = self.generate_edges(self.neg_size, cur_nodes, cur_nb_nodes)
        self.k += 1
        if self.k * self.batch_size > self.num_nodes:
            self.end = True
        return cur_nodes, cur_nb_nodes, one_hop_nodes, one_hop_nb_nodes, edge_a, edge_b, neg_samples

    def generate_edges(self, neg_size, cur_nodes, cur_nb_nodes):
        edge_a = []
        edge_b = []
        for i, n in enumerate(cur_nodes.tolist()):
            nbs = cur_nb_nodes[i, :].tolist()
            for nb in nbs:
                if nb == self.num_nodes:
                    continue
                edge_a.append(n)
                edge_b.append(nb)
        size = len(edge_a)
        neg_samples = np.random.random_integers(self.num_nodes, size=(size, neg_size))
        return np.array(edge_a), np.array(edge_b), neg_samples



class HierRankData(object):
    #generate hierarchical cluster rank data
    def __init__(self, cluster_num_list, adj, directed=False):
        if not directed:
            self.base_graph = nx.Graph()
        else:
            self.base_graph = nx.DiGraph()
        self.graph = nx.from_scipy_sparse_matrix(A=adj, create_using=self.base_graph)

        self.cluster_num_list = cluster_num_list
        self._HierarchicalCluster()
        self._DistinctHierarchicalNeighbors()
        self._negative_prepare_flag = False

    def _SpectralCluster(self, X, n_clusters, assign_labels='discretize'):
        #cluster parameters: assign_labels ("discretize", 'kmeans')
        clustering = SpectralClustering(n_clusters=n_clusters,
                                        assign_labels=assign_labels)
        clustering.fit(X)
        cluster_distribution = LabelBinarizer().fit_transform(clustering.labels_.reshape(-1, 1))
        return cluster_distribution

    def _HierarchicalCluster(self):
        self.hierarchical_graph_list = []
        self.hierarchical_graph_list.append(self.graph)
        self.hierarchical_adj_list = []
        self.hierarchical_adj_list.append(nx.to_scipy_sparse_matrix(self.graph))
        last_cluster_distribution = sp.identity(self.graph.number_of_nodes())
        for n_clusters in self.cluster_num_list:
            cluster_distribution = sp.csr_matrix(self._SpectralCluster(X=self.hierarchical_adj_list[-1], n_clusters=n_clusters))
            next_adj = cluster_distribution.T.dot(self.hierarchical_adj_list[-1].dot(cluster_distribution))
            self.hierarchical_adj_list.append(next_adj)
            last_cluster_distribution = last_cluster_distribution.dot(cluster_distribution)
            cluster_graph = nx.from_scipy_sparse_matrix(last_cluster_distribution.dot(last_cluster_distribution.T))
            self.hierarchical_graph_list.append(cluster_graph)

    def _DistinctHierarchicalNeighbors(self):
        nodes = sorted(self.graph.nodes)
        self.distinct_hierarchical_neighbors = []

        hierarchi_num = len(self.cluster_num_list)
        for n in nodes:
            dn = []
            directed_neighbors = list(self.hierarchical_graph_list[0].neighbors(n))
            dn.append(directed_neighbors)
            directed_neighbors = set(directed_neighbors)
            used_nodes = directed_neighbors
            for i in range(hierarchi_num):
                cluster_neighbors = set(list(self.hierarchical_graph_list[i+1].neighbors(n)))
                dn.append(list(cluster_neighbors.difference(used_nodes)))
                used_nodes= used_nodes.union(cluster_neighbors)
            self.distinct_hierarchical_neighbors.append(dn)



    def GenerateTripleData(self, k=5, nodes=None):
        #using node-anchored sampling strategy as shown in "Deep Gaussian embedding of graphs:unsupervised inductive learning via ranking"
        #sample one node in each level of cluster, and build triple pair
        if not nodes:
            nodes = self.graph.nodes

        sampled_triple = []
        triple_weight = []
        for n in nodes:
            distinct_nodes_list = self.distinct_hierarchical_neighbors[n]
            sampled_node = []
            sampled_cluster_size = []
            for distinct_nodes in distinct_nodes_list:
                if not distinct_nodes:
                    sampled_node.append([])
                else:
                    sampled_node.append(random.choice(distinct_nodes))
                sampled_cluster_size.append(len(distinct_nodes))
            length = len(sampled_node)
            for i in range(length):
                if not sampled_node[i]:
                    continue
                c = sampled_node[i]
                for j in range(i+1, length):
                    if not sampled_node[j]:
                        continue
                    d = sampled_node[j]
                    triple = [n, c, d]
                    weight = sampled_cluster_size[i] * sampled_cluster_size[j]
                    sampled_triple.append(triple)
                    triple_weight.append(weight)
        return sampled_triple, triple_weight

    def _PrepareNegaSample(self):
        nodes = self.graph.nodes
        self.linked_nodes = {}
        for n in nodes:
            self.linked_nodes[n] = {}
            for graph in self.hierarchical_graph_list:
                neighbors = graph.neighbors(n)
                for nei in neighbors:
                    if nei not in self.linked_nodes[n]:
                        self.linked_nodes[n][nei] = 1
        self._negative_prepare_flag = True



    def GenerateNegativeTripleData(self, k=5, nodes=None):
        if not self._negative_prepare_flag:
            self._PrepareNegaSample()
        if not nodes:
            nodes = self.graph.nodes
        negative_triple_data = []
        negative_triple_weight = []
        number_of_nodes = self.graph.number_of_nodes()
        for n in nodes:
            close_nodes = random.choices(list(self.linked_nodes[n].keys()), k=k)
            distant_nodes = []
            while len(distant_nodes) < k:
                distant = random.randint(0, number_of_nodes-1)
                if distant not in self.linked_nodes[n]:
                    distant_nodes.append(distant)
            for i, c in enumerate(close_nodes):
                d = distant_nodes[i]
                negative_triple_data.append([n, c, d])
                negative_triple_weight.append(1.0)
        return negative_triple_data, negative_triple_weight



class Graphdata(object):

    def __init__(self):
        self.node_label = False
        self._data_name = None

    #add node class information
    def _attach_node_class_label(self, label_value):
    #some node in citeseer do not have label
        y_shape = np.shape(label_value)
        if y_shape[1] > 1:
            labels = np.argmax(label_value, axis=1)
            label_filter = np.sum(label_value, axis=1)
        for i in range(y_shape[0]):
            if label_filter[i] > 0:
                self.g.nodes[i]['label'] = labels[i]
            else:
                self.g.nodes[i]['label'] = -1
        return

    #add node train/test information
    #label_value set = ('train', 'value', 'test')
    def _attach_node_label(self, label_value, node_set):
        for node in node_set:
            self.g.nodes[node][label_value] = 1
        return

    def _attach_edge_label(self, label_value, edge_set, direct=False):
        for edge in edge_set:
            x, y = edge[0], edge[1]
            self.g[x][y][label_value] = 1
            if not direct:
                self.g[y][x][label_value] = 1
        return

    def _attach_features(self, features):
        shape = np.shape(features)
        for i in range(shape[0]):
            self.g.nodes[i]['feature'] = features[i]
        return

    def _ismember(self, a, b, tol=5):
        #b[:, None] == b[:, newaxis]
        b = np.array(b)
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)


    def _sample_negative_edges(self, edges_all, train_edges, test_edges, val_edges, node_num):

        test_edges_false = []
        while len(test_edges_false) < len(test_edges):
            idx_i = np.random.randint(0, node_num)
            idx_j = np.random.randint(0, node_num)
            if idx_i == idx_j:
                continue
            if self._ismember([idx_i, idx_j], edges_all):
                continue
            if self._ismember([idx_j, idx_i], edges_all):
                continue
            if test_edges_false:
                if self._ismember([idx_j, idx_i], np.array(test_edges_false)):
                    continue
                if self._ismember([idx_i, idx_j], np.array(test_edges_false)):
                    continue
            test_edges_false.append([idx_i, idx_j])

        val_edges_false = []
        while len(val_edges_false) < len(val_edges):
            idx_i = np.random.randint(0, node_num)
            idx_j = np.random.randint(0, node_num)
            if idx_i == idx_j:
                continue
            #if self._ismember([idx_i, idx_j], edges_all):
             #   continue
            if self._ismember([idx_i, idx_j], train_edges):
                continue
            if self._ismember([idx_j, idx_i], train_edges):
                continue
            if self._ismember([idx_i, idx_j], val_edges):
                continue
            if self._ismember([idx_j, idx_i], val_edges):
                continue
            #make sure negative edge not exists in test data set, kipf gae not do this
            #print(test_edges_false)
            #print([idx_i, idx_j])
            if self._ismember([idx_i, idx_j], test_edges_false):
                continue
            if self._ismember([idx_j, idx_i], test_edges_false):
                continue

            if val_edges_false:
                if self._ismember([idx_j, idx_i], np.array(val_edges_false)):
                    continue
                if self._ismember([idx_i, idx_j], np.array(val_edges_false)):
                    continue

            val_edges_false.append([idx_i, idx_j])


#build train negative node pair, used for train logstic regression model for node2vec/deepwalk
#'''
#        train_edges_false = []
#        while len(train_edges_false) < len(train_edges):
#            idx_i = np.random.randint(0, node_num)
#            idx_j = np.random.randint(0, node_num)
#            if idx_i == idx_j:
#                continue
#            #if self._ismember([idx_i, idx_j], edges_all):
#             #   continue
#            if self._ismember([idx_i, idx_j], train_edges):
#                continue
#            if self._ismember([idx_j, idx_i], train_edges):
#                continue
#            if self._ismember([idx_i, idx_j], test_edges_false):
#                continue
#            if self._ismember([idx_j, idx_i], test_edges_false):
#                continue
#            if train_edges_false:
#                if self._ismember([idx_j, idx_i], np.array(train_edges_false)):
#                    continue
#                if self._ismember([idx_i, idx_j], np.array(train_edges_false)):
#                    continue
#
#            train_edges_false.append([idx_i, idx_j])
#'''

        assert ~self._ismember(test_edges_false, edges_all)
        assert ~self._ismember(val_edges_false, train_edges)
        assert ~self._ismember(val_edges_false, val_edges)
        assert ~self._ismember(val_edges, train_edges)
        assert ~self._ismember(test_edges, train_edges)
        assert ~self._ismember(val_edges, test_edges)
        #check train false edges
        #assert ~self._ismember(train_edges_false, train_edges)
        #assert ~self._ismember(val_edges_false, np.array(test_edges_false))
        #assert ~self._ismember(train_edges_false, np.array(test_edges_false))

        self.g.graph['val_edges_false'] = val_edges_false
        self.g.graph['test_edges_false'] = test_edges_false
        #self.g.graph['train_edges_false'] = train_edges_false
        return

    def get_node_classification_data(self, name='train'):

        if self._data_name == 'ogbn-arxiv':

            def extract_data_by_index(idx):
                #X = self.ogb_x[idx]
                labels = self.ogb_label[idx]
                Y = label_binarize(labels, classes=self.ogb_label_set)
                return idx, Y

            if name == 'train':
                return extract_data_by_index(self.ogb_idx_tr)
            elif name == 'val':
                return extract_data_by_index(self.ogb_idx_va)
            else:
                return extract_data_by_index(self.ogb_idx_te)
        else:
            if name == 'all':
                nodes = self.get_nodes(name)
            else:
                nodes = nx.get_node_attributes(self.g, name).keys()

            if not nodes:
                print("%s set don't have nodes!!!!!"%name)
                return None, None
            labels_items = nx.get_node_attributes(self.g, 'label').items()
            assert len(labels_items) == self.g.number_of_nodes(), 'some nodes without label information'
            labels_items = sorted(labels_items, key=lambda x:int(x[0]))
            labels = [int(items[1]) for items in labels_items]


            labels_set = list(sorted(set(sorted(labels))))
            #max_label = labels[-1]
            print("node label set:" + str(labels_set))

            X = np.array([int(n) for n in nodes], dtype=np.int)
            #some data in citeseer do not have label, marked as -1
            if labels_set[0] == -1:
                labels_set = labels_set[1:]

            labels = np.array(labels)
            label_value = labels[X]

            filter_indx = label_value == -1
            label_value[filter_indx] = 0

            Y = label_binarize(label_value, classes=labels_set)
            zeros = np.zeros(shape=np.shape(Y))
            Y[filter_indx] = zeros[filter_indx]
        return X, Y

    def get_node_degree(self, name='all'):
        degree_dict = {}
        if name == 'train':
            degree_dict = self.train_graph.degree()
        else:
            degree_dict = self.g.degree()
        degree_list = sorted(dict(degree_dict).items(), key=lambda x:int(x[0]))
        degree_array = np.array(degree_list)
        degree_array = np.hsplit(degree_array, 2)[1]
        degree_array = np.squeeze(degree_array)
        return degree_array

    def get_neighbors(self, node, name='all'):
        if name == 'all':
            graph = self.train_graph
        else:
            graph = self.g

        neighbors = graph.neighbors(node)
        return neighbors


    def get_grpahsage_adj_list(self, max_degree=25, name='all'):
        if name == 'train':
            node_num = self.train_graph.number_of_nodes()
            graph = self.train_graph
        else:
            node_num = self.g.number_of_nodes()
            graph = self.g
        adj = node_num * np.ones((node_num+1, max_degree))
        for node in graph.nodes():
            neighbors = np.array(list(graph.neighbors(node)))
            if len(neighbors) == 0:
                    continue
            if len(neighbors) > max_degree:
                neighbors = np.random.choice(neighbors, max_degree, replace=False)
            elif len(neighbors) < max_degree:
                neighbors = np.random.choice(neighbors, max_degree, replace=True)
            adj[node, :] = neighbors
        return adj

    def get_edges_num(self, name='all'):
        if name == 'train':
            return self.train_graph.number_of_edges()
        return self.g.number_of_edges()

    def get_edges(self, name='all'):
        if name == 'train':
            return self.train_graph.edges()
        return self.g.edges()

    def get_nodes(self, name='all'):
        if name == 'train':
            return self.train_graph.nodes()
        return self.g.nodes()


    def get_link_prediction_data(self, data_name='val'):
        pos_edges = self.g.graph['%s_edges'%data_name]
        false_edges = self.g.graph['%s_edges_false'%data_name]
        return pos_edges, false_edges

    def get_features(self):
        #return ogb dataset
        if self._data_name == 'ogbn-arxiv':
            return self.ogb_x

        #if 'feature' in self.g.nodes[0]:
        #    print("load feature")
        #    seed_nodes = sorted(self.g.nodes())
        #    features = []
        #    for seed in seed_nodes:
        #        features.append(self.g.nodes[seed]['feature'])
        #    self.features = sp.vstack(features)
        #    #self.features = nx.attr_matrix(self.g, node_attr='feature', rc_order=seed_nodes)
        if self.features is None:
            print("load featureless")
            self.features = sp.identity(self.num_nodes)

        #self.features = sp.csr_matrix(self.features)
        return self.features
        #return sparse_to_tuple(self.features)

    def get_graph(self, name='all'):
        if name == 'train':
            return self.train_graph
        return self.g

    def get_adj(self, name='all'):
        #return ogb dataset
        if self._data_name == 'ogbn-arxiv':
            return self.ogb_adj

        if name == 'train':
            adj = nx.adjacency_matrix(self.train_graph)
        else:
            adj = nx.adjacency_matrix(self.g)
        return adj
        #return preprocess_graph(adj)

    def get_orig_adj(self, name='all'):
        #return ogb adj matrix
        if self._data_name == 'ogbn-arxiv':
            adj = self.ogb_adj
        else:
            if name == 'train':
                adj = nx.adjacency_matrix(self.train_graph)
            else:
                adj = nx.adjacency_matrix(self.g)
        adj_orig = adj
        adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
        adj_orig.eliminate_zeros()
        return adj_orig
        #return preprocess_graph(adj_orig)

    def get_modularity_matrix(self, name='all'):
        edges_num = self.get_edges_num(name)
        if name == 'train':
            adj = nx.adjacency_matrix(self.train_graph)
        else:
            adj = nx.adjacency_matrix(self.g)
        degree_vec = np.array(adj.sum(1)).reshape((-1, 1))
        modularity_matrix = np.matmul(degree_vec, degree_vec.T) / edges_num
        modularity_matrix = 0.5 * modularity_matrix
        return modularity_matrix


    def second_order_proximity_matrix(self, name='all'):
        adj = self.get_adj(name=name)
#method 1
        adj_sq = adj.power(2)
        rowsum = np.array(adj_sq.sum(1))
        mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        normalized_adj = mat_inv_sqrt.dot(adj)
        normalized_proximity_matrix = normalized_adj.dot(normalized_adj.T)
        #remove diag elements
        normalized_proximity_matrix = normalized_proximity_matrix - sp.dia_matrix((normalized_proximity_matrix.diagonal()[np.newaxis, :], [0]), shape=normalized_proximity_matrix.shape)

#method 2
        #adj = sp.coo_matrix(adj)
        #rowsum = np.array(adj.sum(1))
        #degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        #proximity_matrix = adj.dot(adj.T)
        #normalized_proximity_matrix = (degree_mat_inv_sqrt.dot(proximity_matrix)).dot(degree_mat_inv_sqrt)

        return normalized_proximity_matrix



    def get_static_information(self):
        features = sparse_to_tuple(self.get_features())
        num_features = features[2][1]
        features_nonzero = features[1].shape[0]
        return (num_features, features_nonzero)

    def _build_train_adj(self):
        node_num = self.g.number_of_nodes()
        self.train_graph = nx.Graph()
        self.train_graph.add_nodes_from(range(node_num))
        self.train_graph.add_edges_from(self.g.graph['train_edges'])

    def random_split_train_test(self, node_train_ratio=0.0, node_test_ratio=0.0, edge_train_ratio=0.0, edge_test_ratio=0.0, node_split=False, edge_split=False, set_label_dis=False):
        if node_split:
            if set_label_dis:
                #same num of train in every label
                labels_items = nx.get_node_attributes(self.g, 'label').items()
                assert len(labels_items) == self.g.number_of_nodes(), 'some nodes without label information'
                labeled_node_set = {}
                #some data in citeseer do not have label, marked as -1
                for node, label in labels_items:
                    label = int(label)
                    if label not in labeled_node_set:
                        labeled_node_set[label] = []
                    labeled_node_set[label].append(node)
                train_node_set = []
                val_node_set = []
                test_node_set = []
                length = self.g.number_of_nodes()
                train_length = int(int(length * node_train_ratio) / len(labeled_node_set))
                #test_length = int(length * node_test_ratio)
                for label, node_list in labeled_node_set.items():
                    length = len(node_list)
                    assert length > train_length, 'train node expected is larger than exits nodes'
                    node_list = shuffle(node_list)
                    train_node_set += node_list[:train_length]
                    test_node_set += node_list[train_length:]
                    #val_node_set += node_list[train_length+test_length:]
                self._attach_node_label('train', train_node_set)
                self._attach_node_label('test', test_node_set)
                self._attach_node_label('val', val_node_set)
            else:
                nodes = self.g.nodes()
                nodes = shuffle(list(nodes))
                length = len(nodes)
                train_length = int(length * node_train_ratio)
                test_length = int(length * node_test_ratio)

                self._attach_node_label('train', nodes[:train_length])
                self._attach_node_label('test', nodes[train_length:train_length+test_length])
                self._attach_node_label('val', nodes[train_length+test_length:])

        if edge_split:
            edges = list(self.g.edges())
            #edges = np.random.permutation(edges)
            edges = shuffle(edges)
            length = len(edges)
            train_length = int(length * edge_train_ratio)
            test_length = int(length * edge_test_ratio)

            train_edges = edges[:train_length]
            test_edges = edges[train_length:train_length+test_length]
            val_edges = edges[train_length+test_length:]

            self._attach_edge_label('train_edge', train_edges)
            self._attach_edge_label('test_edge', test_edges)
            self._attach_edge_label('val_edge', val_edges)

            self.g.graph['train_edges'] = train_edges
            self.g.graph['val_edges'] = val_edges
            self.g.graph['test_edges'] = test_edges

            self._build_train_adj()

            self._sample_negative_edges(edges, train_edges, test_edges, val_edges, self.g.number_of_nodes())

        return

    def build_graph_data(self, adj, features=None, train_node_set=None, val_node_set=None, test_node_set=None, train_edge_set=None, val_edge_set=None, test_edge_set=None):
        #self.g = nx.from_numpy_matrix(adj)
        self.g = nx.from_scipy_sparse_matrix(adj)
        self.train_graph = self.g

        if features is None:
            self.features = None
        else:
            self._attach_features(features)
            self.features = features

        if train_node_set:
            self._attach_node_label('train', train_node_set)

        if val_node_set:
            self._attach_node_label('val', val_node_set)

        if test_node_set:
            self._attach_node_label('test', test_node_set)

        if train_edge_set:
            self._attach_edge_label('train_edge', train_edge_set)
            self._build_train_adj()

        if val_edge_set:
            self._attach_edge_label('val_edge', val_edge_set)

        if test_edge_set:
            self._attach_edge_label('test_edge', test_edge_set)

        return

    def get_ogb_edges(self):
        return self.ogb_row, self.ogb_col

    def check_symmetry_adj(self, row, col, n):
        edge_weights = np.ones_like(row)
        a = sp.csr_matrix((edge_weights, (row, col)), shape=(n, n))
        b = a + a.T
        if a.nnz == b.nnz:
            return True, a
        return False, b

    #load gcn format data
    def load_gcn_data(self, dataset_str, fix_node_test=False, node_label=False):
        self._data_name = dataset_str
        if dataset_str in ['ogbn-arxiv']:
            ogb_dataset = NodePropPredDataset(dataset_str)
            if len(ogb_dataset) > 1:
                print("Data set Error!!!")
                sys.exit(-1)
            graph, label = ogb_dataset[0]
            n = graph["num_nodes"]
            x = graph["node_feat"]
            row, col = graph["edge_index"]
            e = graph["edge_feat"]

            _, adj = self.check_symmetry_adj(row, col, n)
            adj = adj.tocoo()
            row = adj.row
            col = adj.col
            row, col = shuffle(row, col)

            self.ogb_row = row
            self.ogb_col = col

            label = np.squeeze(label)
            self.ogb_label_set = list(sorted(list(set(sorted(label)))))
            print(self.ogb_label_set)

            edge_weights = np.ones_like(row)
            self.ogb_label = label
            self.ogb_adj = sp.csr_matrix((edge_weights, (row, col)), shape=(n, n))
            self.ogb_x = x

            print("adj shape (%d, %d), number of values %d"%(self.ogb_adj.shape[0], self.ogb_adj.shape[1], self.ogb_adj.nnz))

            idx = ogb_dataset.get_idx_split()
            self.ogb_idx_tr, self.ogb_idx_va, self.ogb_idx_te = idx["train"], idx["valid"], idx["test"]

            return None

        elif dataset_str not in ['citeseer', 'cora', 'pubmed']:
            names = ['x', 'graph']
            objects = []
            for i in range(len(names)):
                with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
                    if sys.version_info > (3, 0):
                        objects.append(pkl.load(f, encoding='latin1'))
                    else:
                        objects.append(pkl.load(f))
            features, graph = tuple(objects)
            adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        else:
            # load the data: x, tx, allx, graph
            names = ['x', 'tx', 'allx', 'graph']
            objects = []
            for i in range(len(names)):
                with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
                    if sys.version_info > (3, 0):
                        objects.append(pkl.load(f, encoding='latin1'))
                    else:
                        objects.append(pkl.load(f))
                #objects.append(pkl.load(open("data/ind.{}.{}".format(dataset, names[i]))))
            x, tx, allx, graph = tuple(objects)
            test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
            test_idx_range = np.sort(test_idx_reorder)

            if dataset_str == 'citeseer':
                # Fix citeseer dataset (there are some isolated nodes in the graph)
                # Find isolated nodes, add them as zero-vecs into the right position
                test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
                tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
                tx_extended[test_idx_range-min(test_idx_range), :] = tx
                tx = tx_extended

            features = sp.vstack((allx, tx)).tolil()
            features[test_idx_reorder, :] = features[test_idx_range, :]
            adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))


    #fix partition used in gcn
        if fix_node_test:
            names = ['y', 'ty', 'ally']
            objects = []
            for i in range(len(names)):
                with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
                    if sys.version_info > (3, 0):
                        objects.append(pkl.load(f, encoding='latin1'))
                    else:
                        objects.append(pkl.load(f))
                pass
            y, ty, ally = tuple(objects)
            test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
            test_idx_range = np.sort(test_idx_reorder)
            if dataset_str == 'citeseer':
                test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
                ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
                ty_extended[test_idx_range-min(test_idx_range), :] = ty
                ty = ty_extended

            #To Do, check correctness
            #features = sp.vstack((allx, tx)).tolil()
            #features[test_idx_reorder, :] = features[test_idx_range, :]
            labels = np.vstack((ally, ty))
            labels[test_idx_reorder, :] = labels[test_idx_range, :]

            idx_test = test_idx_range.tolist()
            idx_train = range(len(y))
            idx_val = range(len(y), len(y)+500)

            self.build_graph_data(adj, features, train_node_set=idx_train, val_node_set=idx_val, test_node_set=idx_test)


            self._attach_node_class_label(labels)
        else:
            if node_label:
                self.node_label = True
            #load node label class information
                if dataset_str not in ['citeseer', 'cora', 'pubmed']:
                    names = 'y'
                    with open("data/ind.{}.{}".format(dataset_str, names), 'rb') as f:
                        if sys.version_info > (3, 0):
                            labels = pkl.load(f, encoding='latin1')
                        else:
                            labels = pkl.load(f)
                else:
                #y:train data label, ally:label and unlabel data(have true label but unseen in train set, semi-supervised node classification setting), ty:test label data
                    names = ['y', 'ty', 'ally']
                    objects = []
                    for i in range(len(names)):
                        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
                            if sys.version_info > (3, 0):
                                objects.append(pkl.load(f, encoding='latin1'))
                            else:
                                objects.append(pkl.load(f))
                        pass
                    y, ty, ally = tuple(objects)
                    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
                    test_idx_range = np.sort(test_idx_reorder)
                    if dataset_str == 'citeseer':
                        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
                        ty_extended[test_idx_range-min(test_idx_range), :] = ty
                        ty = ty_extended
                    labels = np.vstack((ally, ty))
                    labels[test_idx_reorder, :] = labels[test_idx_range, :]
                self.build_graph_data(adj, features)
                self._attach_node_class_label(labels)
            else:
                self.build_graph_data(adj, features)
        return self.g


