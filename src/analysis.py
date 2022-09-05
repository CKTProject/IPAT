import utils
from utils import Graphdata, NeiSampler, HierRankData, InductiveData, ClusterMiniBatchGenerator
from utils import ConfigerLoader
import networkx as nx
import numpy as np
import sys
from scipy.stats import pearsonr

#dataset = 'ppi'

#gdata = InductiveData('data', dataset)
#num_data, adj, feats, train_feats, test_feats, labels, train_data, val_data, test_data = gdata.get_data()


#G = nx.from_scipy_sparse_matrix(adj)
#NCC = nx.number_connected_components(G)
#print("number of connected in full graph:%d"%NCC)
#ret = utils.dfs_split(adj)
#print(ret)
#print("number of subgraph %d"%int(ret.max()))


#train_adj = adj[train_data, :][:, train_data]
#G = nx.from_scipy_sparse_matrix(train_adj)
#NCC = nx.number_connected_components(G)
#print("number of connected in train graph:%d"%NCC)
#ret = utils.dfs_split(train_adj)
#print(ret)
##print("number of subgraph %d"%int(ret.max()))

#label_dis = np.sum(labels, axis=0)
#print(label_dis.tolist())
#print((label_dis/np.sum(label_dis)).tolist())



#femb = 'emb/cora/DGI-512-debug.txt'
#embeddings = utils.load_embedding(femb, delimiter=' ', skiprows=0)

#res = []
#x, y = embeddings.shape

#print(embeddings[0])

#for i in range(x):
#	for j in range(y):
#		res.append([i, j, embeddings[i][j]])
#res = np.array(res)
#print(res)
#femb = 'emb/cora/DGI-512-test.csv'
#header = "%d %d"%(1, 1)
#utils.save_embedding(femb, res, delimiter=',', header=header)


#dataset = 'cora'
#train_label_ratio = 0.05
#test_label_ratio = 1 - 0.05
#gdata = Graphdata()
#gdata.load_gcn_data(dataset, fix_node_test=False, node_label=True)
#
#
#gdata.random_split_train_test(node_split=True, node_train_ratio=train_label_ratio, node_test_ratio=test_label_ratio, set_label_dis=True)
#trainX_indices, trainY = gdata.get_node_classification_data(name='train')
#valX_indices, valY = gdata.get_node_classification_data(name='val')
#testX_indices, testY = gdata.get_node_classification_data(name='test')
#
#print("load node classification data end")
#print("train data size " + str(len(trainX_indices)))
##print("val data size " + str(len(valX_indices)))
#print("test data size " + str(len(testX_indices)))



#pearson correlation calculation

data_name = sys.argv[1]
if data_name not in ['cora', 'citeseer', 'pubmed', 'wiki']:
    print("no that dataset! ERROR!")
    sys.exit(0)
gdata = Graphdata()
#gdata.load_gcn_data(data_name, fix_node_test=True)
gdata.load_gcn_data(data_name, fix_node_test=False, node_label=True)
_, Y = gdata.get_node_classification_data('all')
X = gdata.get_features()
A = gdata.get_adj('all')


print(Y.shape)
print(X.shape)
print(A.shape)


S_X = X.dot(X.T).reshape([1, -1]).toarray()
S_Y = Y.dot(Y.T).reshape([1, -1])
S_A = A.dot(A.T).reshape([1, -1]).toarray()
print(S_Y.shape)
print(S_X.shape)
print(S_A.shape)

S_X = np.squeeze(S_X)
S_Y = np.squeeze(S_Y)
S_A = np.squeeze(S_A)
print(S_Y.shape)
print(S_X.shape)
print(S_A.shape)


res_AX = pearsonr(S_A, S_X)
res_YA = pearsonr(S_Y, S_A)
res_XY = pearsonr(S_X, S_Y)

print(res_AX)
print(res_YA)
print(res_XY)




