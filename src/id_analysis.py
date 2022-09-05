import numpy as np 
import utils
from utils import Graphdata
from sklearn.manifold import TSNE

from sklearn.preprocessing import normalize

import matplotlib.pyplot as plt
import sys


def plot_augment_data_nb_dis(select_indx, dataset_str, emb_dimension, model_str):
    gdata = Graphdata()
    gdata.load_gcn_data(dataset_str, fix_node_test=True)
    _, labelY = gdata.get_node_classification_data(name='all')

    femb = 'emb/idvisual/%s/%s-%d.txt'%(dataset_str, model_str, emb_dimension)
    embeddings = utils.load_embedding(femb, delimiter=' ')
    print("load embedding end, emb shape " + str(embeddings.shape))

    femb = 'emb/idvisual/%s/%s-%d-aug.txt'%(dataset_str, model_str, emb_dimension)
    aug_embeddings = utils.load_embedding(femb, delimiter=' ')
    print("load embedding end, emb shape " + str(aug_embeddings.shape))

    #adj = gdata.get_adj('all')
    #_, str_nb_index = utils.get_value_and_index_of_spmatrxi_by_row(adj, select_indx)
    #print(str_nb_index)

    str_nb_index = np.random.choice(range(2708), size=4)

    #selected node
    #select_indx = 1982
    #str_nb_index = [172, 1024, 2633]
    #select_indx=1600
    #str_nb_index = [ 504, 1040, 2534, 2429]
    #228
    #[ 355 1298 2593 1281]
    select_indx=2090
    str_nb_index =[1837, 2042, 2401, 136]

    print("***********current nodes*********")
    print(select_indx)
    print(str_nb_index)
    print("*********************************")

    target_emb = embeddings[select_indx,:]
    target_emb = np.reshape(target_emb, (1, -1))
    aug_emb = aug_embeddings[select_indx,:]
    aug_emb = np.reshape(aug_emb, (1, -1))

    
    nb_index = list(str_nb_index)
    nb_emb = embeddings[nb_index,:]
    aug_nb_emb = aug_embeddings[nb_index,:]
    print(nb_emb.shape)
    print(aug_nb_emb.shape)

    labels = [0, 1] + [2] * len(nb_index) + [3]*len(nb_index)

    method = TSNE(n_components=2, init="pca", random_state=0)

    X = np.concatenate((target_emb, aug_emb, nb_emb, aug_nb_emb), axis=0)
    Y = method.fit_transform(X)

    print(Y)
    print(Y.shape)

    colors = np.array(["red", "green", "blue", "orange", "brown", "purple"])
    #markers = ['^', 's'] + ['x']*4

    #plt.scatter(Y[:, 0], Y[:, 1], c=colors[labels], s=100, marker=markers)
    plt.scatter(Y[0, 0], Y[0, 1], c=colors[labels[0]], s=100, marker='s')
    plt.scatter(Y[1, 0], Y[1, 1], c=colors[labels[0]], s=100, marker='^')
    for i in range(len(nb_index)):
        plt.scatter(Y[2+i, 0], Y[2+i, 1], c=colors[min(len(colors)-1, i+1)], s=100, marker='s')
        plt.scatter(Y[2+len(nb_index)+i, 0], Y[2+len(nb_index)+i, 1], c=colors[min(len(colors)-1, i+1)], s=100, marker='^')
    plt.show()



def plot_augment_data_nb(select_indx, dataset_str, emb_dimension, model_str):
    gdata = Graphdata()
    gdata.load_gcn_data(dataset_str, fix_node_test=True)
    _, labelY = gdata.get_node_classification_data(name='all')

    femb = 'emb/idvisual/%s/%s-%d.txt'%(dataset_str, model_str, emb_dimension)
    embeddings = utils.load_embedding(femb, delimiter=' ')
    print("load embedding end, emb shape " + str(embeddings.shape))

    femb = 'emb/idvisual/%s/%s-%d-aug.txt'%(dataset_str, model_str, emb_dimension)
    aug_embeddings = utils.load_embedding(femb, delimiter=' ')
    print("load embedding end, emb shape " + str(aug_embeddings.shape))

    adj = gdata.get_adj('all')
    _, str_nb_index = utils.get_value_and_index_of_spmatrxi_by_row(adj, select_indx)
    print(str_nb_index)


    k=20
    target_emb = embeddings[str_nb_index,:]
    target_emb = np.reshape(target_emb, (1, -1))

    score = np.squeeze(np.dot(normalize(target_emb), normalize(embeddings).T))
    #print(score)
    topk = np.argsort(score)[-k:-1]
    print(topk)
    topk_set = set(topk)

    str_nb_index_set = set(str_nb_index)
    topk_set = set(topk_set)
    only_top_set = topk_set - str_nb_index_set
    print(only_top_set)


    struct_nb = list(str_nb_index_set)
    emb_nb = list(only_top_set)
    nb_index = struct_nb + emb_nb


    aug_emb = aug_embeddings[select_indx,:]
    aug_emb = np.reshape(aug_emb, (1, -1))
    

    neigh_emb = embeddings[nb_index,:]
    print(neigh_emb.shape)


    labels = [0, 0] + [1] * len(struct_nb) + [2]*len(emb_nb)

    method = TSNE(n_components=2, init="pca", random_state=0)

    X = np.concatenate((target_emb, aug_emb, neigh_emb), axis=0)
    Y = method.fit_transform(X)

    print(Y)
    print(Y.shape)

    colors = np.array(["red", "green", "blue"])
    #markers = ['^', 's'] + ['x']*4

    #plt.scatter(Y[:, 0], Y[:, 1], c=colors[labels], s=100, marker=markers)
    plt.scatter(Y[0, 0], Y[0, 1], c=colors[labels[0]], s=100, marker='^')
    plt.scatter(Y[1, 0], Y[1, 1], c=colors[labels[1]], s=100, marker='s')
    plt.scatter(Y[2:, 0], Y[2:, 1], c=colors[labels[2:]], s=100, marker='x')

    plt.show()

def plot_augment_data(select_indx, dataset_str, emb_dimension, model_str):
    gdata = Graphdata()
    gdata.load_gcn_data(dataset_str, fix_node_test=True)
    _, labelY = gdata.get_node_classification_data(name='all')

    femb = 'emb/idvisual/%s/%s-%d.txt'%(dataset_str, model_str, emb_dimension)
    embeddings = utils.load_embedding(femb, delimiter=' ')
    print("load embedding end, emb shape " + str(embeddings.shape))

    femb = 'emb/idvisual/%s/%s-%d-aug.txt'%(dataset_str, model_str, emb_dimension)
    aug_embeddings = utils.load_embedding(femb, delimiter=' ')
    print("load embedding end, emb shape " + str(aug_embeddings.shape))


    k=30
    target_emb = embeddings[select_indx,:]
    target_emb = np.reshape(target_emb, (1, -1))

    score = np.squeeze(np.dot(normalize(target_emb), normalize(embeddings).T))
    print(score)
    topk = np.argsort(score)[-k:-1]
    print(topk)

    aug_emb = aug_embeddings[select_indx,:]
    aug_emb = np.reshape(aug_emb, (1, -1))
    

    neigh_emb = embeddings[topk,:]
    print(neigh_emb.shape)

    total_indx =  [select_indx, select_indx] + topk.tolist()
    labels = np.argmax(labelY[total_indx], axis=1)

    #labels = np.argmax(labelY[topk], axis=1)
    #labels = np.array([9, 9] + labels.tolist())

    method = TSNE(n_components=2, init="pca", random_state=0, perplexity=10)

    X = np.concatenate((target_emb, aug_emb, neigh_emb), axis=0)
    Y = method.fit_transform(X)

    print(Y)
    print(Y.shape)

    colors = np.array(["orange","olive","purple","red", "blue", "green", "gray","pink", "brown", "cyan" ])
    #markers = ['^', 's'] + ['x']*4

    #plt.scatter(Y[:, 0], Y[:, 1], c=colors[labels], s=100, marker=markers)
    plt.scatter(Y[0, 0], Y[0, 1], c=colors[labels[0]], s=100, marker='^')
    plt.scatter(Y[1, 0], Y[1, 1], c=colors[labels[1]], s=100, marker='s')
    plt.scatter(Y[2:, 0], Y[2:, 1], c=colors[labels[2:]], s=100, marker='x')

    plt.show()





fname = 'emb/idvisual/cora/FixR-512-error.txt'
print(fname)
a = set(np.loadtxt(fname).tolist())

fname = 'emb/idvisual/cora/FowardR-512-error.txt'
b = set(np.loadtxt(fname).tolist())
print(a - b)
c = a - b
c = list(c)


select_indx = int(c[5])
print(select_indx)

#res = [c[11], c[1], c[4], c[22], c[24]]
#res = [c[22]]
#res = [1799]
res = np.random.choice(range(2708), size=5)

#res = np.random.choice(c, size=5)


#print(res)

#backup
#res = [2584]
#res = [153]

for select_indx in res:
    select_indx = int(select_indx)
    dataset_str = 'cora'
    emb_dimension = 512
    model_str = 'FixR'
    print("current select index %d"%select_indx)
    #plot_augment_data(select_indx, dataset_str, emb_dimension, model_str)
    #plot_augment_data_nb(select_indx, dataset_str, emb_dimension, model_str)
    plot_augment_data_nb_dis(select_indx, dataset_str, emb_dimension, model_str)

    model_str = 'FowardR'
    #plot_augment_data(select_indx, dataset_str, emb_dimension, model_str)
    #plot_augment_data_nb(select_indx, dataset_str, emb_dimension, model_str)
    plot_augment_data_nb_dis(select_indx, dataset_str, emb_dimension, model_str)








