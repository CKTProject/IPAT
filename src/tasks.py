from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
from pandas import DataFrame
import matplotlib.pyplot as plt

from sklearn import preprocessing



import sys

class Classifier():



    def __init__(self, clf_model, class_num, C=1.):     
        self.class_num = class_num
        self.C = C
        self.clf_model = clf_model
        self.clf = self._get_clf_model(clf_model)

    def _get_clf_model(self, clf_model):
        if clf_model == 'LR':
            clf = LR()
            params = {'penalty':'l2', 'dual':False, 'tol':0.0001, 'C':1., 'fit_intercept':True, 'intercept_scaling':1,
              'class_weight':None, 'random_state':None, 'solver':'lbfgs', 'max_iter':100, 'multi_class':'auto',
              'verbose':10, 'warm_start':False, 'n_jobs':1}
        elif clf_model == 'SVM':
            clf = SVC()
            params = {'C':self.C, 'kernel':'linear', 'degree':3, 'gamma':'auto', 'coef0':0.0, 'shrinking':True,
                  'probability':True, 'tol':0.001, 'cache_size':2000, 'class_weight':None, 'verbose':False,
                  'max_iter':-1, 'decision_function_shape':'ovr', 'random_state':None}
        else:
            print("No such classification model")
            sys.exit(-1)


        clf.set_params(**params)
        clf = OneVsRestClassifier(clf)
        return clf


    def train(self, X, Y):
        #if self.clf_model == 'SVM':
        #    X = preprocessing.scale(X)
        self.clf.fit(X, Y)

    def predict_k(self, X, top_k_list):
        #if self.clf_model == 'SVM':
        #    X = preprocessing.scale(X)
        Y = self.clf.predict_proba(X)
        pred_y = []
        for index, k in enumerate(top_k_list):
            top_k_args = np.argsort(Y[index])[-k:]
            y = np.zeros(self.class_num)
            y[top_k_args] = 1
            pred_y.append(y)
        return np.array(pred_y, dtype=np.int32)

    def evaluate(self, X, Y):
        top_k_list = np.sum(Y, axis=1, dtype=np.int32).tolist()
        pred_y = self.predict_k(X, top_k_list)
       # print(pred_y)
        result = {}
        averages = ['micro', 'macro']
        for avg in averages:
            result[avg] = f1_score(pred_y, Y, average=avg)
        return result

    def collect_error_nodes(self, X, Y, indx):
        top_k_list = np.sum(Y, axis=1, dtype=np.int32).tolist()
        pred_y = self.predict_k(X, top_k_list)

        error_pred = np.multiply(1 - Y, pred_y)
        error_pred = np.sum(error_pred, axis=1)

        error_indx = indx[error_pred>0]
        return error_indx

class Cluster():

    def __init__(self, cluster_model, n_clusters=5):
        self.cluster = self._get_cluster_model(cluster_model, n_clusters)

    def _get_cluster_model(self, cluster_model, n_clusters):
        if cluster_model == 'KMeans':
            cluster = KMeans(n_clusters=n_clusters, random_state=0)
        else:
            print("No such cluster!")
            sys.exit(-1)

        return cluster

    def train(self, X):
        self.cluster.fit(X)

    def predict(self, X=None):
        if X is None:
            return self.cluster.labels_
        else:
            return self.cluster.predict(X)


class LinkPredictor():

    def _sigmoid(self, x):
        return 1 / (1 + np.math.exp(-x))

    def __init__(self, lp_model):
        self.lp_model = lp_model
        self.lp = self._get_lp_model(lp_model)

    def _get_lp_model(self, lp_model):
        if lp_model == 'clf':
        	lp = LR()
        	params = {'penalty':'l2', 'dual':False, 'tol':0.0001, 'C':0.5, 'fit_intercept':True,
            'intercept_scaling':1, 'class_weight':None, 'random_state':None, 'solver':'lbfgs',
            'max_iter':1000, 'multi_class':'multinomial', 'verbose':0, 'warm_start':False, 'n_jobs':1}
        elif lp_model == 'simoid':
        	lp = lambda x:x
        else:
        	print("No such model  !!!")
        return lp

    def train(self, Xl, Xr, Y):
        if self.lp_model == 'sigmoid':
            return
        #X = np.concatenate((Xl, Xr), axis=1)
        X = np.multiply(Xl, Xr)
        self.lp.fit(X, Y)

    def predict(self, Xl, Xr):
        if self.lp_model == 'sigmoid':
            prob = self._sigmoid(np.sum(Xl * Xr, axis=1))
            return prob
        #X = np.concatenate((Xl, Xr), axis=1)
        X = np.multiply(Xl, Xr)
        prob = self.lp.predict(X)
        return prob

    def evaluate(self, Xl, Xr, Y):
        pred_y = self.predict(Xl, Xr)
        auc = roc_auc_score(Y, pred_y)
        ap = average_precision_score(Y, pred_y)
        return auc, ap

class Visualizer():

    def __init__(self, v_model, n_components=2):
        self.v_model = v_model
        if v_model == 'tsne':
            self.vis = TSNE(n_components=n_components)
        else:
            self.vis = lambda x:x

    def plot2D(self, X, Y=None):
        if self.v_model == 'tsne':
            self.emb = self.vis.fit_transform(X).astype(float)
        else:
            self.emb = X

        if Y is None:
            Y = np.zeros(np.shape(self.emb)[0])

        color_num = len(set(sorted(Y.tolist())))

        #sys.exit(0)

        data = {'tsne-2d-one':self.emb[:,0], 'tsne-2d-two':self.emb[:,1], 'y':Y}
        df = DataFrame(data=data)
        ax = sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue="y",
            palette=sns.color_palette("hls", color_num),
            data=df,
            legend="full",
            alpha=0.3
        )
        plt.show()
        #ax.plot()







if __name__ == '__main__':
    emb = np.random.random((10, 30))

    #test node classifiersi
    '''
    labels = np.array([0, 0, 0, 1, 0, 1, 1, 1, 1, 0])
    labels = np.reshape(labels, (-1, 1))
    mlb = MultiLabelBinarizer(classes=[0, 1], sparse_output=False)
    Y = mlb.fit_transform(labels)
    clf = Classifier('LR', 2)
    clf.train(emb, Y)
    print(clf.evaluate(emb, Y))
    top_k_list = np.sum(Y, axis=1, dtype=np.int32).tolist()
    print(clf.predict_k(emb, top_k_list))
    '''

    #test link prediction
    '''
    pos_edges = np.random.random_integers(low=0, high=9, size=(10, 2))
    neg_edges = np.random.random_integers(low=0, high=9, size=(10, 2))
    edges = np.concatenate((pos_edges, neg_edges), axis=0)
    node_list = np.split(edges, 2, axis=1)
    le = np.squeeze(node_list[0])
    re = np.squeeze(node_list[1])
    Xl = emb[le]
    Xr = emb[re]
    Y = np.concatenate((np.ones(10), np.zeros(10)))
    lp = LinkPredictor('clf')
    lp.train(Xl, Xr, Y)
    print(lp.evaluate(Xl, Xr, Y))
    print(lp.predict(Xl, Xr))
    '''

    #test cluster
    '''
    X = emb
    cluster = Cluster('KMeans', 3)
    cluster.train(X)
    print(cluster.predict())
    print(cluster.predict(X))
    '''

    #test visualization
    X = emb
    visualizer = Visualizer('tsne', 2)
    visualizer.plot2D(X)
    Y = np.array([0, 0, 0, 1, 0, 1, 1, 1, 1, 0])
    visualizer.plot2D(X,Y)











