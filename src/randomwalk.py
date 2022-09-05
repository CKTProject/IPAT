import random
import pickle as pkl
import networkx as nx
import sys
import numpy as np

def save_path_corpus(fname, corpus):
    with open(fname, 'wb') as fout:
        pkl.dump(corpus, fout)

def load_path_corpus(fname):
    with open(fname, 'rb') as fin:
        return pkl.load(fin)

def set_alias(prob):
    K = len(prob)

    high = []
    low = []

    jump = np.zeros(K, dtype=np.int32)
    pp = np.zeros(K)

    for indx, p in enumerate(prob):
        pp[indx] = K * p
        if pp[indx] > 1.0:
            high.append(indx)
        else:
            low.append(indx)
    while len(high) > 0 and len(low) > 0:
        large = high.pop()
        small = low.pop()

        pp[large] = pp[large] - (1.0 - pp[small])
        jump[small] = large

        if pp[large] >= 1.0:
            high.append(large)
        else:
            low.append(large)
    return jump, pp


def draw_alias(jump, prob):
    K = len(prob)
    x = int(np.floor(random.random() * K))
    y = random.random()

    if y < prob[x]:
        return x
    else:
        return jump[x]





class RandomWalk():
    def __init__(self, G, method='unbias', alpha=0.0, p=1.0, q=1.0, is_directed=False, weighted=False):
        self.G = G
        self.is_directed = is_directed
        self.p = p
        self.q = q
        self.alpha = alpha
        self.method = method
        #self.G.remove_edges_from(G.selfloop_edges())
        if not weighted:
            self._reweighted_graph()
        if method == 'bias':
            self._preprocess_trans_prob()
        #self._preprocess_trans_prob()

    def _reweighted_graph(self):
        edges = self.G.edges()
        w = np.ones(len(edges)).tolist()
        x, y = zip(*edges)
        x = list(x)
        y = list(y)
        new_edge = zip(x,y,w)
        self.G.add_weighted_edges_from(new_edge)

    def _preprocess_trans_prob(self):
        self.alias_nodes = {}
        for node in self.G.nodes():
            trans_prob = [self.G[node][n]['weight'] for n in sorted(self.G.neighbors(node))]
            norm_const = np.sum(trans_prob, dtype=np.float32)
            norm_trans_prob = [p/norm_const for p in trans_prob]
            self.alias_nodes[node] = set_alias(norm_trans_prob)

        self.alias_edges = {}
        for edge in self.G.edges():
            if self.is_directed:
                self.alias_edges[edge] = self._get_alias_edges(edge)
            else:
                self.alias_edges[edge] = self._get_alias_edges(edge)
                self.alias_edges[(edge[1], edge[0])] = self._get_alias_edges((edge[1], edge[0]))

    def _get_alias_edges(self, edge):
        star, end = edge[0], edge[1]
        unnorm_prob = []
        for node in sorted(self.G.neighbors(end)):
            if node == star:
                unnorm_prob.append(self.G[end][node]['weight'] / self.p)
            elif self.G.has_edge(node, star):
                unnorm_prob.append(self.G[end][node]['weight'])
            else:
                unnorm_prob.append(self.G[end][node]['weight'] / self.q)

        norm_const = np.sum(unnorm_prob, dtype=np.float32)
        norm_prob = [p/norm_const for p in unnorm_prob]
        return set_alias(norm_prob)

    def unbias_walk(self, walk_length, seed_node):
        path = [seed_node]

        while len(path) < walk_length:
            cur = path[-1]
            #neighbors = list(self.G[cur])
            neighbors = list(self.G.neighbors(cur))
            if len(neighbors) > 0:
                if random.random() > self.alpha:
                    path.append(random.choice(neighbors))
                    #sampled_indx = draw_alias(self.alias_nodes[cur][0], self.alias_nodes[cur][1])
                    #path.append(neighbors[sampled_indx])
                else:
                    path.append(path[0])
            else:
                break
        if len(path) < walk_length:
            #print("path len smaller %d, padding with last node %d"%(seed_node, path[-1]))
            pass
        while len(path) < walk_length:
            path.append(path[-1])
        return path

    def bias_walk(self, walk_length, seed_node):
        path = [seed_node]

        while len(path) < walk_length:
            cur = path[-1]
            cur_neighbor = sorted(self.G.neighbors(cur))
            if len(cur_neighbor) < 1:
                break
            if len(path) == 1:
                sampled_indx = draw_alias(self.alias_nodes[cur][0], self.alias_nodes[cur][1])
                path.append(cur_neighbor[sampled_indx])
            else:
                prev = path[-2]
                sampled_indx = draw_alias(self.alias_edges[(prev, cur)][0], self.alias_edges[(prev, cur)][1])
                path.append(cur_neighbor[sampled_indx])
        if len(path) < walk_length:
            #print("path len smaller %d, padding with last node %d"%(seed_node, path[-1]))
            pass
        while len(path) < walk_length:
            path.append(path[-1])
        return path


    def simulate_walk(self, number_of_walks, walk_length, seed_nodes):
        path_corpus = []
        if self.method == 'unbias':
            for seed in seed_nodes:
                tmp_list = []
                for i in range(number_of_walks):
                    tmp_list.append(self.unbias_walk(walk_length, seed))
                path_corpus.append(tmp_list)
        elif self.method == 'bias':
            for seed in seed_nodes:
                tmp_list = []
                for i in range(number_of_walks):
                    tmp_list.append(self.bias_walk(walk_length, seed))
                path_corpus.append(tmp_list)
        else:
            print("simulation method error, only bias and unbias supported")
        return path_corpus


def path_to_input(corpus, ws):
    '''corpus:randomwalk path list, ws:window size'''
    X = []
    Y = []
    for path in corpus:
        for i, w in enumerate(path):
            l = max(0, i-ws)
            r = min(i+ws+1, len(path))
            labels = path[l:i] + path[i+1:r]
            for y in labels:
                X.append(w)
                Y.append(y)
    Y = np.reshape(np.array(Y), (-1, 1))
    return np.array(X), np.array(Y)

    pass
if __name__ == '__main__':
    G = nx.karate_club_graph()
    randomwalk = RandomWalk(G, 'unbias')
    corpus = randomwalk.simulate_walk(10, 10, [0, 5])
    for p in corpus:
        print(p)
    X, Y = path_to_input(p, 3)
    print(X)
    print(Y)
    #bias_randomwalk = RandomWalk(G, 'bias')
    #corpus = bias_randomwalk.simulate_walk(10, 10, [0, 5])
    #for p in corpus:
    #    print(p)


