from layers import GraphConvolution, InnerProductDecoder
import tensorflow as tf
from tensorflow import keras
from utils import gumbel_softmax
from utils import unit_sphere_sample
import numpy as np
from sklearn import preprocessing
from sklearn.utils import shuffle
import sys

class EarlyStoppingAtMinLoss(keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.
    Arguments:
        patience: Number of epochs to wait after min has been hit. After this
        number of no improvement, decrease learning rate.
        lr_patience: After the number of decreasing learning rate, early stop!

    """
    def __init__(self, lr_patience=0, patience=0):
        super(EarlyStoppingAtMinLoss, self).__init__()
        self.patience = patience
        self.lr_patience = lr_patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The number of learning rate decreasing,
        self.lr_dec = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("loss")
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                if self.lr_dec >= self.lr_patience:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
                    print("Restoring model weights from the end of the best epoch.")
                    self.model.set_weights(self.best_weights)
                else:
                    if not hasattr(self.model.optimizer, "lr"):
                        raise ValueError('Optimizer must have a "lr" attribute.')
                    # Get the current learning rate from model's optimizer.
                    lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
                    # Set the value back to the optimizer before this epoch starts
                    scheduled_lr = lr/10.0
                    tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
                    print("\nEpoch %05d: Learning rate is %6.4f." % (epoch, scheduled_lr))


    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))



class LossAndErrorPrintingCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(
            "The average loss for epoch {} is {:7.5f} ".format(
                epoch, logs["loss"]
            )
        )





class DAODNN(keras.layers.Layer):
    """DNN simulates graph data augmentation learning from data"""

    def __init__(self, units_list, act=tf.nn.relu, dropout=0., bias=False, l2_r=1e-5, **kwargs):
        super(DAODNN, self).__init__(**kwargs)
        #self.input_dim = input_dim
        self.units_list = units_list
        self.dropout = dropout
        self.l2_r = l2_r

        self.layers_list = []

        for i, units in enumerate(units_list):
            self.layers_list.append(keras.layers.Dense(units=units,
                                                           #activation=act,
                                                           activation = tf.keras.layers.PReLU(),
                                                           use_bias=bias,
                                                           kernel_initializer='glorot_uniform',
                                                           bias_initializer='zeros',
                                                           kernel_regularizer=tf.keras.regularizers.l2(self.l2_r),
                                                           bias_regularizer=tf.keras.regularizers.l2(self.l2_r),
                                                           name="DAO_%s"%i,
                                                           ))

    def call(self, inputs, training=None):
        x = inputs
        for layer in self.layers_list:
            if training:
                x = tf.nn.dropout(x, self.dropout)
            x = layer(x)
        return x

class GraphEncoder(keras.layers.Layer):
    """Graph encoder layer to get node representation. support GCN layer, TODO add GAT layer, add layer normalization"""

    def __init__(self, input_dim, units_list, dropout=0., sparse_inputs=False,
                act=tf.nn.relu, bias=False, l2_r=0., **kwargs):
        super(GraphEncoder, self).__init__(**kwargs)
        self.units_list = units_list
        self.dropout = dropout
        self.sparse_inputs = sparse_inputs
        self.l2_r = l2_r
        self.bias = bias
        self.act = act

        self.layers_list = []
        in_dim = input_dim
        s_input = self.sparse_inputs

        for i, out_dim in enumerate(units_list):
            self.layers_list.append(GraphConvolution(input_dim=in_dim,
                                                     output_dim=out_dim,
                                                     dropout=self.dropout,
                                                     sparse_inputs=s_input,
                                                     l2_r=self.l2_r,
                                                     bias=self.bias,
                                                     act=self.act)
                                    )
            s_input = False
            in_dim = out_dim
        #self.layers_list.append(GraphConvolution(input_dim=in_dim,
        #                                             output_dim=units_list[-1],
        #                                             dropout=self.dropout,
        #                                             sparse_inputs=s_input,
        #                                             l2_r=self.l2_r,
        #                                             bias=self.bias,
        #                                             act=lambda x:x))

    def call(self, inputs, training=None):
        x, adj = inputs
        for layer in self.layers_list:
            x = layer((x, adj), training)
        return x

class DAONoiseOperator():
    """generate operator type inputs for DAODNN"""
    def __init__(self, dao_categories_num=5, dao_continue_maxval=1, dao_continue_minval=-1, noise_mean=0., noise_stddev=1.0, noise_dim=10):
        self.dao_categories_num = dao_categories_num
        self.dao_continue_minval = dao_continue_minval
        self.dao_continue_maxval = dao_continue_maxval
        self.noise_mean = noise_mean
        self.noise_stddev = noise_stddev
        self.noise_dim = noise_dim

        self.dao_categories_enc = preprocessing.LabelBinarizer()
        self.dao_categories_enc.fit(np.array(range(self.dao_categories_num)))

    def __call__(self, sample_num):
        continue_inputs = np.random.uniform(low=self.dao_continue_minval, high=self.dao_continue_maxval, size=(sample_num, 1))
        noise_inputs = np.random.normal(loc=self.noise_mean, scale=self.noise_stddev, size=(sample_num, self.noise_dim))
        categories_inputs = np.random.randint(low=0, high=self.dao_categories_num, size=(sample_num, 1))
        categories_inputs = self.dao_categories_enc.transform(categories_inputs)
        dao_inputs = np.concatenate((continue_inputs, categories_inputs, noise_inputs), axis=1)
        return dao_inputs

class GraphCLR(keras.Model):
    """Graph contrastive learning with data augmentation operator learned from dataset"""

    def __init__(self, input_dim, gnn_units_list, dnn_units_list, sparse_inputs=False, act=tf.nn.relu, dropout=0., bias=False, l2_r=1e-5,
                 dao_categories_num=5, dao_continue_maxval=1, dao_continue_minval=-1, noise_mean=0., noise_stddev=1.0, noise_dim=10,
                 negative_num=10, aug_dgi_loss_w=1e-5, ins_loss_w=1e-5, learning_rate=0.01, temperature=0.07, norm_loss_w=1e-3, augment_num=10,
                 hinge_loss_w=0.,
                 **kwargs):
        super(GraphCLR, self).__init__(**kwargs)

        self.aug_dgi_loss_w = aug_dgi_loss_w
        self.ins_loss_w = ins_loss_w
        self.temperature = temperature
        self.norm_loss_w = norm_loss_w
        self.augment_num = augment_num
        self.hinge_loss_w = hinge_loss_w

        self.ado_op = DAODNN(units_list=dnn_units_list, act=act, dropout=dropout, bias=True, l2_r=0.)

        self.noise_op = DAONoiseOperator(dao_categories_num=dao_categories_num,
                                         dao_continue_maxval=dao_continue_maxval,
                                         dao_continue_minval=dao_continue_minval,
                                         noise_mean=noise_mean, noise_stddev=noise_stddev,
                                         noise_dim=noise_dim)

        self.graph_enc = GraphEncoder(input_dim=input_dim,
                                      units_list=gnn_units_list,
                                      dropout=dropout,
                                      sparse_inputs=sparse_inputs,
                                      act=act,
                                      l2_r=0.,
                                      bias=True
                                      )

        self.emb_dim = gnn_units_list[-1]

        #dscriminator in DGI used to classify <node, graph>, <corrup_node, graph>
        self.bi_weights = self.add_weight(
            shape=(self.emb_dim, self.emb_dim),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True
        )

        self.cosine = tf.keras.metrics.CosineSimilarity()

        self.negative_num = negative_num
        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        #self.optimizer = tf.keras.optimizer.RMSprop(learning_rate=self.learning_rate)



    def graph_readout(self, node_representation):
        graph_representation = tf.reshape(tf.math.sigmoid(tf.math.reduce_mean(node_representation, axis=0)), shape=[1, -1])
        return graph_representation

    def DGI_loss_fn(self, h, corp_h, c):
        pos_logits = tf.reshape(tf.matmul(tf.matmul(h, self.bi_weights), tf.transpose(c)), shape=[h.shape[0], 1])
        neg_logits = tf.reshape(tf.matmul(tf.matmul(corp_h, self.bi_weights), tf.transpose(c)), shape=[h.shape[0], 1])
        logits = tf.concat([pos_logits, neg_logits], axis=0)
        labels = np.concatenate((np.ones(shape=(h.shape[0], 1)), np.zeros(shape=(h.shape[0], 1))), axis=0)
        loss = keras.losses.binary_crossentropy(labels, logits, from_logits=True)
        loss = tf.math.reduce_mean(loss)
        return loss

    def instance_classify_loss_sig_fn(self, aug_h, h, neg_h):
        pos = tf.reshape(tf.math.reduce_sum(tf.math.multiply(aug_h, h), axis=1), shape=(-1, 1))
        aug_h = tf.reshape(tf.tile(aug_h, [1, self.negative_num]), shape=[h.shape[0], self.negative_num, h.shape[1]])
        neg = tf.math.reduce_sum(tf.math.multiply(aug_h, neg_h), axis=2)
        logits = tf.concat((pos, neg), axis=1)
        ones = np.ones(shape=(h.shape[0], 1), dtype=np.float32)
        zeros = np.zeros(shape=(h.shape[0], self.negative_num), dtype=np.float32)
        labels = np.concatenate((ones, zeros), axis=1)
        loss = tf.math.reduce_mean(tf.math.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits), axis=1))
        return loss

    def instance_hinge_loss_fn(self, aug_h, h, neg_h):
        pos = tf.reshape(tf.math.reduce_sum(tf.math.multiply(aug_h, h), axis=1), shape=(-1, 1))
        aug_h = tf.reshape(tf.tile(aug_h, [1, self.negative_num]), shape=[h.shape[0], self.negative_num, h.shape[1]])
        neg = tf.math.reduce_sum(tf.math.multiply(aug_h, neg_h), axis=2)

        loss = tf.math.reduce_mean(tf.math.reduce_sum(tf.math.maximum(0, neg-pos), axis=-1), axis=-1)
        return loss

    def run_link_logits(self, inputs, val_pos_edges, val_neg_edges):
        x, adj = inputs
        self.h = self.graph_enc((x, adj))
        pos_node_a, pos_node_b = tf.split(tf.nn.embedding_lookup(self.h, val_pos_edges), 2, axis=1)
        neg_node_a, neg_node_b = tf.split(tf.nn.embedding_lookup(self.h, val_neg_edges), 2, axis=1)
        pos_node_a = tf.squeeze(pos_node_a)
        pos_node_b = tf.squeeze(pos_node_b)
        neg_node_a = tf.squeeze(neg_node_a)
        neg_node_b = tf.squeeze(neg_node_b)

        pos_logits = tf.math.sigmoid(tf.math.reduce_sum(tf.math.multiply(pos_node_a, pos_node_b), axis=-1))
        neg_logits = tf.math.sigmoid(tf.math.reduce_sum(tf.math.multiply(neg_node_a, neg_node_b), axis=-1))
        logits = tf.concat((pos_logits, neg_logits), axis=0)

        return logits

    def call(self, inputs, training=None):
        x, adj, corp_inputs, corp_adj, negative_index = inputs

        node_num = x.dense_shape[0]

        #step in trainin

        #get node representation from deep GNN and graph representaion from readout function
        self.h = self.graph_enc((x, adj), training)
        #self.h = tf.keras.utils.normalize(self.h)
        self.c = self.graph_readout(self.h)

        #corrupt graph and get embedding of corrupt graph
        #corp_inputs = shuffle(x)
        #corp_adj = adj
        #corp_adj = shuffle(adj)

        self.corp_h = self.graph_enc((corp_inputs, corp_adj), training)
        #self.corp_h = tf.keras.utils.normalize(self.corp_h)
        #self.corp_c = self.graph_readout(self.corp_h)

        self.dgi_loss = self.DGI_loss_fn(self.h, self.corp_h, self.c)

        #get augmentation samples using DAO operator
        aug_noise = self.noise_op(node_num)
        aug_list = []
        self.aug_dgi_loss = 0.
        self.instance_loss = 0.
        self.hinge_loss = 0.
        for i in range(self.augment_num):
            aug_h = self.ado_op(np.concatenate((self.h, aug_noise), axis=1))
            aug_list.append(aug_h)
            self.aug_dgi_loss += self.DGI_loss_fn(aug_h, self.corp_h, self.c) * self.aug_dgi_loss_w

            negative_samples = tf.nn.embedding_lookup(params=self.h, ids=negative_index)
            self.instance_loss += self.instance_classify_loss_sig_fn(aug_h, self.h, negative_samples) * self.ins_loss_w

            self.hinge_loss += self.instance_hinge_loss_fn(aug_h, self.h, negative_samples) * self.hinge_loss_w


        #aug_h = tf.concat(aug_list, axis=0)
        aug_h = tf.add_n(aug_list) / self.augment_num



        self.norm_loss = tf.math.reduce_mean(tf.math.square(tf.nn.l2_loss(self.h) - tf.nn.l2_loss(aug_h))) * self.norm_loss_w


        loss = self.dgi_loss + self.aug_dgi_loss + self.instance_loss + self.hinge_loss + self.norm_loss
        self.add_loss(loss)

        total_loss = tf.add_n(self.losses)
        loss = (total_loss, self.dgi_loss, self.aug_dgi_loss, self.instance_loss, self.hinge_loss, self.norm_loss)
        #h_list = (self.h, aug_h)

        return loss, self.h, aug_h


class GraphGaussianCLR(keras.Model):
    """Graph contrastive learning with data augmentation operator learned from dataset"""

    def __init__(self, input_dim, gnn_units_list, sparse_inputs=False, act=tf.nn.relu, dropout=0., bias=False, l2_r=1e-5,
                 negative_num=10, aug_dgi_loss_w=1e-5, ins_loss_w=1e-5, learning_rate=0.01, temperature=0.07, augment_num=10,
                 norm_loss_w=-0.1, hinge_loss_w=0.,
                 **kwargs):
        super(GraphGaussianCLR, self).__init__(**kwargs)

        self.aug_dgi_loss_w = aug_dgi_loss_w
        self.ins_loss_w = ins_loss_w
        self.temperature = temperature
        self.augment_num = augment_num

        self.norm_loss_w = norm_loss_w
        self.hinge_loss_w = hinge_loss_w

        self.graph_enc = GraphEncoder(input_dim=input_dim,
                                      units_list=gnn_units_list,
                                      dropout=dropout,
                                      sparse_inputs=sparse_inputs,
                                      act=act,
                                      l2_r=0.,
                                      bias=True
                                      )

        self.emb_dim = gnn_units_list[-1]


        #self.std_enc = keras.layers.Dense(units=gnn_units_list[-1],
        self.std_enc = keras.layers.Dense(1,
                                                           activation = tf.nn.relu,
                                                           use_bias=True,
                                                           kernel_initializer='glorot_uniform',
                                                           bias_initializer='zeros',
                                                           kernel_regularizer=tf.keras.regularizers.l2(l2_r),
                                                           bias_regularizer=tf.keras.regularizers.l2(l2_r),
                                                           name="DAO_std",
                                                           )

        #dscriminator in DGI used to classify <node, graph>, <corrup_node, graph>
        self.bi_weights = self.add_weight(
            shape=(self.emb_dim, self.emb_dim),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True
        )


        self.cosine = tf.keras.metrics.CosineSimilarity()

        self.negative_num = negative_num
        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        #self.optimizer = tf.keras.optimizer.RMSprop(learning_rate=self.learning_rate)



    def graph_readout(self, node_representation):
        graph_representation = tf.reshape(tf.math.sigmoid(tf.math.reduce_mean(node_representation, axis=0)), shape=[1, -1])
        return graph_representation

    def DGI_loss_fn(self, h, corp_h, c):
        pos_logits = tf.reshape(tf.matmul(tf.matmul(h, self.bi_weights), tf.transpose(c)), shape=[h.shape[0], 1])
        neg_logits = tf.reshape(tf.matmul(tf.matmul(corp_h, self.bi_weights), tf.transpose(c)), shape=[h.shape[0], 1])
        logits = tf.concat([pos_logits, neg_logits], axis=0)
        labels = np.concatenate((np.ones(shape=(h.shape[0], 1)), np.zeros(shape=(h.shape[0], 1))), axis=0)
        loss = keras.losses.binary_crossentropy(labels, logits, from_logits=True)
        loss = tf.math.reduce_mean(loss)
        return loss


    def instance_classify_loss_sig_fn(self, aug_h, h, neg_h):
        pos = tf.reshape(tf.math.reduce_sum(tf.math.multiply(aug_h, h), axis=1), shape=(-1, 1))
        aug_h = tf.reshape(tf.tile(aug_h, [1, self.negative_num]), shape=[h.shape[0], self.negative_num, h.shape[1]])
        neg = tf.math.reduce_sum(tf.math.multiply(aug_h, neg_h), axis=2)
        logits = tf.concat((pos, neg), axis=1)
        ones = np.ones(shape=(h.shape[0], 1), dtype=np.float32)
        zeros = np.zeros(shape=(h.shape[0], self.negative_num), dtype=np.float32)
        labels = np.concatenate((ones, zeros), axis=1)
        loss = tf.math.reduce_mean(tf.math.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits), axis=1))
        return loss

    def instance_hinge_loss_fn(self, aug_h, h, neg_h):
        pos = tf.reshape(tf.math.reduce_sum(tf.math.multiply(aug_h, h), axis=1), shape=(-1, 1))
        aug_h = tf.reshape(tf.tile(aug_h, [1, self.negative_num]), shape=[h.shape[0], self.negative_num, h.shape[1]])
        neg = tf.math.reduce_sum(tf.math.multiply(aug_h, neg_h), axis=2)

        #loss = tf.math.reduce_mean(tf.math.reduce_sum(tf.math.maximum(0, (neg-pos)/pos), axis=-1), axis=-1)
        loss = tf.math.reduce_mean(tf.math.reduce_sum(tf.math.maximum(0, neg-pos), axis=-1), axis=-1)
        return loss

    def run_std(self, inputs):
        x, adj = inputs
        h = self.graph_enc((x, adj))
        std = self.std_enc(h)
        return std

    def run_link_logits(self, inputs, val_pos_edges, val_neg_edges):
        x, adj = inputs
        self.h = self.graph_enc((x, adj))
        pos_node_a, pos_node_b = tf.split(tf.nn.embedding_lookup(self.h, val_pos_edges), 2, axis=1)
        neg_node_a, neg_node_b = tf.split(tf.nn.embedding_lookup(self.h, val_neg_edges), 2, axis=1)
        pos_node_a = tf.squeeze(pos_node_a)
        pos_node_b = tf.squeeze(pos_node_b)
        neg_node_a = tf.squeeze(neg_node_a)
        neg_node_b = tf.squeeze(neg_node_b)

        pos_logits = tf.math.sigmoid(tf.math.reduce_sum(tf.math.multiply(pos_node_a, pos_node_b), axis=-1))
        neg_logits = tf.math.sigmoid(tf.math.reduce_sum(tf.math.multiply(neg_node_a, neg_node_b), axis=-1))
        logits = tf.concat((pos_logits, neg_logits), axis=0)

        return logits

    def cal_norm_loss_fn(self, h, nb_h, r):
        h = tf.reshape(tf.tile(h, [1, self.negative_num]), shape=[h.shape[0], self.negative_num, h.shape[1]])
        loss = tf.math.reduce_mean(tf.math.reduce_sum(tf.math.maximum(tf.math.reduce_sum(tf.math.square(h - nb_h), axis=-1) - r, 0), axis=-1))
        return loss


    def call(self, inputs, training=None):
        x, adj, corp_inputs, corp_adj, nb_negative_index = inputs

        node_num = x.dense_shape[0]

        #step in trainin

        #get node representation from deep GNN and graph representaion from readout function
        self.h = self.graph_enc((x, adj), training)
        #self.h = tf.keras.utils.normalize(self.h)
        self.c = self.graph_readout(self.h)

        #corrupt graph and get embedding of corrupt graph
        #corp_inputs = shuffle(x)
        #corp_adj = adj
        #corp_adj = shuffle(adj)

        self.corp_h = self.graph_enc((corp_inputs, corp_adj), training)
        #self.corp_h = tf.keras.utils.normalize(self.corp_h)
        #self.corp_c = self.graph_readout(self.corp_h)

        self.dgi_loss = self.DGI_loss_fn(self.h, self.corp_h, self.c)


        self.std = self.std_enc(self.h)

        aug_list = []
        self.aug_dgi_loss = 0.
        self.instance_loss = 0.
        self.hinge_loss = 0.
        self.norm_loss = 0.
        for i in range(self.augment_num):
            #aug_h = self.h + tf.random.normal(shape=[node_num, self.emb_dim]) * self.std
            aug_h = self.h + unit_sphere_sample(shape=[node_num, self.emb_dim]) * self.std
            aug_list.append(aug_h)
            self.aug_dgi_loss += self.DGI_loss_fn(aug_h, self.corp_h, self.c) * self.aug_dgi_loss_w

            negative_index = np.random.randint(low=0, high=node_num, size=(node_num, self.negative_num))
            negative_samples = tf.nn.embedding_lookup(params=self.h, ids=negative_index)
            #max radius of each node
            nb_negative_samples = tf.nn.embedding_lookup(params=self.h, ids=nb_negative_index)
            #nb_negative_samples = negative_samples
            self.norm_loss += self.cal_norm_loss_fn(aug_h, nb_negative_samples, self.std) * self.norm_loss_w


            self.instance_loss += self.instance_classify_loss_sig_fn(aug_h, self.h, negative_samples) * self.ins_loss_w

            self.hinge_loss += self.instance_hinge_loss_fn(aug_h, self.h, negative_samples) * self.hinge_loss_w

        #aug_h = tf.concat(aug_list, axis=0)
        aug_h = tf.add_n(aug_list) / self.augment_num


        #self.norm_loss = -tf.math.reduce_mean(tf.nn.l2_loss(self.std)) * self.norm_loss_w
        #self.norm_loss = -tf.math.reduce_mean(self.std) * self.norm_loss_w

        loss = self.dgi_loss + self.aug_dgi_loss + self.instance_loss + self.hinge_loss + self.norm_loss

        #loss = self.dgi_loss

        self.add_loss(loss)

        total_loss = tf.add_n(self.losses)
        #total_loss = self.dgi_loss
        loss = (total_loss, self.dgi_loss, self.aug_dgi_loss, self.instance_loss, self.hinge_loss, self.norm_loss)
        #h_list = (self.h, aug_h)

        return loss, self.h, aug_h

class GraphSiameseCLR(keras.Model):
    """Graph contrastive learning with data augmentation operator learned from dataset"""

    def __init__(self, input_dim, gnn_units_list, dnn_units_list, sparse_inputs=False, act=tf.nn.relu, dropout=0., bias=False, l2_r=1e-5,
                 dao_categories_num=5, dao_continue_maxval=1, dao_continue_minval=-1, noise_mean=0., noise_stddev=1.0, noise_dim=10,
                 negative_num=10, aug_dgi_loss_w=1e-5, siamese_loss_w=1e-5, learning_rate=0.01, norm_loss_w=1e-3, siamese_pos_w=10,
                 augment_num=10,
                 **kwargs):
        super(GraphSiameseCLR, self).__init__(**kwargs)

        self.aug_dgi_loss_w = aug_dgi_loss_w
        self.siamese_loss_w = siamese_loss_w
        self.norm_loss_w = norm_loss_w
        self.siamese_pos_w = siamese_pos_w
        self.augment_num = augment_num

        self.ado_op = DAODNN(units_list=dnn_units_list, act=act, dropout=dropout, bias=True, l2_r=0.)

        self.noise_op = DAONoiseOperator(dao_categories_num=dao_categories_num,
                                         dao_continue_maxval=dao_continue_maxval,
                                         dao_continue_minval=dao_continue_minval,
                                         noise_mean=noise_mean, noise_stddev=noise_stddev,
                                         noise_dim=noise_dim)

        self.graph_enc = GraphEncoder(input_dim=input_dim,
                                      units_list=gnn_units_list,
                                      dropout=dropout,
                                      sparse_inputs=sparse_inputs,
                                      act=act,
                                      l2_r=0.,
                                      bias=True
                                      )

        self.emb_dim = gnn_units_list[-1]

        #dscriminator in DGI used to classify <node, graph>, <corrup_node, graph>
        self.bi_weights = self.add_weight(
            shape=(self.emb_dim, self.emb_dim),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True
        )

        #dscriminator in siamese loss to classify <augment node, node>, <augment node, other node>
        self.siamese_w = self.add_weight(
            shape=(1, self.emb_dim),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True
            )

        self.cosine = tf.keras.metrics.CosineSimilarity()

        self.negative_num = negative_num
        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        #self.optimizer = tf.keras.optimizer.RMSprop(learning_rate=self.learning_rate)



    def graph_readout(self, node_representation):
        graph_representation = tf.reshape(tf.math.sigmoid(tf.math.reduce_mean(node_representation, axis=0)), shape=[1, -1])
        return graph_representation

    def DGI_loss_fn(self, h, corp_h, c):
        pos_logits = tf.reshape(tf.matmul(tf.matmul(h, self.bi_weights), tf.transpose(c)), shape=[h.shape[0], 1])
        neg_logits = tf.reshape(tf.matmul(tf.matmul(corp_h, self.bi_weights), tf.transpose(c)), shape=[h.shape[0], 1])
        logits = tf.concat([pos_logits, neg_logits], axis=0)
        labels = np.concatenate((np.ones(shape=(h.shape[0], 1)), np.zeros(shape=(h.shape[0], 1))), axis=0)
        loss = keras.losses.binary_crossentropy(labels, logits, from_logits=True)
        loss = tf.math.reduce_mean(loss)
        return loss

    def simaese_loss_fn(self, aug_h, h, neg_h):
        pos_logits = tf.math.reduce_sum(tf.math.multiply(tf.math.abs(aug_h - h), self.siamese_w), axis=-1)
        pos_labels = np.ones(shape=(h.shape[0],), dtype=np.float32)

        aug_h = tf.reshape(tf.tile(aug_h, [1, self.negative_num]), shape=[-1, self.emb_dim])


        neg_h = tf.reshape(neg_h, shape=[-1, self.emb_dim])
        neg_logits = tf.reduce_sum(tf.multiply(tf.math.abs(aug_h-neg_h), self.siamese_w), axis=-1)
        neg_labels = np.zeros(shape=(aug_h.shape[0],), dtype=np.float32)

        logits = tf.concat([pos_logits, neg_logits], axis=0)
        labels = np.concatenate([pos_labels, neg_labels], axis=0)

        loss = tf.math.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=labels, logits=logits, pos_weight=self.siamese_pos_w))
        return loss


    def call(self, inputs, training=None):
        x, adj, corp_inputs, corp_adj, negative_index = inputs

        node_num = x.dense_shape[0]

        #step in trainin

        #get node representation from deep GNN and graph representaion from readout function
        self.h = self.graph_enc((x, adj), training)
        #self.h = tf.keras.utils.normalize(self.h)
        self.c = self.graph_readout(self.h)

        #corrupt graph and get embedding of corrupt graph
        #corp_inputs = shuffle(x)
        #corp_adj = adj
        #corp_adj = shuffle(adj)

        self.corp_h = self.graph_enc((corp_inputs, corp_adj), training)
        #self.corp_h = tf.keras.utils.normalize(self.corp_h)
        #self.corp_c = self.graph_readout(self.corp_h)

        self.dgi_loss = self.DGI_loss_fn(self.h, self.corp_h, self.c)

        #get augmentation samples using DAO operator
        aug_noise = self.noise_op(node_num)
        #aug_h = self.ado_op(np.concatenate((self.h, aug_noise), axis=1))
        aug_list = []
        self.aug_dgi_loss = 0.
        for i in range(self.augment_num):
            aug_h = self.ado_op(np.concatenate((self.h, aug_noise), axis=1))
            aug_list.append(aug_h)
            self.aug_dgi_loss += self.DGI_loss_fn(aug_h, self.corp_h, self.c) * self.aug_dgi_loss_w
        aug_h = tf.concat(aug_list, axis=0)
        #aug_h = tf.keras.utils.normalize(aug_h)

        #self.aug_dgi_loss = self.DGI_loss_fn(aug_h, self.corp_h, self.c) * self.aug_dgi_loss_w

        #calculate instance discriminitor loss
        #negative_index = np.random.randint(low=0, high=node_num, size=(node_num*self.augment_num, self.negative_num))
        #negative_samples = tf.nn.embedding_lookup(params=self.h, ids=negative_index)
        #self.instance_loss = self.instance_classify_loss_fn(aug_h, self.h, negative_samples) * self.ins_loss_w

        #debug embedding with l2_normolization version = using cosine similarity
        negative_samples = tf.nn.embedding_lookup(params=self.h, ids=negative_index)
        self.siamese_loss = self.simaese_loss_fn(aug_h, tf.tile(self.h,[self.augment_num, 1]), negative_samples) * self.siamese_loss_w

        #norm constrain force augmentation node norm at the same scale as real node
        #self.norm_loss = tf.nn.l2_loss(tf.nn.l2_loss(self.h) - tf.nn.l2_loss(aug_h)) * self.norm_loss_w
        #self.norm_loss = tf.math.reduce_mean(tf.math.square(tf.nn.l2_loss(self.h) - tf.nn.l2_loss(aug_h))) * self.norm_loss_w
        self.norm_loss = 0.


        loss = self.dgi_loss + self.aug_dgi_loss + self.siamese_loss + self.norm_loss
        self.add_loss(loss)

        total_loss = tf.add_n(self.losses)
        loss = (total_loss, self.dgi_loss, self.aug_dgi_loss, self.siamese_loss, self.norm_loss)
        #h_list = (self.h, aug_h)

        return loss, self.h, aug_h



class GAE(keras.Model):

    def __init__(self, input_dim, gnn_units_list, sparse_inputs=False, act=tf.nn.relu, dropout=0., bias=False, l2_r=1e-5,
                learning_rate=0.01, adj_pos_weight=1.0, norm=1.0,
                 **kwargs):
        super(GAE, self).__init__(**kwargs)

        self.norm = norm
        self.adj_pos_weight = adj_pos_weight

        self.graph_enc = GraphEncoder(input_dim=input_dim,
                                      units_list=gnn_units_list,
                                      dropout=dropout,
                                      sparse_inputs=sparse_inputs,
                                      act=act,
                                      l2_r=0.,
                                      #bias=True
                                      bias=False
                                      )

        self.emb_dim = gnn_units_list[-1]

        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        #self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)
        #self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate)

    def GAE_loss_fn(self, h, cont_h, adj_orig):
        #self.reconstructions = InnerProductDecoder(input_dim=self.emb_dim,
        #                              act=lambda x: x,
        #                              )(h)
        reconstructions = tf.reshape(tf.matmul(h, tf.transpose(cont_h)), [-1])
        labels = tf.reshape(tf.sparse.to_dense(adj_orig, validate_indices=False), [-1])
        loss = self.norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=reconstructions, labels=labels, pos_weight=self.adj_pos_weight))
        return loss

    def run_std(self, inputs):
        x, adj = inputs
        h = self.graph_enc((x, adj))
        std = self.std_enc(h)
        return std

    def run_link_logits(self, inputs, val_pos_edges, val_neg_edges):
        x, adj = inputs
        self.h = self.graph_enc((x, adj))
        pos_node_a, pos_node_b = tf.split(tf.nn.embedding_lookup(self.h, val_pos_edges), 2, axis=1)
        neg_node_a, neg_node_b = tf.split(tf.nn.embedding_lookup(self.h, val_neg_edges), 2, axis=1)
        pos_node_a = tf.squeeze(pos_node_a)
        pos_node_b = tf.squeeze(pos_node_b)
        neg_node_a = tf.squeeze(neg_node_a)
        neg_node_b = tf.squeeze(neg_node_b)

        pos_logits = tf.math.sigmoid(tf.math.reduce_sum(tf.math.multiply(pos_node_a, pos_node_b), axis=-1))
        neg_logits = tf.math.sigmoid(tf.math.reduce_sum(tf.math.multiply(neg_node_a, neg_node_b), axis=-1))
        logits = tf.concat((pos_logits, neg_logits), axis=0)

        return logits




    def call(self, inputs, training=None):
        x, adj, adj_orig = inputs

        node_num = x.dense_shape[0]

        #step in trainin

        #get node representation from deep GNN and graph representaion from readout function
        self.h = self.graph_enc((x, adj), training)
        #self.h = tf.keras.utils.normalize(self.h)

        self.gae_loss = self.GAE_loss_fn(self.h, self.h, adj_orig)

        return self.gae_loss, self.h

class GAEAdvID(keras.Model):
    """Graph contrastive learning with data augmentation operator learned from dataset"""

    def __init__(self, input_dim, gnn_units_list, sparse_inputs=False, act=tf.nn.relu, dropout=0., bias=False, l2_r=1e-5,
                 negative_num=10, aug_gae_loss_w=1e-5, ins_loss_w=1e-5, learning_rate=0.01, temperature=0.07, augment_num=1,
                 norm=1e-1, adj_pos_weight=1.0, norm_loss_w=-0.1, hinge_loss_w=0.,
                 **kwargs):
        super(GAEAdvID, self).__init__(**kwargs)

        self.aug_gae_loss_w = aug_gae_loss_w
        self.ins_loss_w = ins_loss_w
        self.temperature = temperature
        self.augment_num = augment_num
        self.norm = norm
        self.adj_pos_weight = adj_pos_weight

        self.hinge_loss_w = hinge_loss_w

        self.norm_loss_w = norm_loss_w

        self.graph_enc = GraphEncoder(input_dim=input_dim,
                                      units_list=gnn_units_list,
                                      dropout=dropout,
                                      sparse_inputs=sparse_inputs,
                                      act=act,
                                      l2_r=0.,
                                      #bias=True
                                      bias=False
                                      )

        self.emb_dim = gnn_units_list[-1]

        #self.std_enc = keras.layers.Dense(units=gnn_units_list[-1],
        self.std_enc = keras.layers.Dense(1,
                                                           activation = tf.math.sigmoid,
                                                           #activation = lambda x:x,
                                                           #use_bias=True,
                                                           use_bias=False,
                                                           kernel_initializer='glorot_uniform',
                                                           bias_initializer='zeros',
                                                           kernel_regularizer=tf.keras.regularizers.l2(l2_r),
                                                           bias_regularizer=tf.keras.regularizers.l2(l2_r),
                                                           name="DAO_std",
                                                           )


        self.negative_num = negative_num
        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        #self.optimizer = tf.keras.optimizer.RMSprop(learning_rate=self.learning_rate)


    def GAE_loss_fn(self, h, cont_h, adj_orig):

        reconstructions = tf.reshape(tf.matmul(h, tf.transpose(cont_h)), [-1])
        labels = tf.reshape(tf.sparse.to_dense(adj_orig, validate_indices=False), [-1])
        loss = self.norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=reconstructions, labels=labels, pos_weight=self.adj_pos_weight))
        return loss



    def instance_classify_loss_sig_fn(self, aug_h, h, neg_h):
        pos = tf.reshape(tf.math.reduce_sum(tf.math.multiply(aug_h, h), axis=1), shape=(-1, 1)) / self.temperature

        aug_h = tf.reshape(tf.tile(aug_h, [1, self.negative_num]), shape=[h.shape[0], self.negative_num, h.shape[1]])
        neg_1 = tf.math.reduce_sum(tf.math.multiply(aug_h, neg_h), axis=2) / self.temperature

        h = tf.reshape(tf.tile(h, [1, self.negative_num]), shape=[h.shape[0], self.negative_num, h.shape[1]])
        neg_2 = tf.math.reduce_sum(tf.math.multiply(h, neg_h), axis=2) / self.temperature

        neg = np.concatenate((neg_1, neg_2), axis=1)

        logits = tf.concat((pos, neg), axis=1)
        ones = np.ones(shape=(h.shape[0], 1), dtype=np.float32)
        zeros = np.zeros(shape=(h.shape[0], self.negative_num*2), dtype=np.float32)
        labels = np.concatenate((ones, zeros), axis=1)
        loss = tf.math.reduce_mean(tf.math.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits), axis=1))
        return loss

    def instance_hinge_loss_fn(self, aug_h, h, neg_h):
        pos = tf.reshape(tf.math.reduce_sum(tf.math.multiply(aug_h, h), axis=1), shape=(-1, 1))
        aug_h = tf.reshape(tf.tile(aug_h, [1, self.negative_num]), shape=[h.shape[0], self.negative_num, h.shape[1]])
        neg = tf.math.reduce_sum(tf.math.multiply(aug_h, neg_h), axis=2)

        loss = tf.math.reduce_mean(tf.math.reduce_sum(tf.math.maximum(0, neg-pos), axis=-1), axis=-1)
        return loss

    def instance_hinge_loss_fn(self, aug_h, h, neg_h):
        pos = tf.reshape(tf.math.reduce_sum(tf.math.multiply(aug_h, h), axis=1), shape=(-1, 1))
        aug_h = tf.reshape(tf.tile(aug_h, [1, self.negative_num]), shape=[h.shape[0], self.negative_num, h.shape[1]])
        neg = tf.math.reduce_sum(tf.math.multiply(aug_h, neg_h), axis=2)

        #loss = tf.math.reduce_mean(tf.math.reduce_sum(tf.math.maximum(0, (neg-pos)/pos), axis=-1), axis=-1)
        loss = tf.math.reduce_mean(tf.math.reduce_sum(tf.math.maximum(0, neg-pos), axis=-1), axis=-1)
        return loss

    def cal_norm_loss_fn(self, h, nb_h, r):
        h = tf.reshape(tf.tile(h, [1, self.negative_num]), shape=[h.shape[0], self.negative_num, h.shape[1]])
        loss = tf.math.reduce_mean(tf.math.reduce_sum(tf.math.maximum(tf.math.reduce_sum(tf.math.square(h - nb_h), axis=-1) - r, 0), axis=-1))
        return loss


    def run_std(self, inputs):
        x, adj = inputs
        h = self.graph_enc((x, adj))
        std = self.std_enc(h)
        return std

    def run_link_logits(self, inputs, val_pos_edges, val_neg_edges):
        x, adj = inputs
        self.h = self.graph_enc((x, adj))
        pos_node_a, pos_node_b = tf.split(tf.nn.embedding_lookup(self.h, val_pos_edges), 2, axis=1)
        neg_node_a, neg_node_b = tf.split(tf.nn.embedding_lookup(self.h, val_neg_edges), 2, axis=1)
        pos_node_a = tf.squeeze(pos_node_a)
        pos_node_b = tf.squeeze(pos_node_b)
        neg_node_a = tf.squeeze(neg_node_a)
        neg_node_b = tf.squeeze(neg_node_b)

        pos_logits = tf.math.sigmoid(tf.math.reduce_sum(tf.math.multiply(pos_node_a, pos_node_b), axis=-1))
        neg_logits = tf.math.sigmoid(tf.math.reduce_sum(tf.math.multiply(neg_node_a, neg_node_b), axis=-1))
        logits = tf.concat((pos_logits, neg_logits), axis=0)

        return logits




    def call(self, inputs, training=None):
        x, adj, adj_orig, nb_negative_index, gradint_dir = inputs

        gradint_dir = tf.keras.utils.normalize(gradint_dir)

        node_num = x.dense_shape[0]

        #step in trainin

        #get node representation from deep GNN and graph representaion from readout function
        self.h = self.graph_enc((x, adj), training)
        #self.h = tf.keras.utils.normalize(self.h)

        self.gae_loss = self.GAE_loss_fn(self.h, self.h, adj_orig)


        self.std = self.std_enc(self.h)
        gradint_dir = tf.multiply(gradint_dir, self.std)
        #self.std = 0.0018
        aug_list = []
        self.aug_gae_loss = 0.
        self.instance_loss = 0.
        self.hinge_loss = 0.
        self.norm_loss = 0.

        for i in range(self.augment_num):
            #aug_h = self.h + tf.random.normal(shape=[node_num, self.emb_dim]) * self.std
            aug_h = self.h + gradint_dir

            aug_list.append(aug_h)
            self.aug_gae_loss += self.GAE_loss_fn(aug_h, self.h, adj_orig) * self.aug_gae_loss_w

            negative_index = np.random.randint(low=0, high=node_num, size=(node_num, self.negative_num))
            negative_samples = tf.nn.embedding_lookup(params=self.h, ids=negative_index)
            #max radius of each node
            nb_negative_samples = tf.nn.embedding_lookup(params=self.h, ids=nb_negative_index)
            #nb_negative_samples = negative_samples
            #self.norm_loss += self.cal_norm_loss_fn(aug_h, nb_negative_samples, tf.squeeze(tf.nn.embedding_lookup(self.std, nb_negative_index))) * self.norm_loss_w

            #calculate instance discriminitor loss
            
            self.instance_loss += self.instance_classify_loss_sig_fn(aug_h, self.h, negative_samples) * self.ins_loss_w

            self.hinge_loss += self.instance_hinge_loss_fn(aug_h, self.h, negative_samples) * self.hinge_loss_w

        #aug_h = tf.concat(aug_list, axis=0)
        aug_h = tf.add_n(aug_list) / self.augment_num


        #self.norm_loss = -tf.math.reduce_mean(tf.nn.l2_loss(self.std)) * self.norm_loss_w
        self.norm_loss = tf.math.reduce_mean(1.0 - self.std) * self.norm_loss_w

        loss = self.gae_loss + self.aug_gae_loss + self.instance_loss + self.hinge_loss + self.norm_loss
        self.add_loss(loss)

        total_loss = tf.add_n(self.losses)
        #total_loss = self.gae_loss
        loss = (total_loss, self.gae_loss, self.aug_gae_loss, self.instance_loss, self.hinge_loss, self.norm_loss)
        #h_list = (self.h, aug_h)

        #return loss, self.h, aug_h, tf.math.reduce_mean(tf.math.reduce_sum(tf.math.exp(self.log_std), axis=-1))
        #return loss, self.h, aug_h, tf.math.reduce_mean(tf.math.reduce_sum(self.std, axis=-1))
        return loss, self.h, aug_h

class GAEGaussianCLR(keras.Model):
    """Graph contrastive learning with data augmentation operator learned from dataset"""

    def __init__(self, input_dim, gnn_units_list, sparse_inputs=False, act=tf.nn.relu, dropout=0., bias=False, l2_r=1e-5,
                 negative_num=10, aug_gae_loss_w=1e-5, ins_loss_w=1e-5, learning_rate=0.01, temperature=0.07, augment_num=1,
                 norm=1e-1, adj_pos_weight=1.0, norm_loss_w=-0.1, hinge_loss_w=0.,
                 **kwargs):
        super(GAEGaussianCLR, self).__init__(**kwargs)

        self.aug_gae_loss_w = aug_gae_loss_w
        self.ins_loss_w = ins_loss_w
        self.temperature = temperature
        self.augment_num = augment_num
        self.norm = norm
        self.adj_pos_weight = adj_pos_weight

        self.hinge_loss_w = hinge_loss_w

        self.norm_loss_w = norm_loss_w

        self.graph_enc = GraphEncoder(input_dim=input_dim,
                                      units_list=gnn_units_list,
                                      dropout=dropout,
                                      sparse_inputs=sparse_inputs,
                                      act=act,
                                      l2_r=0.,
                                      #bias=True
                                      bias=False
                                      )

        self.emb_dim = gnn_units_list[-1]

        #self.std_enc = keras.layers.Dense(units=gnn_units_list[-1],
        self.std_enc = keras.layers.Dense(1,
                                                           #activation = tf.nn.relu,
                                                           #activation = lambda x:x,
                                                           activation = tf.math.sigmoid,
                                                           #use_bias=True,
                                                           use_bias=False,
                                                           kernel_initializer='glorot_uniform',
                                                           bias_initializer='zeros',
                                                           kernel_regularizer=tf.keras.regularizers.l2(l2_r),
                                                           bias_regularizer=tf.keras.regularizers.l2(l2_r),
                                                           name="DAO_std",
                                                           )


        self.negative_num = negative_num
        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        #self.optimizer = tf.keras.optimizer.RMSprop(learning_rate=self.learning_rate)


    def GAE_loss_fn(self, h, cont_h, adj_orig):

        reconstructions = tf.reshape(tf.matmul(h, tf.transpose(cont_h)), [-1])
        labels = tf.reshape(tf.sparse.to_dense(adj_orig, validate_indices=False), [-1])
        loss = self.norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=reconstructions, labels=labels, pos_weight=self.adj_pos_weight))
        return loss



    def instance_classify_loss_sig_fn(self, aug_h, h, neg_h):
        pos = tf.reshape(tf.math.reduce_sum(tf.math.multiply(aug_h, h), axis=1), shape=(-1, 1))/ self.temperature
        aug_h = tf.reshape(tf.tile(aug_h, [1, self.negative_num]), shape=[h.shape[0], self.negative_num, h.shape[1]])
        neg = tf.math.reduce_sum(tf.math.multiply(aug_h, neg_h), axis=2)/ self.temperature
        logits = tf.concat((pos, neg), axis=1)
        ones = np.ones(shape=(h.shape[0], 1), dtype=np.float32)
        zeros = np.zeros(shape=(h.shape[0], self.negative_num), dtype=np.float32)
        labels = np.concatenate((ones, zeros), axis=1)
        loss = tf.math.reduce_mean(tf.math.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits), axis=1))
        return loss

    def instance_hinge_loss_fn(self, aug_h, h, neg_h):
        pos = tf.reshape(tf.math.reduce_sum(tf.math.multiply(aug_h, h), axis=1), shape=(-1, 1))
        aug_h = tf.reshape(tf.tile(aug_h, [1, self.negative_num]), shape=[h.shape[0], self.negative_num, h.shape[1]])
        neg = tf.math.reduce_sum(tf.math.multiply(aug_h, neg_h), axis=2)

        loss = tf.math.reduce_mean(tf.math.reduce_sum(tf.math.maximum(0, neg-pos), axis=-1), axis=-1)
        return loss

    def instance_hinge_loss_fn(self, aug_h, h, neg_h):
        pos = tf.reshape(tf.math.reduce_sum(tf.math.multiply(aug_h, h), axis=1), shape=(-1, 1))
        aug_h = tf.reshape(tf.tile(aug_h, [1, self.negative_num]), shape=[h.shape[0], self.negative_num, h.shape[1]])
        neg = tf.math.reduce_sum(tf.math.multiply(aug_h, neg_h), axis=2)

        #loss = tf.math.reduce_mean(tf.math.reduce_sum(tf.math.maximum(0, (neg-pos)/pos), axis=-1), axis=-1)
        loss = tf.math.reduce_mean(tf.math.reduce_sum(tf.math.maximum(0, neg-pos), axis=-1), axis=-1)
        return loss

    def cal_norm_loss_fn(self, h, nb_h, r):
        h = tf.reshape(tf.tile(h, [1, self.negative_num]), shape=[h.shape[0], self.negative_num, h.shape[1]])
        loss = tf.math.reduce_mean(tf.math.reduce_sum(tf.math.maximum(tf.math.reduce_sum(tf.math.square(h - nb_h), axis=-1) - r, 0), axis=-1))
        return loss


    def run_std(self, inputs):
        x, adj = inputs
        h = self.graph_enc((x, adj))
        std = self.std_enc(h)
        return std

    def run_link_logits(self, inputs, val_pos_edges, val_neg_edges):
        x, adj = inputs
        self.h = self.graph_enc((x, adj))
        pos_node_a, pos_node_b = tf.split(tf.nn.embedding_lookup(self.h, val_pos_edges), 2, axis=1)
        neg_node_a, neg_node_b = tf.split(tf.nn.embedding_lookup(self.h, val_neg_edges), 2, axis=1)
        pos_node_a = tf.squeeze(pos_node_a)
        pos_node_b = tf.squeeze(pos_node_b)
        neg_node_a = tf.squeeze(neg_node_a)
        neg_node_b = tf.squeeze(neg_node_b)

        pos_logits = tf.math.sigmoid(tf.math.reduce_sum(tf.math.multiply(pos_node_a, pos_node_b), axis=-1))
        neg_logits = tf.math.sigmoid(tf.math.reduce_sum(tf.math.multiply(neg_node_a, neg_node_b), axis=-1))
        logits = tf.concat((pos_logits, neg_logits), axis=0)

        return logits




    def call(self, inputs, training=None):
        x, adj, adj_orig, nb_negative_index = inputs

        node_num = x.dense_shape[0]

        #step in trainin

        #get node representation from deep GNN and graph representaion from readout function
        self.h = self.graph_enc((x, adj), training)
        #self.h = tf.keras.utils.normalize(self.h)

        self.gae_loss = self.GAE_loss_fn(self.h, self.h, adj_orig)


        self.std = self.std_enc(self.h)
        #self.std = 0.0018
        aug_list = []
        self.aug_gae_loss = 0.
        self.instance_loss = 0.
        self.hinge_loss = 0.
        self.norm_loss = 0.

        for i in range(self.augment_num):
            aug_h = self.h + tf.keras.utils.normalize(tf.random.normal(shape=[node_num, self.emb_dim])) * self.std
            #aug_h = self.h + tf.random.normal(shape=[node_num, self.emb_dim]) * self.std
            #aug_h = self.h + unit_sphere_sample(shape=[node_num, self.emb_dim]) * self.std

            aug_list.append(aug_h)
            self.aug_gae_loss += self.GAE_loss_fn(aug_h, self.h, adj_orig) * self.aug_gae_loss_w

            #negative_index = np.random.randint(low=0, high=node_num, size=(node_num, self.negative_num))
            negative_index = nb_negative_index
            negative_samples = tf.nn.embedding_lookup(params=self.h, ids=negative_index)
            #max radius of each node
            nb_negative_samples = tf.nn.embedding_lookup(params=self.h, ids=nb_negative_index)
            #nb_negative_samples = negative_samples
            #self.norm_loss += self.cal_norm_loss_fn(aug_h, nb_negative_samples, tf.squeeze(tf.nn.embedding_lookup(self.std, nb_negative_index))) * self.norm_loss_w

            #calculate instance discriminitor loss
            self.instance_loss += self.instance_classify_loss_sig_fn(aug_h, self.h, negative_samples) * self.ins_loss_w

            self.hinge_loss += self.instance_hinge_loss_fn(aug_h, self.h, negative_samples) * self.hinge_loss_w

        #aug_h = tf.concat(aug_list, axis=0)
        aug_h = tf.add_n(aug_list) / self.augment_num


        #self.norm_loss = -tf.math.reduce_mean(tf.nn.l2_loss(self.std)) * self.norm_loss_w
        #self.norm_loss = -tf.math.reduce_mean(self.std) * self.norm
        self.norm_loss = tf.math.reduce_mean(1.0 - self.std) * self.norm_loss_w

        loss = self.gae_loss + self.aug_gae_loss + self.instance_loss + self.hinge_loss + self.norm_loss
        self.add_loss(loss)

        total_loss = tf.add_n(self.losses)
        #total_loss = self.gae_loss
        loss = (total_loss, self.gae_loss, self.aug_gae_loss, self.instance_loss, self.hinge_loss, self.norm_loss)
        #h_list = (self.h, aug_h)

        #return loss, self.h, aug_h, tf.math.reduce_mean(tf.math.reduce_sum(tf.math.exp(self.log_std), axis=-1))
        #return loss, self.h, aug_h, tf.math.reduce_mean(tf.math.reduce_sum(self.std, axis=-1))
        return loss, self.h, aug_h

class GAECLR(keras.Model):
    """Graph contrastive learning with data augmentation operator learned from dataset"""

    def __init__(self, input_dim, gnn_units_list, dnn_units_list, sparse_inputs=False, act=tf.nn.relu, dropout=0., bias=False, l2_r=1e-5,
                 negative_num=10, aug_gae_loss_w=1e-5, ins_loss_w=1e-5, learning_rate=0.01, temperature=0.07, augment_num=10,
                 dao_categories_num=5, dao_continue_maxval=1, dao_continue_minval=-1, noise_mean=0., noise_stddev=1.0, noise_dim=10,
                 norm=1e-1, adj_pos_weight=1.0, hinge_loss_w=0.,
                 **kwargs):
        super(GAECLR, self).__init__(**kwargs)

        self.aug_gae_loss_w = aug_gae_loss_w
        self.ins_loss_w = ins_loss_w
        self.temperature = temperature
        self.augment_num = augment_num
        self.norm = norm
        self.adj_pos_weight = adj_pos_weight
        self.hinge_loss_w = hinge_loss_w

        self.ado_op = DAODNN(units_list=dnn_units_list, act=act, dropout=dropout, bias=True, l2_r=0.)

        self.noise_op = DAONoiseOperator(dao_categories_num=dao_categories_num,
                                         dao_continue_maxval=dao_continue_maxval,
                                         dao_continue_minval=dao_continue_minval,
                                         noise_mean=noise_mean, noise_stddev=noise_stddev,
                                         noise_dim=noise_dim)

        self.graph_enc = GraphEncoder(input_dim=input_dim,
                                      units_list=gnn_units_list,
                                      dropout=dropout,
                                      sparse_inputs=sparse_inputs,
                                      act=act,
                                      l2_r=0.,
                                      bias=True
                                      )

        self.emb_dim = gnn_units_list[-1]


        self.negative_num = negative_num
        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        #self.optimizer = tf.keras.optimizer.RMSprop(learning_rate=self.learning_rate)


    def GAE_loss_fn(self, h, cont_h, adj_orig):

        self.reconstructions = tf.reshape(tf.matmul(h, tf.transpose(cont_h)), [-1])
        labels = tf.reshape(tf.sparse.to_dense(adj_orig, validate_indices=False), [-1])
        reconstructions = tf.reshape(self.reconstructions, [-1])
        loss = self.norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=reconstructions, labels=labels, pos_weight=self.adj_pos_weight))
        return loss


    def instance_classify_loss_sig_fn(self, aug_h, h, neg_h):
        pos = tf.reshape(tf.math.reduce_sum(tf.math.multiply(aug_h, h), axis=1), shape=(-1, 1))
        aug_h = tf.reshape(tf.tile(aug_h, [1, self.negative_num]), shape=[h.shape[0], self.negative_num, h.shape[1]])
        neg = tf.math.reduce_sum(tf.math.multiply(aug_h, neg_h), axis=2)
        logits = tf.concat((pos, neg), axis=1)
        ones = np.ones(shape=(h.shape[0], 1), dtype=np.float32)
        zeros = np.zeros(shape=(h.shape[0], self.negative_num), dtype=np.float32)
        labels = np.concatenate((ones, zeros), axis=1)
        loss = tf.math.reduce_mean(tf.math.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits), axis=1))
        return loss

    def instance_hinge_loss_fn(self, aug_h, h, neg_h):
        pos = tf.reshape(tf.math.reduce_sum(tf.math.multiply(aug_h, h), axis=1), shape=(-1, 1))
        aug_h = tf.reshape(tf.tile(aug_h, [1, self.negative_num]), shape=[h.shape[0], self.negative_num, h.shape[1]])
        neg = tf.math.reduce_sum(tf.math.multiply(aug_h, neg_h), axis=2)

        loss = tf.math.reduce_mean(tf.math.reduce_sum(tf.math.maximum(0, neg-pos), axis=-1), axis=-1)
        return loss

    def run_link_logits(self, inputs, val_pos_edges, val_neg_edges):
        x, adj = inputs
        self.h = self.graph_enc((x, adj))
        pos_node_a, pos_node_b = tf.split(tf.nn.embedding_lookup(self.h, val_pos_edges), 2, axis=1)
        neg_node_a, neg_node_b = tf.split(tf.nn.embedding_lookup(self.h, val_neg_edges), 2, axis=1)
        pos_node_a = tf.squeeze(pos_node_a)
        pos_node_b = tf.squeeze(pos_node_b)
        neg_node_a = tf.squeeze(neg_node_a)
        neg_node_b = tf.squeeze(neg_node_b)

        pos_logits = tf.math.sigmoid(tf.math.reduce_sum(tf.math.multiply(pos_node_a, pos_node_b), axis=-1))
        neg_logits = tf.math.sigmoid(tf.math.reduce_sum(tf.math.multiply(neg_node_a, neg_node_b), axis=-1))
        logits = tf.concat((pos_logits, neg_logits), axis=0)

        return logits


    def call(self, inputs, training=None):
        x, adj, adj_orig, negative_index = inputs

        node_num = x.dense_shape[0]


        #get node representation from deep GNN and graph representaion from readout function
        self.h = self.graph_enc((x, adj), training)

        self.gae_loss = self.GAE_loss_fn(self.h, self.h, adj_orig)

        #get augmentation samples using DAO operator
        aug_list = []
        self.aug_gae_loss = 0.
        self.instance_loss = 0.
        self.hinge_loss = 0.

        for i in range(self.augment_num):
            aug_noise = self.noise_op(node_num)
            aug_h = self.ado_op(np.concatenate((self.h, aug_noise), axis=1))
            aug_list.append(aug_h)
            self.aug_gae_loss += self.GAE_loss_fn(aug_h, self.h, adj_orig) * self.aug_gae_loss_w

            negative_samples = tf.nn.embedding_lookup(params=self.h, ids=negative_index)
            self.instance_loss += self.instance_classify_loss_sig_fn(aug_h, self.h, negative_samples) * self.ins_loss_w

            self.hinge_loss += self.instance_hinge_loss_fn(aug_h, self.h, negative_samples) * self.hinge_loss_w

        #aug_h = tf.concat(aug_list, axis=0)
        aug_h = tf.add_n(aug_list) / self.augment_num

        loss = self.gae_loss + self.aug_gae_loss + self.instance_loss + self.hinge_loss
        self.add_loss(loss)

        total_loss = tf.add_n(self.losses)
        loss = (total_loss, self.gae_loss, self.aug_gae_loss, self.instance_loss, self.hinge_loss)
        #h_list = (self.h, aug_h)

        return loss, self.h, aug_h

class GAESiameseCLR(keras.Model):
    """Graph contrastive learning with data augmentation operator learned from dataset"""

    def __init__(self, input_dim, gnn_units_list, dnn_units_list, sparse_inputs=False, act=tf.nn.relu, dropout=0., bias=False, l2_r=1e-5,
                 negative_num=10, aug_gae_loss_w=1e-5, siamese_loss_w=1e-5, learning_rate=0.01, temperature=0.07, augment_num=10,
                 dao_categories_num=5, dao_continue_maxval=1, dao_continue_minval=-1, noise_mean=0., noise_stddev=1.0, noise_dim=10,
                 norm=1e-1, adj_pos_weight=1.0, siamese_pos_w=1.0,
                 **kwargs):
        super(GAESiameseCLR, self).__init__(**kwargs)

        self.aug_gae_loss_w = aug_gae_loss_w
        self.siamese_loss_w = siamese_loss_w
        self.temperature = temperature
        self.augment_num = augment_num
        self.norm = norm
        self.adj_pos_weight = adj_pos_weight
        self.siamese_pos_w = siamese_pos_w

        self.ado_op = DAODNN(units_list=dnn_units_list, act=act, dropout=dropout, bias=True, l2_r=0.)

        self.noise_op = DAONoiseOperator(dao_categories_num=dao_categories_num,
                                         dao_continue_maxval=dao_continue_maxval,
                                         dao_continue_minval=dao_continue_minval,
                                         noise_mean=noise_mean, noise_stddev=noise_stddev,
                                         noise_dim=noise_dim)

        self.graph_enc = GraphEncoder(input_dim=input_dim,
                                      units_list=gnn_units_list,
                                      dropout=dropout,
                                      sparse_inputs=sparse_inputs,
                                      act=act,
                                      l2_r=0.,
                                      bias=True
                                      )

        self.emb_dim = gnn_units_list[-1]

        #dscriminator in DGI used to classify <node, graph>, <corrup_node, graph>
        #self.bi_weights = self.add_weight(
        #    shape=(self.emb_dim, self.emb_dim),
        #    initializer=tf.keras.initializers.GlorotUniform(),
        #    trainable=True
        #)

        #dscriminator in siamese loss to classify <augment node, node>, <augment node, other node>
        self.siamese_w = self.add_weight(
            shape=(1, self.emb_dim),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True
            )


        #self.cosine = tf.keras.metrics.CosineSimilarity()

        self.negative_num = negative_num
        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        #self.optimizer = tf.keras.optimizer.RMSprop(learning_rate=self.learning_rate)


    def GAE_loss_fn(self, h, cont_h, adj_orig):
        #self.reconstructions = InnerProductDecoder(input_dim=self.emb_dim,
        #                              act=lambda x: x,
        #                              )(h)
        self.reconstructions = tf.reshape(tf.matmul(h, tf.transpose(cont_h)), [-1])
        labels = tf.reshape(tf.sparse.to_dense(adj_orig, validate_indices=False), [-1])
        reconstructions = tf.reshape(self.reconstructions, [-1])
        loss = self.norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=reconstructions, labels=labels, pos_weight=self.adj_pos_weight))
        return loss


    def simaese_loss_fn(self, aug_h, h, neg_h):
        pos_logits = tf.math.reduce_sum(tf.math.multiply(tf.math.abs(aug_h - h), self.siamese_w), axis=-1)
        pos_labels = np.ones(shape=(h.shape[0],), dtype=np.float32)

        aug_h = tf.reshape(tf.tile(aug_h, [1, self.negative_num]), shape=[-1, self.emb_dim])


        neg_h = tf.reshape(neg_h, shape=[-1, self.emb_dim])
        neg_logits = tf.reduce_sum(tf.multiply(tf.math.abs(aug_h-neg_h), self.siamese_w), axis=-1)
        neg_labels = np.zeros(shape=(aug_h.shape[0],), dtype=np.float32)

        logits = tf.concat([pos_logits, neg_logits], axis=0)
        labels = np.concatenate([pos_labels, neg_labels], axis=0)

        loss = tf.math.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=labels, logits=logits, pos_weight=self.siamese_pos_w))
        return loss


    def call(self, inputs, training=None):
        x, adj, adj_orig, negative_index = inputs

        node_num = x.dense_shape[0]

        #step in trainin

        #get node representation from deep GNN and graph representaion from readout function
        self.h = self.graph_enc((x, adj), training)
        #self.h = tf.keras.utils.normalize(self.h)

        self.gae_loss = self.GAE_loss_fn(self.h, self.h, adj_orig)

        #get augmentation samples using DAO operator
        aug_list = []
        self.aug_dgi_loss = 0.
        aug_noise = self.noise_op(node_num)
        for i in range(self.augment_num):
            aug_h = self.ado_op(np.concatenate((self.h, aug_noise), axis=1))
            aug_list.append(aug_h)
            self.aug_gae_loss = self.GAE_loss_fn(aug_h, self.h, adj_orig) * self.aug_gae_loss_w
        aug_h = tf.concat(aug_list, axis=0)
        #self.log_std = 0.001
        #aug_h = self.h + tf.random.normal(shape=[node_num, self.emb_dim]) * self.log_std
        #aug_h = tf.keras.utils.normalize(aug_h)
        #aug_h = self.h

        #self.aug_gae_loss = self.GAE_loss_fn(self.aug_h, adj_orig) * self.aug_gae_loss_w

        #calculate instance discriminitor loss
        #negative_index = np.random.randint(low=0, high=node_num, size=(node_num*self.augment_num, self.negative_num))
        negative_samples = tf.nn.embedding_lookup(params=self.h, ids=negative_index)
        #negative_samples = (tf.nn.embedding_lookup(params=norm_h, ids=negative_index), tf.nn.embedding_lookup(params=norm_aug_h, ids=negative_index))
        self.siamese_loss = self.simaese_loss_fn(aug_h, tf.tile(self.h,[self.augment_num, 1]), negative_samples) * self.siamese_loss_w


        #loss = self.dgi_loss + self.aug_dgi_loss + self.instance_loss + tf.reduce_mean(tf.nn.l2_loss(tf.math.exp(self.log_std)))
        loss = self.gae_loss + self.aug_gae_loss + self.siamese_loss
        self.add_loss(loss)

        total_loss = tf.add_n(self.losses)
        loss = (total_loss, self.gae_loss, self.aug_gae_loss, self.siamese_loss, 0.)
        #h_list = (self.h, aug_h)

        return loss, self.h, aug_h



class DNNFeatureCLR(keras.Model):
    '''deepwalk model with space contrain'''
    '''No use TODO add feature AE + GRR'''
    def __init__(self, gnn_units_list, dnn_units_list, learning_rate=0.1, negative_num=10, act=tf.nn.relu,
                ins_loss_w=1e-3, siamese_loss_w=1e-3, model_name="DNNFeatureCLR", dropout=0.,
                dao_categories_num=5, dao_continue_maxval=1, dao_continue_minval=-1, noise_mean=0., noise_stddev=1.0, noise_dim=10,
                siamese_pos_w=1.0, l2_r=1e-6, temperature=1.0, hinge_loss_w=0.,
                **kwargs):
        super(DNNFeatureCLR, self).__init__(**kwargs)

        self.emb_dim = gnn_units_list[-1]
        self.ins_loss_w = ins_loss_w
        self.model_name = model_name
        self.siamese_loss_w = siamese_loss_w
        self.siamese_pos_w = siamese_pos_w
        self.temperature = temperature
        self.negative_num = negative_num
        self.l2_r = l2_r
        self.dropout = dropout
        self.hinge_loss_w = 0.

#keras.layers.Embedding version
        self.emb_layer = keras.layers.Dense(units=self.emb_dim,
                                                           #activation=act,
                                                           activation = tf.keras.layers.PReLU(),
                                                           use_bias=True,
                                                           kernel_initializer='glorot_uniform',
                                                           bias_initializer='zeros',
                                                           kernel_regularizer=tf.keras.regularizers.l2(self.l2_r),
                                                           bias_regularizer=tf.keras.regularizers.l2(self.l2_r),
                                                           name="F_enc",
                                                           )



        if self.model_name == 'DNNFeatureGaussianCLR':
            self.std_enc = keras.layers.Dense(units=self.emb_dim,
                                                           activation = tf.nn.relu,
                                                           use_bias=True,
                                                           kernel_initializer='glorot_uniform',
                                                           bias_initializer='zeros',
                                                           kernel_regularizer=tf.keras.regularizers.l2(l2_r),
                                                           bias_regularizer=tf.keras.regularizers.l2(l2_r),
                                                           name="DAO_std",
                                                           )
        elif self.model_name == 'DNNFeatureCLR' or self.model_name == 'DNNFeatureSiameseCLR':
            self.ado_op = DAODNN(units_list=dnn_units_list, act=act, dropout=dropout, bias=True, l2_r=0.)

            self.noise_op = DAONoiseOperator(dao_categories_num=dao_categories_num,
                                         dao_continue_maxval=dao_continue_maxval,
                                         dao_continue_minval=dao_continue_minval,
                                         noise_mean=noise_mean, noise_stddev=noise_stddev,
                                         noise_dim=noise_dim)

            if self.model_name == 'DNNFeatureSiameseCLR':
            #dscriminator in siamese loss to classify <augment node, node>, <augment node, other node>
                self.siamese_w = self.add_weight(
                                        shape=(1, self.emb_dim),
                                        initializer=tf.keras.initializers.GlorotUniform(),
                                        trainable=True
                                        )

        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)


    def instance_classify_loss_sig_fn(self, aug_h, h, neg_h):
        pos = tf.reshape(tf.math.reduce_sum(tf.math.multiply(aug_h, h), axis=1), shape=(-1, 1))
        aug_h = tf.reshape(tf.tile(aug_h, [1, self.negative_num]), shape=[h.shape[0], self.negative_num, h.shape[1]])
        neg = tf.math.reduce_sum(tf.math.multiply(aug_h, neg_h), axis=2)
        logits = tf.concat((pos, neg), axis=1)
        ones = np.ones(shape=(h.shape[0], 1), dtype=np.float32)
        zeros = np.zeros(shape=(h.shape[0], self.negative_num), dtype=np.float32)
        labels = np.concatenate((ones, zeros), axis=1)
        loss = tf.math.reduce_mean(tf.math.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits), axis=1))
        return loss


    def simaese_loss_fn(self, aug_h, h, neg_h):
        pos_logits = tf.math.reduce_sum(tf.math.multiply(tf.math.abs(aug_h - h), self.siamese_w), axis=-1)
        pos_labels = np.ones(shape=(h.shape[0],), dtype=np.float32)

        aug_h = tf.reshape(tf.tile(aug_h, [1, self.negative_num]), shape=[-1, self.emb_dim])


        neg_h = tf.reshape(neg_h, shape=[-1, self.emb_dim])
        neg_logits = tf.reduce_sum(tf.multiply(tf.math.abs(aug_h-neg_h), self.siamese_w), axis=-1)
        neg_labels = np.zeros(shape=(aug_h.shape[0],), dtype=np.float32)

        logits = tf.concat([pos_logits, neg_logits], axis=0)
        labels = np.concatenate([pos_labels, neg_labels], axis=0)

        loss = tf.math.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=labels, logits=logits, pos_weight=self.siamese_pos_w))
        return loss

    def instance_hinge_loss_fn(self, aug_h, h, neg_h):
        pos = tf.reshape(tf.math.reduce_sum(tf.math.multiply(aug_h, h), axis=1), shape=(-1, 1))
        aug_h = tf.reshape(tf.tile(aug_h, [1, self.negative_num]), shape=[h.shape[0], self.negative_num, h.shape[1]])
        neg = tf.math.reduce_sum(tf.math.multiply(aug_h, neg_h), axis=2)

        loss = tf.math.reduce_mean(tf.math.reduce_sum(tf.math.maximum(0, neg-pos), axis=-1), axis=-1)
        return loss


    def call(self, inputs, training=None):
        X = inputs

        negative_index = np.random.randint(low=0, high=X.shape[0], size=(X.shape[0], self.negative_num))

        if training:
            X = tf.SparseTensor(indices=X.indices,
                        values=tf.nn.dropout(X.values, self.dropout),
                        dense_shape=X.dense_shape)

        self.embedding = self.emb_layer(X)



        loss = 0.
        self.siamese_loss = 0.
        self.instance_loss = 0.

        if self.model_name == 'DNNFeatureGaussianCLR':
            self.std = self.std_enc(self.embedding)
            aug_h = self.embedding + tf.random.normal(shape=[X.shape[0], self.emb_dim]) * self.std
        elif self.model_name in ['DNNFeatureCLR', 'DNNFeatureSiameseCLR']:
            aug_noise = self.noise_op(X.shape[0])
            aug_h = self.ado_op(np.concatenate((self.embedding, aug_noise), axis=1))


        negative_samples = tf.nn.embedding_lookup(params=self.embedding, ids=negative_index)
        if self.model_name in ['DNNFeatureGaussianCLR', 'DNNFeatureCLR']:
            #norm_aug_h = tf.keras.utils.normalize(aug_h)
            #norm_embedding = tf.keras.utils.normalize(self.embedding)
            #norm_neg_emb = tf.keras.utils.normalize(negative_samples)

            #self.instance_loss = self.instance_classify_loss_fn(norm_aug_h, norm_embedding, norm_neg_emb) * self.ins_loss_w
            self.instance_loss = self.instance_classify_loss_sig_fn(aug_h, self.embedding, negative_samples) * self.ins_loss_w
        else:
            self.siamese_loss = self.simaese_loss_fn(aug_h, self.embedding, negative_samples) * self.siamese_loss_w

        loss = self.instance_loss + self.siamese_loss

        self.add_loss(loss)

        total_loss = tf.add_n(self.losses)

        loss = (total_loss, self.instance_loss, self.siamese_loss)

        return loss, self.embedding, aug_h



class Deepwalk(keras.Model):
    '''deepwalk model with space contrain'''
    '''TODO fix bugs'''
    def __init__(self, node_num, emb_dim, dw_neg_num, dnn_units_list, learning_rate=0.1, negative_num=10, act=tf.nn.relu,
                ins_loss_w=1e-3, aug_dw_loss_w=1e-3, siamese_loss_w=1e-3, model_name="Deepwalk", dropout=0.,
                dao_categories_num=5, dao_continue_maxval=1, dao_continue_minval=-1, noise_mean=0., noise_stddev=1.0, noise_dim=10,
                siamese_pos_w=1.0, l2_r=1e-6, temperature=1.0, norm_loss_w=-0.1,
                **kwargs):
        super(Deepwalk, self).__init__(**kwargs)

        self.node_num = node_num
        self.emb_dim = emb_dim
        self.dw_neg_num = dw_neg_num
        self.ins_loss_w = ins_loss_w
        self.aug_dw_loss_w = aug_dw_loss_w
        self.model_name = model_name
        self.siamese_loss_w = siamese_loss_w
        self.siamese_pos_w = siamese_pos_w
        self.temperature = temperature
        self.negative_num = negative_num
        self.norm_loss_w = norm_loss_w

        #self.label_enc = preprocessing.LabelBinarizer()
        #self.label_enc.fit(np.array(list(range(self.node_num))))

#variable version Error
        #self.embedding = self.add_weight(
        #    shape=(self.node_num, self.emb_dim),
        #    initializer=tf.keras.initializers.GlorotUniform(),
        #    trainable=True
        #)

        #self.cont_embedding = self.add_weight(
        #    shape=(self.node_num, self.emb_dim),
        #    initializer=tf.keras.initializers.GlorotUniform(),
        #    trainable=True
        #)
        #self.nce_bias = self.add_weight(
        #    shape=(self.node_num,),
        #    #initializer=tf.keras.initializers.GlorotUniform(),
        #    initializer=tf.keras.initializers.Zeros(),
        #    trainable=False
        #)
#keras.layers.Embedding version
        self.emb_layer = tf.keras.layers.Embedding(
                        input_dim=self.node_num, output_dim=self.emb_dim,
                        embeddings_initializer=tf.keras.initializers.GlorotUniform(),
                        )

        #self.cont_emb_layer = tf.keras.layers.Embedding(
        #                input_dim=self.node_num, output_dim=self.emb_dim,
        #                embeddings_initializer=tf.keras.initializers.GlorotUniform(),
        #                )


        if self.model_name == 'DeepwalkGaussian':
            self.std_enc = keras.layers.Dense(units=self.emb_dim,
                                                           activation = tf.nn.relu,
                                                           use_bias=True,
                                                           kernel_initializer='glorot_uniform',
                                                           bias_initializer='zeros',
                                                           kernel_regularizer=tf.keras.regularizers.l2(l2_r),
                                                           bias_regularizer=tf.keras.regularizers.l2(l2_r),
                                                           name="DAO_std",
                                                           )
        elif self.model_name == 'DeepwalkCLR' or self.model_name == 'DeepwalkSiamese':
            self.ado_op = DAODNN(units_list=dnn_units_list, act=act, dropout=dropout, bias=True, l2_r=0.)

            self.noise_op = DAONoiseOperator(dao_categories_num=dao_categories_num,
                                         dao_continue_maxval=dao_continue_maxval,
                                         dao_continue_minval=dao_continue_minval,
                                         noise_mean=noise_mean, noise_stddev=noise_stddev,
                                         noise_dim=noise_dim)

            if self.model_name == 'DeepwalkSiamese':
            #dscriminator in siamese loss to classify <augment node, node>, <augment node, other node>
                self.siamese_w = self.add_weight(
            shape=(1, self.emb_dim),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True
            )

        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)


    def instance_classify_loss_sig_fn(self, aug_h, h, neg_h):
        pos = tf.reshape(tf.math.reduce_sum(tf.math.multiply(aug_h, h), axis=1), shape=(-1, 1))
        aug_h = tf.reshape(tf.tile(aug_h, [1, self.negative_num]), shape=[h.shape[0], self.negative_num, h.shape[1]])
        neg = tf.math.reduce_sum(tf.math.multiply(aug_h, neg_h), axis=2)
        logits = tf.concat((pos, neg), axis=1)
        ones = np.ones(shape=(h.shape[0], 1), dtype=np.float32)
        zeros = np.zeros(shape=(h.shape[0], self.negative_num), dtype=np.float32)
        labels = np.concatenate((ones, zeros), axis=1)
        loss = tf.math.reduce_mean(tf.math.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits), axis=1))
        return loss

    def instance_classify_loss_sig_fn(self, aug_h, h, neg_h):
        pos = tf.reshape(tf.math.reduce_sum(tf.math.multiply(aug_h, h), axis=1), shape=(-1, 1))
        aug_h = tf.reshape(tf.tile(aug_h, [1, self.negative_num]), shape=[h.shape[0], self.negative_num, h.shape[1]])
        neg = tf.math.reduce_sum(tf.math.multiply(aug_h, neg_h), axis=2)
        logits = tf.concat((pos, neg), axis=1)
        ones = np.ones(shape=(h.shape[0], 1), dtype=np.float32)
        zeros = np.zeros(shape=(h.shape[0], self.negative_num), dtype=np.float32)
        labels = np.concatenate((ones, zeros), axis=1)
        loss = tf.math.reduce_mean(tf.math.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits), axis=1))
        return loss

    def instance_hinge_loss_fn(self, aug_h, h, neg_h):
        pos = tf.reshape(tf.math.reduce_sum(tf.math.multiply(aug_h, h), axis=1), shape=(-1, 1))
        aug_h = tf.reshape(tf.tile(aug_h, [1, self.negative_num]), shape=[h.shape[0], self.negative_num, h.shape[1]])
        neg = tf.math.reduce_sum(tf.math.multiply(aug_h, neg_h), axis=2)

        loss = tf.math.reduce_mean(tf.math.reduce_sum(tf.math.maximum(0, neg-pos), axis=-1), axis=-1)
        return loss


    def simaese_loss_fn(self, aug_h, h, neg_h):
        pos_logits = tf.math.reduce_sum(tf.math.multiply(tf.math.abs(aug_h - h), self.siamese_w), axis=-1)
        pos_labels = np.ones(shape=(h.shape[0],), dtype=np.float32)

        aug_h = tf.reshape(tf.tile(aug_h, [1, self.negative_num]), shape=[-1, self.emb_dim])


        neg_h = tf.reshape(neg_h, shape=[-1, self.emb_dim])
        neg_logits = tf.reduce_sum(tf.multiply(tf.math.abs(aug_h-neg_h), self.siamese_w), axis=-1)
        neg_labels = np.zeros(shape=(aug_h.shape[0],), dtype=np.float32)

        logits = tf.concat([pos_logits, neg_logits], axis=0)
        labels = np.concatenate([pos_labels, neg_labels], axis=0)

        loss = tf.math.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=labels, logits=logits, pos_weight=self.siamese_pos_w))
        return loss

    #variable version Error!
    def skipgram_loss_fn(self, word_voc_emb, cont_voc_emb, X, Y):
        X = tf.squeeze(X)
        Y = tf.squeeze(Y)
        centor_embedding = tf.nn.embedding_lookup(word_voc_emb, X)
        negative_index = np.random.randint(low=0, high=self.node_num, size=(Y.shape[0], self.dw_neg_num))
        context_embedding = tf.nn.embedding_lookup(cont_voc_emb, Y)
        neg_context_embedding = tf.nn.embedding_lookup(cont_voc_emb, negative_index)
        pos_logits = tf.expand_dims(tf.math.reduce_sum(tf.math.multiply(centor_embedding, context_embedding), axis=-1), axis=1)
        neg_logits = tf.math.reduce_sum(tf.math.multiply(tf.expand_dims(centor_embedding, axis=1), neg_context_embedding), axis=-1)
        logits = tf.concat((pos_logits, neg_logits), axis=1)
        ones = np.ones(shape=(Y.shape[0], 1), dtype=np.float32)
        zeros = np.zeros(shape=(Y.shape[0], self.dw_neg_num), dtype=np.float32)
        labels = np.concatenate((ones, zeros), axis=1)
        loss = tf.math.reduce_mean(tf.math.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits), axis=1))
        return loss

    #def skipgram_loss_fn(self, centor_embedding, context_embedding, input_num):
    #    negative_index = np.random.randint(low=0, high=self.node_num, size=(input_num, self.dw_neg_num))
    #    neg_context_embedding = tf.squeeze(self.emb_layer(negative_index))
    #    pos_logits = tf.expand_dims(tf.math.reduce_sum(tf.math.multiply(centor_embedding, context_embedding), axis=-1), axis=1)
    #    neg_logits = tf.math.reduce_sum(tf.math.multiply(tf.expand_dims(centor_embedding, axis=1), neg_context_embedding), axis=-1)
    #    logits = tf.concat((pos_logits, neg_logits), axis=1)
    #    ones = np.ones(shape=(input_num, 1), dtype=np.float32)
    #    zeros = np.zeros(shape=(input_num, self.dw_neg_num), dtype=np.float32)
    #    labels = np.concatenate((ones, zeros), axis=1)
    #    loss = tf.math.reduce_mean(tf.math.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits), axis=1))
    #    return loss

    def run_emb(self):
        all_node_idx = np.array(range(self.node_num), dtype=np.int)
        self.embedding = tf.squeeze(self.emb_layer(all_node_idx))
        if self.model_name == 'DeepwalkGaussian':
            self.std = self.std_enc(self.embedding)
            aug_h = self.embedding + tf.random.normal(shape=[self.node_num, self.emb_dim]) * self.std
        elif self.model_name in ['DeepwalkCLR', 'DeepwalkSiamese']:
            aug_noise = self.noise_op(self.node_num)
            aug_h = self.ado_op(np.concatenate((self.embedding, aug_noise), axis=1))
        return self.embedding, aug_h




    def call(self, inputs, training=None):
        X, Y, negative_index, idx = inputs

        all_node_idx = np.array(range(self.node_num), dtype=np.int)
        self.embedding = tf.squeeze(self.emb_layer(all_node_idx))
        #self.cont_embedding = tf.squeeze(self.cont_emb_layer(all_node_idx))

        #build skipgram using nce loss
        #input_x = tf.nn.embedding_lookup(self.embedding, X)
        #input_y = Y
        #self.dw_loss = tf.math.reduce_mean(tf.nn.nce_loss(
        #    weights=self.cont_embedding,
        #    biases=self.nce_bias,
        #    #biases=None,
        #    labels=input_y,
        #    inputs=input_x,
        #    num_sampled=self.dw_neg_num,
        #    num_classes=self.node_num,
        #))

        #build skipgram
        #self.dw_loss = self.skipgram_loss_fn(self.embedding, self.cont_embedding, X, Y)
        self.dw_loss = self.skipgram_loss_fn(self.embedding, self.embedding, X, Y)



        loss = 0.
        self.siamese_loss = 0.
        self.instance_loss = 0.
        self.aug_dw_loss = 0.


        batch_size = len(idx)
        if self.model_name == 'DeepwalkGaussian':
            self.std = self.std_enc(self.embedding)
            aug_h = self.embedding + tf.random.normal(shape=[self.node_num, self.emb_dim]) * self.std
        elif self.model_name in ['DeepwalkCLR', 'DeepwalkSiamese']:
            aug_noise = self.noise_op(self.node_num)
            aug_h = self.ado_op(np.concatenate((self.embedding, aug_noise), axis=1))


        if self.model_name == 'Deepwalk':
            loss = self.dw_loss
        else:
            batch_embedding = tf.nn.embedding_lookup(self.embedding, idx)
            batch_aug_embdding = tf.nn.embedding_lookup(aug_h, idx)
            #negative_index = np.random.randint(low=0, high=node_num, size=(node_num*self.augment_num, self.negative_num))

            negative_samples = tf.nn.embedding_lookup(params=self.embedding, ids=negative_index)
            if self.model_name in ['DeepwalkGaussian', 'DeepwalkCLR']:
                #norm_aug_h = tf.keras.utils.normalize(tf.nn.embedding_lookup(aug_h, idx))
                #norm_batch_embedding = tf.keras.utils.normalize(batch_embedding)
                #norm_neg_emb = tf.keras.utils.normalize(negative_samples)

                #self.instance_loss = self.instance_classify_loss_fn(norm_aug_h, norm_batch_embedding, norm_neg_emb) * self.ins_loss_w
                self.instance_loss = self.instance_classify_loss_sig_fn(aug_h, self.embedding, negative_samples) * self.ins_loss_w
            else:
                self.siamese_loss = self.simaese_loss_fn(batch_aug_embdding, batch_embedding, negative_samples) * self.siamese_loss_w

            #input_aug_x = tf.nn.embedding_lookup(aug_h, X)
            #self.aug_dw_loss = tf.math.reduce_mean(tf.nn.nce_loss(
            #    weights=self.cont_embedding,
            #    biases=self.nce_bias,
            #    #biases=None,
            #    labels=input_y,
            #    inputs=input_aug_x,
            #    num_sampled=self.dw_neg_num,
            #    num_classes=self.node_num,
            #)) * self.aug_dw_loss_w
            #self.aug_dw_loss = self.skipgram_loss_fn(aug_h, self.cont_embedding, X, Y) * self.aug_dw_loss_w
            self.aug_dw_loss = self.skipgram_loss_fn(aug_h, self.embedding, X, Y) * self.aug_dw_loss_w
            loss = self.dw_loss + self.instance_loss + self.aug_dw_loss + self.siamese_loss

        self.add_loss(loss)

        total_loss = tf.add_n(self.losses)

        loss = (total_loss, self.dw_loss, self.aug_dw_loss, self.instance_loss, self.siamese_loss)

        return loss, self.embedding

#useless functino
#    def instance_classify_loss_fn(self, aug_h, h, neg_h):
#        #numerator = tf.math.reduce_sum(tf.math.multiply(aug_h, h), axis=1)
#        numerator = tf.math.reduce_sum(tf.math.multiply(aug_h, h), axis=1) / self.temperature
#        aug_h = tf.reshape(tf.tile(aug_h, [1, self.negative_num]), shape=[h.shape[0], self.negative_num, h.shape[1]])
#       #denominator = tf.math.log(tf.math.reduce_sum(tf.math.exp(tf.math.reduce_sum(tf.math.multiply(aug_h, neg_h), axis=2) / self.temperature), axis=1))
#        denominator = tf.math.log(tf.math.reduce_sum(tf.math.exp(tf.math.reduce_sum(tf.math.multiply(aug_h, neg_h[0]), axis=2) / self.temperature), axis=1) +
#                                  #tf.math.reduce_sum(tf.math.exp(tf.math.reduce_sum(tf.math.multiply(aug_h, neg_h[1]), axis=2) / self.temperature), axis=1) +
#                                  tf.math.exp(numerator))
#        #(N,K,D)*(N,K,D)#
#
#        loss = -numerator + denominator
#        loss = tf.math.reduce_mean(loss)
#        return loss

