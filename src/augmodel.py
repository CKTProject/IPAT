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
            "The average loss for epoch {} is {:7.2f} ".format(
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


class FowardR(keras.Model):
    """Graph contrastive learning with data augmentation operator learned from dataset"""

    def __init__(self, input_dim, gnn_units_list, sparse_inputs=False, act=tf.nn.relu, dropout=0., bias=False, l2_r=1e-5,
                 negative_num=10, aug_gae_loss_w=1e-5, ins_loss_w=1e-5, learning_rate=0.01, temperature=0.07, augment_num=1,
                 norm=1e-1, adj_pos_weight=1.0, norm_loss_w=-0.1, hinge_loss_w=0.,
                 **kwargs):
        super(FowardR, self).__init__(**kwargs)

        self.aug_gae_loss_w = aug_gae_loss_w
        self.ins_loss_w = ins_loss_w
        self.temperature = temperature
        self.augment_num = augment_num
        self.norm = norm
        self.adj_pos_weight = adj_pos_weight
        self.sparse_inputs = sparse_inputs

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

    def GAE_loss_fn_large(self, gae_h, h, row, col, negs):
        #h = tf.keras.utils.normalize(h)
        #cont_h = tf.keras.utils.normalize(cont_h)

        row_h = tf.reshape(tf.nn.embedding_lookup(gae_h, row), [-1, self.emb_dim])
        col_h = tf.reshape(tf.nn.embedding_lookup(h, col), [-1, self.emb_dim])
        pos_logits = tf.reshape(tf.math.reduce_sum(tf.multiply(row_h, col_h), axis=-1), [-1, 1])

        negative_num = negs.shape[1]
        neg_h = tf.nn.embedding_lookup(h, negs)
        aug_h = tf.reshape(tf.tile(row_h, [1, negative_num]),
                shape=[row_h.shape[0], negative_num, row_h.shape[1]])

        neg_logits = tf.math.reduce_sum(tf.math.multiply(aug_h, neg_h), axis=2)

        logits = tf.concat((pos_logits, neg_logits), axis=1)
        ones = np.ones(shape=(row_h.shape[0], 1), dtype=np.float64)
        zeros = np.zeros(shape=(row_h.shape[0], negative_num), dtype=np.float64)
        labels = np.concatenate((ones, zeros), axis=1)
        #loss = tf.math.reduce_mean(tf.math.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits), axis=1))
        loss = tf.math.reduce_mean(tf.math.reduce_sum(tf.nn.weighted_cross_entropy_with_logits(labels=labels, logits=logits, pos_weight=negative_num), axis=1))

        return loss



    def instance_classify_loss_sig_fn(self, aug_h, h, neg_h):
        pos = tf.reshape(tf.math.reduce_sum(tf.math.multiply(aug_h, h), axis=1), shape=(-1, 1))/ self.temperature
        aug_h = tf.reshape(tf.tile(aug_h, [1, self.negative_num]), shape=[h.shape[0], self.negative_num, h.shape[1]])
        neg = tf.math.reduce_sum(tf.math.multiply(aug_h, neg_h), axis=2)/ self.temperature
        logits = tf.concat((pos, neg), axis=1)
        ones = np.ones(shape=(h.shape[0], 1), dtype=np.float64)
        zeros = np.zeros(shape=(h.shape[0], self.negative_num), dtype=np.float64)
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

    def run_embedding(self, inputs):
        x, adj = inputs
        self.h = self.graph_enc((x, adj))
        return self.h

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
        if len(inputs) > 4:
        #if False:

            x, adj, gradint_dir, select_index, id2id, new_rows, cols, negs, id_negs = inputs 
            gradint_dir = tf.keras.utils.normalize(gradint_dir)


            self.h = self.graph_enc((x, adj), training)
            self.gae_h = tf.nn.embedding_lookup(self.h, select_index)

            self.gae_loss = self.GAE_loss_fn_large(self.gae_h, self.h, new_rows, cols, negs)

            self.std = self.std_enc(self.gae_h)

            self.aug_gae_loss = 0.
            self.instance_loss = 0.
            self.hinge_loss = 0.
            self.norm_loss = 0.

            gradint_dir = tf.multiply(gradint_dir, self.std)
            aug_h = self.gae_h + gradint_dir

            self.aug_gae_loss += self.GAE_loss_fn_large(aug_h, self.h, new_rows, cols, negs) * self.aug_gae_loss_w

            negative_samples = tf.nn.embedding_lookup(params=self.h, ids=id_negs)
            self.instance_loss += self.instance_classify_loss_sig_fn(aug_h, self.gae_h, negative_samples) * self.ins_loss_w

            loss = self.gae_loss + self.aug_gae_loss + self.instance_loss + self.hinge_loss + self.norm_loss
            self.add_loss(loss)

            total_loss = tf.add_n(self.losses)
            loss = (total_loss, self.gae_loss, self.aug_gae_loss, self.instance_loss, self.hinge_loss, self.norm_loss)

            return loss, self.gae_h, aug_h

        x, adj, adj_orig, gradint_dir = inputs
        gradint_dir = tf.keras.utils.normalize(gradint_dir)

        if self.sparse_inputs:
            node_num = x.dense_shape[0]
        else:
            node_num = x.shape[0]

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

        gradint_dir = tf.multiply(gradint_dir, self.std)

        for i in range(self.augment_num):
            aug_h = self.h + gradint_dir
            #aug_h = self.h + tf.keras.utils.normalize(tf.random.normal(shape=[node_num, self.emb_dim])) * self.std
            #aug_h = self.h + tf.random.normal(shape=[node_num, self.emb_dim]) * self.std
            #aug_h = self.h + unit_sphere_sample(shape=[node_num, self.emb_dim]) * self.std

            aug_list.append(aug_h)
            if len(inputs) > 4:
                self.aug_gae_loss += self.GAE_loss_fn_large(aug_h, self.h, rows, cols, negs) * self.aug_gae_loss_w
            else:
                self.aug_gae_loss += self.GAE_loss_fn(aug_h, self.h, adj_orig) * self.aug_gae_loss_w

            negative_index = np.random.randint(low=0, high=node_num, size=(node_num, self.negative_num))
            negative_samples = tf.nn.embedding_lookup(params=self.h, ids=negative_index)
            #max radius of each node
            #nb_negative_samples = tf.nn.embedding_lookup(params=self.h, ids=nb_negative_index)
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

class FixR(keras.Model):
    """Graph contrastive learning with data augmentation operator learned from dataset"""

    def __init__(self, input_dim, gnn_units_list, sparse_inputs=False, act=tf.nn.relu, dropout=0., bias=False, l2_r=1e-5,
                 negative_num=10, aug_gae_loss_w=1e-5, ins_loss_w=1e-5, learning_rate=0.01, temperature=0.07, augment_num=1,
                 norm=1e-1, adj_pos_weight=1.0, norm_loss_w=-0.1, hinge_loss_w=0.,
                 **kwargs):
        super(FixR, self).__init__(**kwargs)

        self.aug_gae_loss_w = aug_gae_loss_w
        self.ins_loss_w = ins_loss_w
        self.temperature = temperature
        self.augment_num = augment_num
        self.norm = norm
        self.adj_pos_weight = adj_pos_weight
        self.sparse_inputs = sparse_inputs

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

        self.negative_num = negative_num
        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        #self.optimizer = tf.keras.optimizer.RMSprop(learning_rate=self.learning_rate)


    def GAE_loss_fn(self, h, cont_h, adj_orig):

        reconstructions = tf.reshape(tf.matmul(h, tf.transpose(cont_h)), [-1])
        labels = tf.reshape(tf.sparse.to_dense(adj_orig, validate_indices=False), [-1])
        loss = self.norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=reconstructions, labels=labels, pos_weight=self.adj_pos_weight))
        return loss


    def GAE_loss_fn_large(self, gae_h, h, row, col, negs):
        #h = tf.keras.utils.normalize(h)
        #cont_h = tf.keras.utils.normalize(cont_h)

        row_h = tf.reshape(tf.nn.embedding_lookup(gae_h, row), [-1, self.emb_dim])
        col_h = tf.reshape(tf.nn.embedding_lookup(h, col), [-1, self.emb_dim])
        pos_logits = tf.reshape(tf.math.reduce_sum(tf.multiply(row_h, col_h), axis=-1), [-1, 1])

        negative_num = negs.shape[1]
        neg_h = tf.nn.embedding_lookup(h, negs)
        aug_h = tf.reshape(tf.tile(row_h, [1, negative_num]),
                shape=[row_h.shape[0], negative_num, row_h.shape[1]])

        neg_logits = tf.math.reduce_sum(tf.math.multiply(aug_h, neg_h), axis=2)

        logits = tf.concat((pos_logits, neg_logits), axis=1)
        ones = np.ones(shape=(row_h.shape[0], 1), dtype=np.float64)
        zeros = np.zeros(shape=(row_h.shape[0], negative_num), dtype=np.float64)
        labels = np.concatenate((ones, zeros), axis=1)
        #loss = tf.math.reduce_mean(tf.math.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits), axis=1))
        loss = tf.math.reduce_mean(tf.math.reduce_sum(tf.nn.weighted_cross_entropy_with_logits(labels=labels, logits=logits, pos_weight=negative_num), axis=1))

        return loss

    def instance_classify_loss_sig_fn(self, aug_h, h, neg_h):
        pos = tf.reshape(tf.math.reduce_sum(tf.math.multiply(aug_h, h), axis=1), shape=(-1, 1))/ self.temperature
        aug_h = tf.reshape(tf.tile(aug_h, [1, self.negative_num]), shape=[h.shape[0], self.negative_num, h.shape[1]])
        neg = tf.math.reduce_sum(tf.math.multiply(aug_h, neg_h), axis=2)/ self.temperature
        logits = tf.concat((pos, neg), axis=1)
        ones = np.ones(shape=(h.shape[0], 1), dtype=np.float64)
        zeros = np.zeros(shape=(h.shape[0], self.negative_num), dtype=np.float64)
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
        #x, adj = inputs
        #h = self.graph_enc((x, adj))
        #std = self.std_enc(h)
        return self.std

    def run_embedding(self, inputs):
        x, adj = inputs
        self.h = self.graph_enc((x, adj))
        return self.h

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
        if len(inputs) > 4:
        #if False:

            x, adj, gradint_dir, select_index, id2id, new_rows, cols, negs, id_negs = inputs 
            gradint_dir = tf.keras.utils.normalize(gradint_dir)


            self.h = self.graph_enc((x, adj), training)
            self.gae_h = tf.nn.embedding_lookup(self.h, select_index)

            self.gae_loss = self.GAE_loss_fn_large(self.gae_h, self.h, new_rows, cols, negs)

            self.std = 1.0

            self.aug_gae_loss = 0.
            self.instance_loss = 0.
            self.hinge_loss = 0.
            self.norm_loss = 0.

            gradint_dir = tf.multiply(gradint_dir, self.std)
            aug_h = self.gae_h + gradint_dir

            self.aug_gae_loss += self.GAE_loss_fn_large(aug_h, self.h, new_rows, cols, negs) * self.aug_gae_loss_w

            negative_samples = tf.nn.embedding_lookup(params=self.h, ids=id_negs)
            self.instance_loss += self.instance_classify_loss_sig_fn(aug_h, self.gae_h, negative_samples) * self.ins_loss_w

            loss = self.gae_loss + self.aug_gae_loss + self.instance_loss + self.hinge_loss + self.norm_loss
            self.add_loss(loss)

            total_loss = tf.add_n(self.losses)
            loss = (total_loss, self.gae_loss, self.aug_gae_loss, self.instance_loss, self.hinge_loss, self.norm_loss)

            return loss, self.gae_h, aug_h


        #x, adj, adj_orig, nb_negative_index = inputs
        if len(inputs) > 4:
            x, adj, gradint_dir, rows, cols, negs = inputs 
        else:
            x, adj, adj_orig, gradint_dir = inputs

        gradint_dir = tf.keras.utils.normalize(gradint_dir)

        if self.sparse_inputs:
            node_num = x.dense_shape[0]
        else:
            node_num = x.shape[0]

        #step in trainin

        #get node representation from deep GNN and graph representaion from readout function
        self.h = self.graph_enc((x, adj), training)
        #self.h = tf.keras.utils.normalize(self.h)

        if len(inputs) > 4:
            self.gae_loss = self.GAE_loss_fn_large(self.h, self.h, rows, cols, negs)
        else:
            self.gae_loss = self.GAE_loss_fn(self.h, self.h, adj_orig)


        self.std = 1.0
        #self.std = 0.0018
        aug_list = []
        self.aug_gae_loss = 0.
        self.instance_loss = 0.
        self.hinge_loss = 0.
        self.norm_loss = 0.

        gradint_dir = tf.multiply(gradint_dir, self.std)

        for i in range(self.augment_num):
            aug_h = self.h + gradint_dir
            #aug_h = self.h + tf.keras.utils.normalize(tf.random.normal(shape=[node_num, self.emb_dim])) * self.std
            #aug_h = self.h + tf.random.normal(shape=[node_num, self.emb_dim]) * self.std
            #aug_h = self.h + unit_sphere_sample(shape=[node_num, self.emb_dim]) * self.std

            aug_list.append(aug_h)
            if len(inputs) > 4:
                self.aug_gae_loss += self.GAE_loss_fn_large(aug_h, self.h, rows, cols, negs)
            else:
                self.aug_gae_loss += self.GAE_loss_fn(aug_h, self.h, adj_orig) * self.aug_gae_loss_w

            negative_index = np.random.randint(low=0, high=node_num, size=(node_num, self.negative_num))
            negative_samples = tf.nn.embedding_lookup(params=self.h, ids=negative_index)
            #max radius of each node
            #nb_negative_samples = tf.nn.embedding_lookup(params=self.h, ids=nb_negative_index)
            #nb_negative_samples = negative_samples
            #self.norm_loss += self.cal_norm_loss_fn(aug_h, nb_negative_samples, tf.squeeze(tf.nn.embedding_lookup(self.std, nb_negative_index))) * self.norm_loss_w

            #calculate instance discriminitor loss
            
            #self.instance_loss += self.instance_classify_loss_sig_fn(aug_h, self.h, negative_samples) * self.ins_loss_w

            #self.hinge_loss += self.instance_hinge_loss_fn(aug_h, self.h, negative_samples) * self.hinge_loss_w

        #aug_h = tf.concat(aug_list, axis=0)
        aug_h = tf.add_n(aug_list) / self.augment_num


        #self.norm_loss = -tf.math.reduce_mean(tf.nn.l2_loss(self.std)) * self.norm_loss_w
        #self.norm_loss = -tf.math.reduce_mean(self.std) * self.norm
        #self.norm_loss = tf.math.reduce_mean(1.0 - self.std) * self.norm_loss_w

        loss = self.gae_loss + self.aug_gae_loss + self.instance_loss + self.hinge_loss + self.norm_loss
        self.add_loss(loss)

        total_loss = tf.add_n(self.losses)
        #total_loss = self.gae_loss
        loss = (total_loss, self.gae_loss, self.aug_gae_loss, self.instance_loss, self.hinge_loss, self.norm_loss)
        #h_list = (self.h, aug_h)

        #return loss, self.h, aug_h, tf.math.reduce_mean(tf.math.reduce_sum(tf.math.exp(self.log_std), axis=-1))
        #return loss, self.h, aug_h, tf.math.reduce_mean(tf.math.reduce_sum(self.std, axis=-1))
        return loss, self.h, aug_h



class AdvR(keras.Model):
    """Graph contrastive learning with data augmentation operator learned from dataset"""

    def __init__(self, input_dim, gnn_units_list, sparse_inputs=False, act=tf.nn.relu, dropout=0., bias=False, l2_r=1e-5,
                 negative_num=10, aug_gae_loss_w=1e-5, ins_loss_w=1e-5, learning_rate=0.01, temperature=0.07, augment_num=1,
                 norm=1e-1, adj_pos_weight=1.0, norm_loss_w=-0.1, hinge_loss_w=0.,
                 **kwargs):
        super(AdvR, self).__init__(**kwargs)

        self.aug_gae_loss_w = aug_gae_loss_w
        self.ins_loss_w = ins_loss_w
        self.temperature = temperature
        self.augment_num = augment_num
        self.norm = norm
        self.adj_pos_weight = adj_pos_weight
        self.sparse_inputs = sparse_inputs

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


        self.negative_num = negative_num
        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        #self.optimizer = tf.keras.optimizer.RMSprop(learning_rate=self.learning_rate)


    def GAE_loss_fn(self, h, cont_h, adj_orig):

        reconstructions = tf.reshape(tf.matmul(h, tf.transpose(cont_h)), [-1])
        labels = tf.reshape(tf.sparse.to_dense(adj_orig, validate_indices=False), [-1])
        loss = self.norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=reconstructions, labels=labels, pos_weight=self.adj_pos_weight))
        return loss

    def GAE_adv_loss_fn(self, h, grad, std, adj_orig):

        reconstructions = tf.matmul(h, tf.transpose(h))
        reconstructions = reconstructions + 2.0 * tf.math.multiply(tf.matmul(h, tf.transpose(grad)), std)
        reconstructions = reconstructions + std * std * tf.matmul(grad, tf.transpose(grad))
        reconstructions = tf.reshape(reconstructions, [-1])
        labels = tf.reshape(tf.sparse.to_dense(adj_orig, validate_indices=False), [-1])
        loss = self.norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=reconstructions, labels=labels, pos_weight=self.adj_pos_weight))
        return loss




    def instance_classify_loss_sig_fn(self, aug_h, h, neg_h):
        pos = tf.reshape(tf.math.reduce_sum(tf.math.multiply(aug_h, h), axis=1), shape=(-1, 1))/ self.temperature
        aug_h = tf.reshape(tf.tile(aug_h, [1, self.negative_num]), shape=[h.shape[0], self.negative_num, h.shape[1]])
        neg = tf.math.reduce_sum(tf.math.multiply(aug_h, neg_h), axis=2)/ self.temperature
        logits = tf.concat((pos, neg), axis=1)
        ones = np.ones(shape=(h.shape[0], 1), dtype=np.float64)
        zeros = np.zeros(shape=(h.shape[0], self.negative_num), dtype=np.float64)
        labels = np.concatenate((ones, zeros), axis=1)
        loss = tf.math.reduce_mean(tf.math.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits), axis=1))
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

    def run_embedding(self, inputs):
        x, adj = inputs
        self.h = self.graph_enc((x, adj))
        return self.h

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
        #x, adj, adj_orig, nb_negative_index = inputs
        x, adj, adj_orig, gradint_dir, std = inputs

        gradint_dir = tf.keras.utils.normalize(gradint_dir)

        if self.sparse_inputs:
            node_num = x.dense_shape[0]
        else:
            node_num = x.shape[0]

        #step in trainin

        #get node representation from deep GNN and graph representaion from readout function
        self.h = self.graph_enc((x, adj), training)
        #self.h = tf.keras.utils.normalize(self.h)

        self.gae_loss = self.GAE_loss_fn(self.h, self.h, adj_orig)


        self.std = std
        aug_list = []
        self.aug_gae_loss = 0.
        self.instance_loss = 0.
        self.hinge_loss = 0.
        self.norm_loss = 0.


        for i in range(self.augment_num):
            self.aug_gae_loss += self.GAE_adv_loss_fn(self.h, gradint_dir, std, adj_orig) * self.aug_gae_loss_w

        
        loss = self.gae_loss + self.aug_gae_loss
        self.add_loss(loss)

        total_loss = tf.add_n(self.losses)

        loss = (total_loss, self.gae_loss, self.aug_gae_loss)

        #debug
        aug_h = self.h + std * gradint_dir

        return loss, self.h, aug_h


class HyperR(keras.Model):
    """Graph contrastive learning with data augmentation operator learned from dataset"""

    def __init__(self, input_dim, gnn_units_list, sparse_inputs=False, act=tf.nn.relu, dropout=0., bias=False, l2_r=1e-5,
                 negative_num=10, aug_gae_loss_w=1e-5, ins_loss_w=1e-5, learning_rate=0.01, temperature=0.07, augment_num=1,
                 norm=1e-1, adj_pos_weight=1.0, norm_loss_w=-0.1, hinge_loss_w=0.,
                 **kwargs):
        super(HyperR, self).__init__(**kwargs)

        self.aug_gae_loss_w = aug_gae_loss_w
        self.ins_loss_w = ins_loss_w
        self.temperature = temperature
        self.augment_num = augment_num
        self.norm = norm
        self.adj_pos_weight = adj_pos_weight
        self.sparse_inputs = sparse_inputs

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
        ones = np.ones(shape=(h.shape[0], 1), dtype=np.float64)
        zeros = np.zeros(shape=(h.shape[0], self.negative_num), dtype=np.float64)
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

    def run_embedding(self, inputs):
        x, adj = inputs
        self.h = self.graph_enc((x, adj))
        return self.h

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
        #x, adj, adj_orig, nb_negative_index = inputs
        x, adj, adj_orig, gradint_dir, std = inputs

        gradint_dir = tf.keras.utils.normalize(gradint_dir)

        if self.sparse_inputs:
            node_num = x.dense_shape[0]
        else:
            node_num = x.shape[0]

        #step in trainin

        #get node representation from deep GNN and graph representaion from readout function
        self.h = self.graph_enc((x, adj), training)
        #self.h = tf.keras.utils.normalize(self.h)

        self.gae_loss = self.GAE_loss_fn(self.h, self.h, adj_orig)


        self.std = std
        #self.std = 0.0018
        aug_list = []
        self.aug_gae_loss = 0.
        self.instance_loss = 0.
        self.hinge_loss = 0.
        self.norm_loss = 0.

        gradint_dir = tf.multiply(gradint_dir, self.std)

        for i in range(self.augment_num):
            aug_h = self.h + gradint_dir
            #aug_h = self.h + tf.keras.utils.normalize(tf.random.normal(shape=[node_num, self.emb_dim])) * self.std
            #aug_h = self.h + tf.random.normal(shape=[node_num, self.emb_dim]) * self.std
            #aug_h = self.h + unit_sphere_sample(shape=[node_num, self.emb_dim]) * self.std

            aug_list.append(aug_h)
            self.aug_gae_loss += self.GAE_loss_fn(aug_h, self.h, adj_orig) * self.aug_gae_loss_w

            negative_index = np.random.randint(low=0, high=node_num, size=(node_num, self.negative_num))
            negative_samples = tf.nn.embedding_lookup(params=self.h, ids=negative_index)
            #max radius of each node
            #nb_negative_samples = tf.nn.embedding_lookup(params=self.h, ids=nb_negative_index)
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

class LearnR(keras.Model):
    """Graph contrastive learning with data augmentation operator learned from dataset"""

    def __init__(self, input_dim, gnn_units_list, sparse_inputs=False, act=tf.nn.relu, dropout=0., bias=False, l2_r=1e-5,
                 negative_num=10, aug_gae_loss_w=1e-5, ins_loss_w=1e-5, learning_rate=0.01, temperature=0.07, augment_num=1,
                 norm=1e-1, adj_pos_weight=1.0, norm_loss_w=-0.1, hinge_loss_w=0.,
                 **kwargs):
        super(LearnR, self).__init__(**kwargs)

        self.aug_gae_loss_w = aug_gae_loss_w
        self.ins_loss_w = ins_loss_w
        self.temperature = temperature
        self.augment_num = augment_num
        self.norm = norm
        self.adj_pos_weight = adj_pos_weight

        self.sparse_inputs = sparse_inputs

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
        ones = np.ones(shape=(h.shape[0], 1), dtype=np.float64)
        zeros = np.zeros(shape=(h.shape[0], self.negative_num), dtype=np.float64)
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

    def learn_radius(self, h, n):
        S = tf.math.sigmoid(tf.matmul(h, tf.transpose(h)))
        S = tf.nn.softmax(S, axis=1)
        S = tf.math.log(S) - np.log(1.0/float(n))
        pmi_S = tf.reduce_max(tf.math.maximum(S, tf.zeros(shape=(n,n), dtype=tf.float64)), axis=1)
        max_pmi = tf.math.reduce_max(pmi_S)
        radius = 1.0 - pmi_S/max_pmi
        radius = tf.reshape(radius, (-1, 1))
        return radius

    def run_embedding(self, inputs):
        x, adj = inputs
        self.h = self.graph_enc((x, adj))
        return self.h


    def call(self, inputs, training=None):
        #x, adj, adj_orig, nb_negative_index = inputs
        x, adj, adj_orig, gradint_dir = inputs

        gradint_dir = tf.keras.utils.normalize(gradint_dir)

        if self.sparse_inputs:
            node_num = x.dense_shape[0]
        else:
            node_num = x.shape[0]

        #step in trainin

        #get node representation from deep GNN and graph representaion from readout function
        self.h = self.graph_enc((x, adj), training)
        #self.h = tf.keras.utils.normalize(self.h)

        self.gae_loss = self.GAE_loss_fn(self.h, self.h, adj_orig)


        self.std = self.learn_radius(self.h, node_num)
        #self.std = 0.0018
        aug_list = []
        self.aug_gae_loss = 0.
        self.instance_loss = 0.
        self.hinge_loss = 0.
        self.norm_loss = 0.

        gradint_dir = tf.multiply(gradint_dir, self.std)

        for i in range(self.augment_num):
            aug_h = self.h + gradint_dir
            #aug_h = self.h + tf.keras.utils.normalize(tf.random.normal(shape=[node_num, self.emb_dim])) * self.std
            #aug_h = self.h + tf.random.normal(shape=[node_num, self.emb_dim]) * self.std
            #aug_h = self.h + unit_sphere_sample(shape=[node_num, self.emb_dim]) * self.std

            aug_list.append(aug_h)
            self.aug_gae_loss += self.GAE_loss_fn(aug_h, self.h, adj_orig) * self.aug_gae_loss_w

            negative_index = np.random.randint(low=0, high=node_num, size=(node_num, self.negative_num))
            negative_samples = tf.nn.embedding_lookup(params=self.h, ids=negative_index)
            #max radius of each node
            #nb_negative_samples = tf.nn.embedding_lookup(params=self.h, ids=nb_negative_index)
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

class LearnSimpleR(keras.Model):
    """Graph contrastive learning with data augmentation operator learned from dataset"""

    def __init__(self, input_dim, gnn_units_list, sparse_inputs=False, act=tf.nn.relu, dropout=0., bias=False, l2_r=1e-5,
                 negative_num=10, aug_gae_loss_w=1e-5, ins_loss_w=1e-5, learning_rate=0.01, temperature=0.07, augment_num=1,
                 norm=1e-1, adj_pos_weight=1.0, norm_loss_w=-0.1, hinge_loss_w=0.,
                 **kwargs):
        super(LearnSimpleR, self).__init__(**kwargs)

        self.aug_gae_loss_w = aug_gae_loss_w
        self.ins_loss_w = ins_loss_w
        self.temperature = temperature
        self.augment_num = augment_num
        self.norm = norm
        self.adj_pos_weight = adj_pos_weight

        self.sparse_inputs = sparse_inputs

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

        #self.std_enc = keras.layers.Dense(units=1,
        #                                                   #activation=act,
        #                                                   #activation = 'sigmoid',
        #                                                   activation=tf.nn.relu,
        #                                                   use_bias=False,
        #                                                   kernel_initializer='glorot_uniform',
        #                                                   bias_initializer='zeros',
        #                                                   kernel_regularizer=tf.keras.regularizers.L2(l2_r),
        #                                                   bias_regularizer=tf.keras.regularizers.L2(l2_r),
        #                                                   )

        self.emb_dim = gnn_units_list[-1]



        self.negative_num = negative_num
        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        #self.optimizer = tf.keras.optimizer.RMSprop(learning_rate=self.learning_rate)


    def GAE_loss_fn(self, h, cont_h, adj_orig):

        reconstructions = tf.reshape(tf.matmul(h, tf.transpose(cont_h)), [-1])
        labels = tf.reshape(tf.sparse.to_dense(adj_orig, validate_indices=False), [-1])
        loss = self.norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=reconstructions, labels=labels, pos_weight=self.adj_pos_weight))
        return loss

    def GAE_loss_fn_large(self, h, cont_h, row, col, negs):
        #h = tf.keras.utils.normalize(h)
        #cont_h = tf.keras.utils.normalize(cont_h)

        row_h = tf.reshape(tf.nn.embedding_lookup(h, row), [-1, self.emb_dim])
        col_h = tf.reshape(tf.nn.embedding_lookup(h, col), [-1, self.emb_dim])
        pos_logits = tf.reshape(tf.math.reduce_sum(tf.multiply(row_h, col_h), axis=-1), [-1, 1])

        negative_num = negs.shape[1]
        neg_h = tf.nn.embedding_lookup(h, negs)
        aug_h = tf.reshape(tf.tile(row_h, [1, negative_num]),
                shape=[row_h.shape[0], negative_num, row_h.shape[1]])

        neg_logits = tf.math.reduce_sum(tf.math.multiply(aug_h, neg_h), axis=2)

        logits = tf.concat((pos_logits, neg_logits), axis=1)
        ones = np.ones(shape=(row_h.shape[0], 1), dtype=np.float64)
        zeros = np.zeros(shape=(row_h.shape[0], negative_num), dtype=np.float64)
        labels = np.concatenate((ones, zeros), axis=1)
        loss = tf.math.reduce_mean(tf.math.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits), axis=1))

        return loss




    def instance_classify_loss_sig_fn(self, aug_h, h, neg_h):
        pos = tf.reshape(tf.math.reduce_sum(tf.math.multiply(aug_h, h), axis=1), shape=(-1, 1))/ self.temperature
        aug_h = tf.reshape(tf.tile(aug_h, [1, self.negative_num]), shape=[h.shape[0], self.negative_num, h.shape[1]])
        neg = tf.math.reduce_sum(tf.math.multiply(aug_h, neg_h), axis=2)/ self.temperature
        logits = tf.concat((pos, neg), axis=1)
        ones = np.ones(shape=(h.shape[0], 1), dtype=np.float64)
        zeros = np.zeros(shape=(h.shape[0], self.negative_num), dtype=np.float64)
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
        std = self.learn_simple(h)
        return std

    def run_embedding(self, inputs):
        x, adj = inputs
        self.h = self.graph_enc((x, adj))
        return self.h

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

    def learn_simple(self, h, training=False):
        S = tf.math.sigmoid(tf.matmul(h, tf.transpose(h)))
        r = tf.math.reduce_max(1.0 - S, axis=1)
        r = tf.reshape(r, shape=(-1, 1))
        return r

    #def learn_simple(self, h, training=False):
    #    r = self.std_enc(h)
    #    return r

    def call(self, inputs, training=None):
        if len(inputs) > 4:
            x, adj, gradint_dir, rows, cols, negs = inputs 
        else:
            x, adj, adj_orig, gradint_dir = inputs

        gradint_dir = tf.keras.utils.normalize(gradint_dir)

        if self.sparse_inputs:
            node_num = x.dense_shape[0]
        else:
            node_num = x.shape[0]

        #step in trainin

        #get node representation from deep GNN and graph representaion from readout function
        self.h = self.graph_enc((x, adj), training)
        #self.h = tf.keras.utils.normalize(self.h)

        if len(inputs) > 4:
            self.gae_loss = self.GAE_loss_fn_large(self.h, self.h, rows, cols, negs)
        else:
            self.gae_loss = self.GAE_loss_fn(self.h, self.h, adj_orig)


        self.std = self.learn_simple(self.h, training)
        #self.std = 0.0018
        aug_list = []
        self.aug_gae_loss = 0.
        self.instance_loss = 0.
        self.hinge_loss = 0.
        self.norm_loss = 0.

        gradint_dir = tf.multiply(gradint_dir, self.std)

        for i in range(self.augment_num):
            aug_h = self.h + gradint_dir
            #aug_h = self.h + tf.keras.utils.normalize(tf.random.normal(shape=[node_num, self.emb_dim])) * self.std
            #aug_h = self.h + tf.random.normal(shape=[node_num, self.emb_dim]) * self.std
            #aug_h = self.h + unit_sphere_sample(shape=[node_num, self.emb_dim]) * self.std

            aug_list.append(aug_h)
            if len(inputs) > 4:
                self.aug_gae_loss = self.GAE_loss_fn_large(self.h, self.h, rows, cols, negs) * self.aug_gae_loss_w
            else:
                self.aug_gae_loss = self.GAE_loss_fn(self.h, self.h, adj_orig) * self.aug_gae_loss_w

            negative_index = np.random.randint(low=0, high=node_num, size=(node_num, self.negative_num))
            negative_samples = tf.nn.embedding_lookup(params=self.h, ids=negative_index)
            #max radius of each node
            #nb_negative_samples = tf.nn.embedding_lookup(params=self.h, ids=nb_negative_index)
            #nb_negative_samples = negative_samples
            #self.norm_loss += self.cal_norm_loss_fn(aug_h, nb_negative_samples, tf.squeeze(tf.nn.embedding_lookup(self.std, nb_negative_index))) * self.norm_loss_w

            #calculate instance discriminitor loss
            
            self.instance_loss += self.instance_classify_loss_sig_fn(aug_h, self.h, negative_samples) * self.ins_loss_w

            self.hinge_loss += self.instance_hinge_loss_fn(aug_h, self.h, negative_samples) * self.hinge_loss_w

        #aug_h = tf.concat(aug_list, axis=0)
        aug_h = tf.add_n(aug_list) / self.augment_num


        #self.norm_loss = -tf.math.reduce_mean(tf.nn.l2_loss(self.std)) * self.norm_loss_w
        #self.norm_loss = -tf.math.reduce_mean(self.std) * self.norm
        self.norm_loss = tf.math.reduce_mean(10.0 - self.std) * self.norm_loss_w

        loss = self.gae_loss + self.aug_gae_loss + self.instance_loss + self.hinge_loss + self.norm_loss
        self.add_loss(loss)

        total_loss = tf.add_n(self.losses)
        #total_loss = self.gae_loss
        loss = (total_loss, self.gae_loss, self.aug_gae_loss, self.instance_loss, self.hinge_loss, self.norm_loss)
        #h_list = (self.h, aug_h)

        #return loss, self.h, aug_h, tf.math.reduce_mean(tf.math.reduce_sum(tf.math.exp(self.log_std), axis=-1))
        #return loss, self.h, aug_h, tf.math.reduce_mean(tf.math.reduce_sum(self.std, axis=-1))
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

    def GAE_loss_fn_large(self, h, cont_h, row, col, negs):
        #h = tf.keras.utils.normalize(h)
        #cont_h = tf.keras.utils.normalize(cont_h)

        row_h = tf.reshape(tf.nn.embedding_lookup(h, row), [-1, self.emb_dim])
        col_h = tf.reshape(tf.nn.embedding_lookup(h, col), [-1, self.emb_dim])
        pos_logits = tf.reshape(tf.math.reduce_sum(tf.multiply(row_h, col_h), axis=-1), [-1, 1])

        negative_num = negs.shape[1]
        neg_h = tf.nn.embedding_lookup(h, negs)
        aug_h1 = tf.reshape(tf.tile(row_h, [1, negative_num]),
                shape=[row_h.shape[0], negative_num, row_h.shape[1]])

        neg_logits1 = tf.math.reduce_sum(tf.math.multiply(aug_h1, neg_h), axis=2)

        aug_h2 = tf.reshape(tf.tile(col_h, [1, negative_num]),
                shape=[row_h.shape[0], negative_num, row_h.shape[1]])

        neg_logits2 = tf.math.reduce_sum(tf.math.multiply(aug_h2, neg_h), axis=2)

        logits = tf.concat((pos_logits, neg_logits1, neg_logits2), axis=1)
        ones = np.ones(shape=(row_h.shape[0], 1), dtype=np.float64)
        zeros = np.zeros(shape=(row_h.shape[0], negative_num*2), dtype=np.float64)
        labels = np.concatenate((ones, zeros), axis=1)
        #loss = tf.math.reduce_mean(tf.math.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits), axis=1))
        loss = tf.math.reduce_mean(tf.math.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=labels, logits=logits, pos_weight=negative_num*2), axis=1))
        #loss = tf.math.reduce_mean(tf.math.reduce_sum(tf.nn.weighted_cross_entropy_with_logits(labels=labels, logits=logits, pos_weight=self.adj_pos_weight), axis=1))

        return loss

    def run_embedding(self, inputs):
        x, adj = inputs
        h = self.graph_enc((x, adj))
        return h

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
        if len(inputs) > 3:
            x, adj, rows, cols, negs = inputs 
            self.h = self.graph_enc((x, adj), training)
            self.gae_loss = self.GAE_loss_fn_large(self.h, self.h, rows, cols, negs)
        else:
            x, adj, adj_orig = inputs
            self.h = self.graph_enc((x, adj), training)
            self.gae_loss = self.GAE_loss_fn(self.h, self.h, adj_orig)

        return self.gae_loss, self.h
