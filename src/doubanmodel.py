from layers import GraphConvolution, InnerProductDecoder, BatchGraphConvolution, Dense
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
            self.layers_list.append(BatchGraphConvolution(input_dim=in_dim,
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
        x, nbx = inputs
        for layer in self.layers_list:
            x = layer((x, nbx), training)
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
                 norm_loss_w=-0.1, hinge_loss_w=0.,
                 **kwargs):
        super(FowardR, self).__init__(**kwargs)

        self.aug_gae_loss_w = aug_gae_loss_w
        self.ins_loss_w = ins_loss_w
        self.temperature = temperature
        self.augment_num = augment_num

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


    def GAE_loss_fn(self, h, nb_h, negative_samples, nb_size, neg_size):
        pos_h = tf.reshape(tf.tile(h, [1, nb_size]), shape=[h.shape[0], nb_size, h.shape[1]])
        pos = tf.reshape(tf.math.reduce_sum(tf.math.multiply(pos_h, nb_h), axis=2), shape=[-1]) / self.temperature
        neg_h = tf.reshape(tf.tile(h, [1, neg_size]), shape=[h.shape[0], neg_size, h.shape[1]])
        neg = tf.reshape(tf.math.reduce_sum(tf.math.multiply(neg_h, negative_samples), axis=2), shape=[-1]) / self.temperature
        logits = tf.concat((pos, neg), axis=0)

        ones = np.ones(shape=(h.shape[0]*nb_size, ), dtype=np.float64)
        zeros = np.zeros(shape=(h.shape[0]*neg_size, ), dtype=np.float64)
        labels = np.concatenate((ones, zeros), axis=0)
        loss = tf.math.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
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
        x, nbx = inputs
        self.h = self.graph_enc((x, nbx))
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
        x, nbx, nbnbx, gradint_dir = inputs
        #x (batch_size, f), nbx (batch_size, nb_size, f), nbnb (batch_size*nb_size, nb_size, f)

        in_dim = nbx.shape[2]
        nb_size = nbx.shape[1]
        batch_size = nbx.shape[0]

        gradint_dir = tf.keras.utils.normalize(gradint_dir)

        #step in trainin

        #get node representation from deep GNN
        self.h = self.graph_enc((x, nbx), training)
        #self.h = tf.keras.utils.normalize(self.h)
        nbx = tf.reshape(nbx, shape=(-1, in_dim))
        self.nb_h = self.graph_enc((nbx, nbnbx), training)
        self.nb_h = tf.reshape(self.nb_h, shape=(batch_size, nb_size, -1))


        negative_index = np.random.randint(low=0, high=batch_size, size=(batch_size, self.negative_num))
        negative_samples = tf.nn.embedding_lookup(params=self.h, ids=negative_index)
        self.gae_loss = self.GAE_loss_fn(self.h, self.nb_h, negative_samples, nb_size, self.negative_num)


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
            self.aug_gae_loss += self.GAE_loss_fn(aug_h, self.nb_h, negative_samples, nb_size, self.negative_num) * self.aug_gae_loss_w

            ins_negative_index = np.random.randint(low=0, high=batch_size, size=(batch_size, self.negative_num))
            ins_negative_samples = tf.nn.embedding_lookup(params=self.h, ids=negative_index)
            #max radius of each node
            #nb_negative_samples = tf.nn.embedding_lookup(params=self.h, ids=nb_negative_index)
            #nb_negative_samples = negative_samples
            #self.norm_loss += self.cal_norm_loss_fn(aug_h, nb_negative_samples, tf.squeeze(tf.nn.embedding_lookup(self.std, nb_negative_index))) * self.norm_loss_w

            #calculate instance discriminitor loss

            self.instance_loss += self.instance_classify_loss_sig_fn(aug_h, self.h, ins_negative_samples) * self.ins_loss_w


        #aug_h = tf.concat(aug_list, axis=0)
        aug_h = tf.add_n(aug_list) / self.augment_num


        #self.norm_loss = -tf.math.reduce_mean(tf.nn.l2_loss(self.std)) * self.norm_loss_w
        #self.norm_loss = -tf.math.reduce_mean(self.std) * self.norm
        self.norm_loss = tf.math.reduce_mean(1.0 - self.std) * self.norm_loss_w

        loss = self.gae_loss + self.aug_gae_loss + self.instance_loss + self.hinge_loss + self.norm_loss
        self.add_loss(loss)

        total_loss = tf.add_n(self.losses)
        #total_loss = self.gae_loss
        loss = (total_loss, self.gae_loss, self.aug_gae_loss, self.instance_loss, None, self.norm_loss)
        #h_list = (self.h, aug_h)

        #return loss, self.h, aug_h, tf.math.reduce_mean(tf.math.reduce_sum(tf.math.exp(self.log_std), axis=-1))
        #return loss, self.h, aug_h, tf.math.reduce_mean(tf.math.reduce_sum(self.std, axis=-1))
        return loss, self.h, aug_h

class FixR(keras.Model):
    """Graph contrastive learning with data augmentation operator learned from dataset"""

    def __init__(self, input_dim, gnn_units_list, sparse_inputs=False, act=tf.nn.relu, dropout=0., bias=False, l2_r=1e-5,
                 negative_num=10, aug_gae_loss_w=1e-5, ins_loss_w=1e-5, learning_rate=0.01, temperature=0.07, augment_num=1,
                 norm_loss_w=-0.1, hinge_loss_w=0.,
                 **kwargs):
        super(FixR, self).__init__(**kwargs)

        self.aug_gae_loss_w = aug_gae_loss_w
        self.ins_loss_w = ins_loss_w
        self.temperature = temperature
        self.augment_num = augment_num

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


    def GAE_loss_fn(self, h, nb_h, negative_samples, nb_size, neg_size):
        pos_h = tf.reshape(tf.tile(h, [1, nb_size]), shape=[h.shape[0], nb_size, h.shape[1]])
        pos = tf.reshape(tf.math.reduce_sum(tf.math.multiply(pos_h, nb_h), axis=2), shape=[-1]) / self.temperature
        neg_h = tf.reshape(tf.tile(h, [1, neg_size]), shape=[h.shape[0], neg_size, h.shape[1]])
        neg = tf.reshape(tf.math.reduce_sum(tf.math.multiply(neg_h, negative_samples), axis=2), shape=[-1]) / self.temperature
        logits = tf.concat((pos, neg), axis=0)

        ones = np.ones(shape=(h.shape[0]*nb_size, ), dtype=np.float64)
        zeros = np.zeros(shape=(h.shape[0]*neg_size, ), dtype=np.float64)
        labels = np.concatenate((ones, zeros), axis=0)
        loss = tf.math.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
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
        x, nbx = inputs
        self.h = self.graph_enc((x, nbx), training)
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
        x, nbx, nbnbx, gradint_dir = inputs
        #x (batch_size, f), nbx (batch_size, nb_size, f), nbnb (batch_size*nb_size, nb_size, f)

        in_dim = nbx.shape[2]
        nb_size = nbx.shape[1]
        batch_size = nbx.shape[0]

        gradint_dir = tf.keras.utils.normalize(gradint_dir)

        #step in trainin

        #get node representation from deep GNN
        self.h = self.graph_enc((x, nbx), training)
        #self.h = tf.keras.utils.normalize(self.h)
        nbx = tf.reshape(nbx, shape=(-1, in_dim))
        self.nb_h = self.graph_enc((nbx, nbnbx), training)
        self.nb_h = tf.reshape(self.nb_h, shape=(batch_size, nb_size, -1))


        negative_index = np.random.randint(low=0, high=batch_size, size=(batch_size, self.negative_num))
        negative_samples = tf.nn.embedding_lookup(params=self.h, ids=negative_index)
        self.gae_loss = self.GAE_loss_fn(self.h, self.nb_h, negative_samples, nb_size, self.negative_num)


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
            self.aug_gae_loss += self.GAE_loss_fn(aug_h, self.nb_h, negative_samples, nb_size, self.negative_num) * self.aug_gae_loss_w

            negative_index = np.random.randint(low=0, high=batch_size, size=(batch_size, self.negative_num))
            negative_samples = tf.nn.embedding_lookup(params=self.h, ids=negative_index)
            #max radius of each node
            #nb_negative_samples = tf.nn.embedding_lookup(params=self.h, ids=nb_negative_index)
            #nb_negative_samples = negative_samples
            #self.norm_loss += self.cal_norm_loss_fn(aug_h, nb_negative_samples, tf.squeeze(tf.nn.embedding_lookup(self.std, nb_negative_index))) * self.norm_loss_w

            #calculate instance discriminitor loss

            self.instance_loss += self.instance_classify_loss_sig_fn(aug_h, self.h, negative_samples) * self.ins_loss_w


        #aug_h = tf.concat(aug_list, axis=0)
        aug_h = tf.add_n(aug_list) / self.augment_num



        loss = self.gae_loss + self.aug_gae_loss + self.instance_loss
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
                 norm_loss_w=-0.1, hinge_loss_w=0.,
                 **kwargs):
        super(AdvR, self).__init__(**kwargs)

        self.aug_gae_loss_w = aug_gae_loss_w
        self.ins_loss_w = ins_loss_w
        self.temperature = temperature
        self.augment_num = augment_num

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


    def GAE_loss_fn(self, h, nb_h, negative_samples, nb_size, neg_size):
        pos_h = tf.reshape(tf.tile(h, [1, nb_size]), shape=[h.shape[0], nb_size, h.shape[1]])
        pos = tf.reshape(tf.math.reduce_sum(tf.math.multiply(pos_h, nb_h), axis=2), shape=[-1]) / self.temperature
        neg_h = tf.reshape(tf.tile(h, [1, neg_size]), shape=[h.shape[0], neg_size, h.shape[1]])
        neg = tf.reshape(tf.math.reduce_sum(tf.math.multiply(neg_h, negative_samples), axis=2), shape=[-1]) / self.temperature
        logits = tf.concat((pos, neg), axis=0)

        ones = np.ones(shape=(h.shape[0]*nb_size, ), dtype=np.float64)
        zeros = np.zeros(shape=(h.shape[0]*neg_size, ), dtype=np.float64)
        labels = np.concatenate((ones, zeros), axis=0)
        loss = tf.math.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
        return loss

    def GAE_adv_loss_fn(self, h, nb_h, negative_samples, gd, nb_gd, std, nb_size, neg_size):
        # h(batch_size, f), nb_h(batch_size, nb_size, f), negative_samples(batch_size, neg_size, f)
        # gd(batch_size, f), nb_gd(batch_size, nb_size, f), std(batch_size, nb_size)

        ones = np.ones(shape=(h.shape[0]*nb_size, ), dtype=np.float64)
        zeros = np.zeros(shape=(h.shape[0]*neg_size, ), dtype=np.float64)
        labels = np.concatenate((ones, zeros), axis=0)

        neg_h = tf.reshape(tf.tile(h, [1, neg_size]), shape=[h.shape[0], neg_size, h.shape[1]])
        neg = tf.reshape(tf.math.reduce_sum(tf.math.multiply(neg_h, negative_samples), axis=2), shape=[-1]) / self.temperature

        h = tf.reshape(tf.tile(h, [1, nb_size]), shape=[h.shape[0], nb_size, h.shape[1]])
        gd = tf.reshape(tf.tile(gd, [1, nb_size]), shape=[gd.shape[0], nb_size, gd.shape[1]])
        std = tf.expand_dims(std, axis=2)
        gd = tf.math.multiply(std, gd)
        nb_gd = tf.math.multiply(std, nb_gd)
        h = h + gd
        nb_h = nb_h + nb_gd
        pos = tf.reshape(tf.math.reduce_sum(tf.math.multiply(h, nb_h), axis=2), shape=[-1]) / self.temperature

        logits = tf.concat((pos, neg), axis=0)

        loss = tf.math.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
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
        x, nbx = inputs
        self.h = self.graph_enc((x, nbx))
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
        x, nbx, nbnbx, gradint_dir, nb_gradint_dir, std = inputs
        #x (batch_size, f), nbx (batch_size, nb_size, f), nbnb (batch_size*nb_size, nb_size, f)
        #std (batch_size, nb_size)

        in_dim = nbx.shape[2]
        nb_size = nbx.shape[1]
        batch_size = nbx.shape[0]

        gradint_dir = tf.keras.utils.normalize(gradint_dir)
        nb_gradint_dir = tf.keras.utils.normalize(nb_gradint_dir)

        #step in trainin

        #get node representation from deep GNN
        self.h = self.graph_enc((x, nbx), training)
        #self.h = tf.keras.utils.normalize(self.h)
        nbx = tf.reshape(nbx, shape=(-1, in_dim))
        self.nb_h = self.graph_enc((nbx, nbnbx), training)
        self.nb_h = tf.reshape(self.nb_h, shape=(batch_size, nb_size, -1))


        negative_index = np.random.randint(low=0, high=batch_size, size=(batch_size, self.negative_num))
        negative_samples = tf.nn.embedding_lookup(params=self.h, ids=negative_index)
        self.gae_loss = self.GAE_loss_fn(self.h, self.nb_h, negative_samples, nb_size, self.negative_num)


        self.std = std
        aug_list = []
        self.aug_gae_loss = 0.
        self.instance_loss = 0.
        self.hinge_loss = 0.
        self.norm_loss = 0.


        for i in range(self.augment_num):
            self.aug_gae_loss += self.GAE_adv_loss_fn(self.h, self.nb_h, negative_samples, gradint_dir, nb_gradint_dir, std, nb_size, self.negative_num) * self.aug_gae_loss_w


        loss = self.gae_loss + self.aug_gae_loss
        self.add_loss(loss)

        total_loss = tf.add_n(self.losses)

        loss = (total_loss, self.gae_loss, self.aug_gae_loss, self.hinge_loss, self.norm_loss)

        return loss, self.h, None


class HyperR(keras.Model):
    """Graph contrastive learning with data augmentation operator learned from dataset"""

    def __init__(self, input_dim, gnn_units_list, sparse_inputs=False, act=tf.nn.relu, dropout=0., bias=False, l2_r=1e-5,
                 negative_num=10, aug_gae_loss_w=1e-5, ins_loss_w=1e-5, learning_rate=0.01, temperature=0.07, augment_num=1,
                 norm_loss_w=-0.1, hinge_loss_w=0.,
                 **kwargs):
        super(HyperR, self).__init__(**kwargs)

        self.aug_gae_loss_w = aug_gae_loss_w
        self.ins_loss_w = ins_loss_w
        self.temperature = temperature
        self.augment_num = augment_num

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


    def GAE_loss_fn(self, h, nb_h, negative_samples, nb_size, neg_size):
        pos_h = tf.reshape(tf.tile(h, [1, nb_size]), shape=[h.shape[0], nb_size, h.shape[1]])
        pos = tf.reshape(tf.math.reduce_sum(tf.math.multiply(pos_h, nb_h), axis=2), shape=[-1]) / self.temperature
        neg_h = tf.reshape(tf.tile(h, [1, neg_size]), shape=[h.shape[0], neg_size, h.shape[1]])
        neg = tf.reshape(tf.math.reduce_sum(tf.math.multiply(neg_h, negative_samples), axis=2), shape=[-1]) / self.temperature
        logits = tf.concat((pos, neg), axis=0)

        ones = np.ones(shape=(h.shape[0]*nb_size, ), dtype=np.float64)
        zeros = np.zeros(shape=(h.shape[0]*neg_size, ), dtype=np.float64)
        labels = np.concatenate((ones, zeros), axis=0)
        loss = tf.math.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
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
        x, nbx = inputs
        self.h = self.graph_enc((x, nbx))
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
        x, nbx, nbnbx, gradint_dir, std = inputs
        #x (batch_size, f), nbx (batch_size, nb_size, f), nbnb (batch_size*nb_size, nb_size, f)
        #gradint_dir (batch_size, f), nb_gradint_dir (batch_size, nb_size, f)
        #std (batch_size,)

        in_dim = nbx.shape[2]
        nb_size = nbx.shape[1]
        batch_size = nbx.shape[0]

        gradint_dir = tf.keras.utils.normalize(gradint_dir)

        #step in trainin

        #get node representation from deep GNN
        self.h = self.graph_enc((x, nbx), training)
        #self.h = tf.keras.utils.normalize(self.h)
        nbx = tf.reshape(nbx, shape=(-1, in_dim))
        self.nb_h = self.graph_enc((nbx, nbnbx), training)
        self.nb_h = tf.reshape(self.nb_h, shape=(batch_size, nb_size, -1))


        negative_index = np.random.randint(low=0, high=batch_size, size=(batch_size, self.negative_num))
        negative_samples = tf.nn.embedding_lookup(params=self.h, ids=negative_index)
        self.gae_loss = self.GAE_loss_fn(self.h, self.nb_h, negative_samples, nb_size, self.negative_num)


        self.std = std
        #self.std = 0.0018
        aug_list = []
        self.aug_gae_loss = 0.
        self.instance_loss = 0.
        self.hinge_loss = 0.
        self.norm_loss = 0.

        #self.std = tf.expand_dims(self.std, axis=1)
        gradint_dir = tf.multiply(gradint_dir, self.std)

        for i in range(self.augment_num):
            aug_h = self.h + gradint_dir
            #aug_h = self.h + tf.keras.utils.normalize(tf.random.normal(shape=[node_num, self.emb_dim])) * self.std
            #aug_h = self.h + tf.random.normal(shape=[node_num, self.emb_dim]) * self.std
            #aug_h = self.h + unit_sphere_sample(shape=[node_num, self.emb_dim]) * self.std

            aug_list.append(aug_h)
            self.aug_gae_loss += self.GAE_loss_fn(aug_h, self.nb_h, negative_samples, nb_size, self.negative_num) * self.aug_gae_loss_w

            ins_negative_index = np.random.randint(low=0, high=batch_size, size=(batch_size, self.negative_num))
            ins_negative_samples = tf.nn.embedding_lookup(params=self.h, ids=ins_negative_index)
            #max radius of each node
            #nb_negative_samples = tf.nn.embedding_lookup(params=self.h, ids=nb_negative_index)
            #nb_negative_samples = negative_samples
            #self.norm_loss += self.cal_norm_loss_fn(aug_h, nb_negative_samples, tf.squeeze(tf.nn.embedding_lookup(self.std, nb_negative_index))) * self.norm_loss_w

            #calculate instance discriminitor loss

            self.instance_loss += self.instance_classify_loss_sig_fn(aug_h, self.h, ins_negative_samples) * self.ins_loss_w


        #aug_h = tf.concat(aug_list, axis=0)
        aug_h = tf.add_n(aug_list) / self.augment_num



        loss = self.gae_loss + self.aug_gae_loss + self.instance_loss
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
                 norm_loss_w=-0.1, hinge_loss_w=0.,
                 **kwargs):
        super(LearnR, self).__init__(**kwargs)

        self.aug_gae_loss_w = aug_gae_loss_w
        self.ins_loss_w = ins_loss_w
        self.temperature = temperature
        self.augment_num = augment_num


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


    def GAE_loss_fn(self, h, nb_h, negative_samples, nb_size, neg_size):
        pos_h = tf.reshape(tf.tile(h, [1, nb_size]), shape=[h.shape[0], nb_size, h.shape[1]])
        pos = tf.reshape(tf.math.reduce_sum(tf.math.multiply(pos_h, nb_h), axis=2), shape=[-1]) / self.temperature
        neg_h = tf.reshape(tf.tile(h, [1, neg_size]), shape=[h.shape[0], neg_size, h.shape[1]])
        neg = tf.reshape(tf.math.reduce_sum(tf.math.multiply(neg_h, negative_samples), axis=2), shape=[-1]) / self.temperature
        logits = tf.concat((pos, neg), axis=0)

        ones = np.ones(shape=(h.shape[0]*nb_size, ), dtype=np.float64)
        zeros = np.zeros(shape=(h.shape[0]*neg_size, ), dtype=np.float64)
        labels = np.concatenate((ones, zeros), axis=0)
        loss = tf.math.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
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
        x, nbx = inputs
        self.h = self.graph_enc((x, nbx))
        return self.h


    def call(self, inputs, training=None):
        x, nbx, nbnbx, gradint_dir = inputs
        #x (batch_size, f), nbx (batch_size, nb_size, f), nbnb (batch_size*nb_size, nb_size, f)
        #gradint_dir (batch_size, f), nb_gradint_dir (batch_size, nb_size, f)
        #std (batch_size,)

        in_dim = nbx.shape[2]
        nb_size = nbx.shape[1]
        batch_size = nbx.shape[0]

        gradint_dir = tf.keras.utils.normalize(gradint_dir)

        #step in trainin

        #get node representation from deep GNN
        self.h = self.graph_enc((x, nbx), training)
        #self.h = tf.keras.utils.normalize(self.h)
        nbx = tf.reshape(nbx, shape=(-1, in_dim))
        self.nb_h = self.graph_enc((nbx, nbnbx), training)
        self.nb_h = tf.reshape(self.nb_h, shape=(batch_size, nb_size, -1))


        negative_index = np.random.randint(low=0, high=batch_size, size=(batch_size, self.negative_num))
        negative_samples = tf.nn.embedding_lookup(params=self.h, ids=negative_index)
        self.gae_loss = self.GAE_loss_fn(self.h, self.nb_h, negative_samples, nb_size, self.negative_num)


        self.std = self.learn_radius(self.h, batch_size)
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
            self.aug_gae_loss += self.GAE_loss_fn(aug_h, self.nb_h, negative_samples, nb_size, self.negative_num) * self.aug_gae_loss_w

            ins_negative_index = np.random.randint(low=0, high=batch_size, size=(batch_size, self.negative_num))
            ins_negative_samples = tf.nn.embedding_lookup(params=self.h, ids=ins_negative_index)
            #max radius of each node
            #nb_negative_samples = tf.nn.embedding_lookup(params=self.h, ids=nb_negative_index)
            #nb_negative_samples = negative_samples
            #self.norm_loss += self.cal_norm_loss_fn(aug_h, nb_negative_samples, tf.squeeze(tf.nn.embedding_lookup(self.std, nb_negative_index))) * self.norm_loss_w

            #calculate instance discriminitor loss

            self.instance_loss += self.instance_classify_loss_sig_fn(aug_h, self.h, ins_negative_samples) * self.ins_loss_w


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
                 norm_loss_w=-0.1, hinge_loss_w=0.,
                 **kwargs):
        super(LearnSimpleR, self).__init__(**kwargs)

        self.aug_gae_loss_w = aug_gae_loss_w
        self.ins_loss_w = ins_loss_w
        self.temperature = temperature
        self.augment_num = augment_num

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


    def GAE_loss_fn(self, h, nb_h, negative_samples, nb_size, neg_size):
        pos_h = tf.reshape(tf.tile(h, [1, nb_size]), shape=[h.shape[0], nb_size, h.shape[1]])
        pos = tf.reshape(tf.math.reduce_sum(tf.math.multiply(pos_h, nb_h), axis=2), shape=[-1]) / self.temperature
        neg_h = tf.reshape(tf.tile(h, [1, neg_size]), shape=[h.shape[0], neg_size, h.shape[1]])
        neg = tf.reshape(tf.math.reduce_sum(tf.math.multiply(neg_h, negative_samples), axis=2), shape=[-1]) / self.temperature
        logits = tf.concat((pos, neg), axis=0)

        ones = np.ones(shape=(h.shape[0]*nb_size, ), dtype=np.float64)
        zeros = np.zeros(shape=(h.shape[0]*neg_size, ), dtype=np.float64)
        labels = np.concatenate((ones, zeros), axis=0)
        loss = tf.math.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
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
        x, nbx = inputs
        self.h = self.graph_enc((x, nbx))
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

    def learn_simple(self, h):
        S = tf.math.sigmoid(tf.matmul(h, tf.transpose(h)))
        r = tf.math.reduce_max(1.0 - S, axis=1)
        r = tf.reshape(r, shape=(-1, 1))
        return r

    def call(self, inputs, training=None):
        x, nbx, nbnbx, gradint_dir = inputs
        #x (batch_size, f), nbx (batch_size, nb_size, f), nbnb (batch_size*nb_size, nb_size, f)
        #gradint_dir (batch_size, f), nb_gradint_dir (batch_size, nb_size, f)
        #std (batch_size,)

        in_dim = nbx.shape[2]
        nb_size = nbx.shape[1]
        batch_size = nbx.shape[0]

        gradint_dir = tf.keras.utils.normalize(gradint_dir)

        #step in trainin

        #get node representation from deep GNN
        self.h = self.graph_enc((x, nbx), training)
        #self.h = tf.keras.utils.normalize(self.h)
        nbx = tf.reshape(nbx, shape=(-1, in_dim))
        self.nb_h = self.graph_enc((nbx, nbnbx), training)
        self.nb_h = tf.reshape(self.nb_h, shape=(batch_size, nb_size, -1))


        negative_index = np.random.randint(low=0, high=batch_size, size=(batch_size, self.negative_num))
        negative_samples = tf.nn.embedding_lookup(params=self.h, ids=negative_index)
        self.gae_loss = self.GAE_loss_fn(self.h, self.nb_h, negative_samples, nb_size, self.negative_num)


        self.std = self.learn_simple(self.h)
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
            self.aug_gae_loss += self.GAE_loss_fn(aug_h, self.nb_h, negative_samples, nb_size, self.negative_num) * self.aug_gae_loss_w

            ins_negative_index = np.random.randint(low=0, high=batch_size, size=(batch_size, self.negative_num))
            ins_negative_samples = tf.nn.embedding_lookup(params=self.h, ids=negative_index)
            #max radius of each node
            #nb_negative_samples = tf.nn.embedding_lookup(params=self.h, ids=nb_negative_index)
            #nb_negative_samples = negative_samples
            #self.norm_loss += self.cal_norm_loss_fn(aug_h, nb_negative_samples, tf.squeeze(tf.nn.embedding_lookup(self.std, nb_negative_index))) * self.norm_loss_w

            #calculate instance discriminitor loss

            self.instance_loss += self.instance_classify_loss_sig_fn(aug_h, self.h, ins_negative_samples) * self.ins_loss_w


        #aug_h = tf.concat(aug_list, axis=0)
        aug_h = tf.add_n(aug_list) / self.augment_num


        #self.norm_loss = -tf.math.reduce_mean(tf.nn.l2_loss(self.std)) * self.norm_loss_w
        #self.norm_loss = -tf.math.reduce_mean(self.std) * self.norm
        self.norm_loss = tf.math.reduce_mean(1.0 - self.std) * self.norm_loss_w

        loss = self.gae_loss + self.aug_gae_loss + self.instance_loss + self.norm_loss
        self.add_loss(loss)

        total_loss = tf.add_n(self.losses)
        #total_loss = self.gae_loss
        loss = (total_loss, self.gae_loss, self.aug_gae_loss, self.instance_loss, self.hinge_loss, self.norm_loss)
        #h_list = (self.h, aug_h)

        #return loss, self.h, aug_h, tf.math.reduce_mean(tf.math.reduce_sum(tf.math.exp(self.log_std), axis=-1))
        #return loss, self.h, aug_h, tf.math.reduce_mean(tf.math.reduce_sum(self.std, axis=-1))
        return loss, self.h, aug_h

class GCN(keras.Model):
    """Graph contrastive learning with data augmentation operator learned from dataset"""

    def __init__(self, input_dim, hidden_dim, output_dim, sparse_inputs=False, act=tf.nn.relu, dropout=0., bias=False, l2_r=1e-5,
                 learning_rate=0.01, multi_flag=False,
                 **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.sparse_inputs = sparse_inputs
        self.multi_flag = multi_flag

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.dropout = dropout
        self.l2_r = l2_r
        self.bias = bias
        self.act = act

        self.graph_enc1 = BatchGraphConvolution(input_dim=input_dim,
                                                     output_dim=hidden_dim,
                                                     dropout=self.dropout,
                                                     sparse_inputs=sparse_inputs,
                                                     l2_r=self.l2_r,
                                                     bias=self.bias,
                                                     act=self.act)

        self.graph_enc2 = BatchGraphConvolution(input_dim=hidden_dim,
                                                     output_dim=output_dim,
                                                     dropout=self.dropout,
                                                     sparse_inputs=False,
                                                     l2_r=self.l2_r,
                                                     bias=self.bias,
                                                     act=lambda x:x)



        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        #self.optimizer = tf.keras.optimizer.RMSprop(learning_rate=self.learning_rate)


    def run_embedding(self, inputs):
        x, nbx = inputs
        self.h = self.graph_enc1((x, nbx))
        return self.h


    def call(self, inputs, training=None):
        x, nbx, nbnbx, labels = inputs
        #x (batch_size, f), nbx (batch_size, nb_size, f), nbnb (batch_size*nb_size, nb_size, f)
        #gradint_dir (batch_size, f), nb_gradint_dir (batch_size, nb_size, f)
        #std (batch_size,)

        in_dim = nbx.shape[2]
        nb_size = nbx.shape[1]
        batch_size = nbx.shape[0]

        #step in trainin

        #get node representation from deep GNN
        self.h = self.graph_enc1((x, nbx), training)
        #self.h = tf.keras.utils.normalize(self.h)
        nbx = tf.reshape(nbx, shape=(-1, in_dim))
        self.nb_h = self.graph_enc1((nbx, nbnbx), training)
        self.nb_h = tf.reshape(self.nb_h, shape=(batch_size, nb_size, -1))

        self.logits = self.graph_enc2((self.h, self.nb_h), training)

        if self.multi_flag:
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=tf.cast(labels, tf.float64))
            loss = tf.reduce_mean(tf.reduce_sum(cross_entropy, axis=1))
            acc = 0.
            y_pred = tf.math.sigmoid(self.logits)
        else:
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=labels))
            acc = tf.math.reduce_mean(tf.cast(tf.math.equal(tf.argmax(self.logits, 1),tf.argmax(labels, 1)), dtype=tf.float64))

            y_pred = tf.nn.softmax(self.logits)

        self.add_loss(loss)
        total_loss = tf.add_n(self.losses)




        return total_loss, acc, y_pred

class MLP(keras.Model):
    """Graph contrastive learning with data augmentation operator learned from dataset"""

    def __init__(self, input_dim, hidden_dim, output_dim, sparse_inputs=False, act=tf.nn.relu, dropout=0., bias=False, l2_r=1e-5,
                 learning_rate=0.01, multi_flag=False,
                 **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.sparse_inputs = sparse_inputs
        self.multi_flag = multi_flag

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.dropout = dropout
        self.l2_r = l2_r
        self.bias = bias
        self.act = act

        self.graph_enc1 = Dense(input_dim=input_dim,
                                                     output_dim=hidden_dim,
                                                     dropout=self.dropout,
                                                     sparse_inputs=sparse_inputs,
                                                     #l2_r=self.l2_r,
                                                     bias=self.bias,
                                                     act=self.act)

        self.graph_enc2 = Dense(input_dim=hidden_dim,
                                                     output_dim=output_dim,
                                                     dropout=self.dropout,
                                                     sparse_inputs=False,
                                                     #l2_r=self.l2_r,
                                                     bias=self.bias,
                                                     act=lambda x:x)



        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        #self.optimizer = tf.keras.optimizer.RMSprop(learning_rate=self.learning_rate)


    def run_embedding(self, inputs):
        x = inputs
        self.h = self.graph_enc1(x)
        return self.h


    def call(self, inputs, training=None):
        x, labels = inputs
        #x (batch_size, f)
        #gradint_dir (batch_size, f), nb_gradint_dir (batch_size, nb_size, f)
        #std (batch_size,)
        #step in trainin

        #get node representation from deep GNN
        self.h = self.graph_enc1(x, training)
        self.logits = self.graph_enc2(self.h, training)

        if self.multi_flag:
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=tf.cast(labels, tf.float64))
            loss = tf.reduce_mean(tf.reduce_sum(cross_entropy, axis=1))
            acc = 0.
            y_pred = tf.math.sigmoid(self.logits)
        else:
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=labels))
            acc = tf.math.reduce_mean(tf.cast(tf.math.equal(tf.argmax(self.logits, 1),tf.argmax(labels, 1)), dtype=tf.float64))

            y_pred = tf.nn.softmax(self.logits)

        self.add_loss(loss)
        total_loss = tf.add_n(self.losses)

        return total_loss, acc, y_pred

