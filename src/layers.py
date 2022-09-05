import tensorflow as tf
from tensorflow import keras


def my_sparse_add_n(x_list):
    x = x_list[0]
    for y in x_list[1:]:
        x = tf.sparse_add(x, y)
    return x


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse.sparse_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res

def dropout_sparse(x, keep_prob, num_nonzero_elems):
    """Dropout for sparse tensors. Currently fails for very large sparse tensors (>1M elements)
    """
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


class Dense(keras.layers.Layer):
    """Dense layer. suitable for both dense and sparse input"""
    def __init__(self, input_dim, output_dim, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        self.dropout = dropout
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.bias = bias

        # helper variable for sparse dropout

        self.w = self.add_weight(
                       shape=(input_dim, output_dim),
                       initializer=tf.keras.initializers.GlorotUniform(),
                       trainable=True
            )

        if self.bias:
            self.b = self.add_weight(
                            shape=(output_dim,),
                            initializer="zeros",
                            trainable=True
                )


    def call(self, inputs, training=None):
        x = inputs
        # dropout
        if training:
            if self.sparse_inputs:
                x = tf.SparseTensor(indices=x.indices,
                        values=tf.nn.dropout(x.values, self.dropout),
                        dense_shape=x.dense_shape)
            else:
                x = tf.nn.dropout(x, self.dropout)
        # transform
        output = dot(x, self.w, sparse=self.sparse_inputs)
        # bias
        if self.bias:
            output += self.b

        return self.act(output)

    def get_config(self):
        config = super(Linear, self).get_config()
        config.update({"input_dim":self.input_dim,
                      "output_dim":self.output_dim,
                      "sparse_inputs":self.sparse_inputs,
                      "dropout":self.dropout,
                      "bias":self.bias})
        return config


class GraphConvolution(keras.layers.Layer):
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim, dropout=0., sparse_inputs=False,
                 l2_r=0.0, bias=False, act=tf.nn.relu, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        self.w = self.add_weight(
                       shape=(input_dim, output_dim),
                       initializer=tf.keras.initializers.GlorotUniform(),
                       trainable=True
            )
        if bias:
            self.b = self.add_weight(
                            shape=(output_dim,),
                            initializer="zeros",
                            trainable=True
                )
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.bias = bias
        #self.act = act
        self.act = tf.keras.layers.PReLU()
        self.sparse_inputs = sparse_inputs
        self.l2_r = l2_r
        self.regularizers = keras.regularizers.L2(l2_r)

    def call(self, inputs, training=None):
        x, adj = inputs

        if self.sparse_inputs:
            if training:
                x = tf.SparseTensor(indices=x.indices,
                        values=tf.nn.dropout(x.values, self.dropout),
                        dense_shape=x.dense_shape)
            x = tf.sparse.sparse_dense_matmul(x, self.w)
        else:
            if training:
                x = tf.nn.dropout(x, self.dropout)
            x = tf.matmul(x, self.w)

        x = tf.sparse.sparse_dense_matmul(adj, x)
        if self.bias:
            x += self.b
        outputs = self.act(x)

        #add l2 regularization
        self.add_loss(self.regularizers(self.w))
        return outputs

    def get_config(self):
        config = super(Linear, self).get_config()
        config.update({"input_dim":self.input_dim,
                      "output_dim":self.output_dim,
                      "sparse_inputs":self.sparse_inputs,
                      "dropout":self.dropout,
                      "bias":self.bias,
                      "l2_r":self.l2_r})
        return config

class BatchGraphConvolution(keras.layers.Layer):
    """Basic graph convolution layer for undirected graph without edge labels."""
    """Only Support Dense Feature !!"""
    def __init__(self, input_dim, output_dim, dropout=0., sparse_inputs=False,
                 l2_r=0.0, bias=False, act=tf.nn.relu, **kwargs):
        super(BatchGraphConvolution, self).__init__(**kwargs)
        self.w = self.add_weight(
                       shape=(input_dim, output_dim),
                       initializer=tf.keras.initializers.GlorotUniform(),
                       trainable=True
            )
        if bias:
            self.b = self.add_weight(
                            shape=(output_dim,),
                            initializer="zeros",
                            trainable=True
                )
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.bias = bias
        #self.act = act
        self.act = tf.keras.layers.PReLU()
        self.sparse_inputs = sparse_inputs
        self.l2_r = l2_r
        self.regularizers = keras.regularizers.L2(l2_r)

    def call(self, inputs, training=None):
        x, nbx = inputs

        batch_size, nb_size, f_size = nbx.shape
        nbx = tf.reshape(nbx, shape=(-1, f_size))

        if training:
            x = tf.nn.dropout(x, self.dropout)
            nbx = tf.nn.dropout(nbx, self.dropout)
        x = tf.matmul(x, self.w)
        nbx = tf.matmul(nbx, self.w)

        if self.bias:
            x += self.b
            nbx += self.b

        nbx = tf.reshape(nbx, shape=(batch_size, nb_size, self.output_dim))
        nbx = tf.reduce_sum(nbx, axis=1)

        x = x + nbx
        outputs = self.act(x)

        #add l2 regularization
        self.add_loss(self.regularizers(self.w))
        return outputs

    def get_config(self):
        config = super(Linear, self).get_config()
        config.update({"input_dim":self.input_dim,
                      "output_dim":self.output_dim,
                      "sparse_inputs":self.sparse_inputs,
                      "dropout":self.dropout,
                      "bias":self.bias,
                      "l2_r":self.l2_r})
        return config


class GraphAttention(keras.layers.Layer):
    def __init__(self, input_dim, output_dim, n_heads,
                dropout=0., in_drop=0.0, coef_drop=0.0, act=tf.nn.elu,
                sparse_inputs=False, bias=False, adj_mat_sparse=False,
                head_agg_type='mean', share_weight='independent',
                reverse_axis=1, att_key_type='identity',
                **kwargs):
        super(GraphAttention, self).__init__(**kwargs)

        #self.config = config

        self.n_heads = n_heads
        self.in_drop = in_drop
        self.coef_drop = coef_drop
        self.dropout = dropout
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act = act
        self.sparse_inputs = sparse_inputs
        self.bias = bias

        self.adj_mat_sparse = adj_mat_sparse
        self.head_agg_type = head_agg_type
        self.share_weight = share_weight

        self.reverse_axis = reverse_axis

        self.att_key_type = att_key_type

    def attention_coef_reweight(self, f1, f2, x, coef_drop, in_drop, sparse=False, training=None):
        if sparse:
            logits = tf.sparse_add(f1, f2)
            lrelu = tf.SparseTensor(indices=logits.indices,
                    values=tf.nn.leaky_relu(logits.values),
                    dense_shape=logits.dense_shape)
            if self.reverse_axis == 1:
                coefs = tf.sparse_softmax(lrelu)
            else:
                coefs = tf.sparse_transpose(tf.sparse_softmax(tf.sparse_transpose(lrelu)))
            if training:
                coefs = tf.SparseTensor(indices=coefs.indices,
                        values=tf.nn.dropout(coefs.values, coef_drop),
                        dense_shape=coefs.dense_shape)
                x = tf.nn.dropout(x, 1.0 - in_drop)
            vals = tf.sparse.sparse_dense_matmul(coefs, x)
        else:
            logits = f1 + f2
            coefs = tf.nn.softmax(tf.nn.leaky_relu(logits), axis=self.reverse_axis)
            if training:
                coefs = tf.nn.dropout(coefs, coef_drop)
                x = tf.nn.dropout(x, in_drop)
            vals = tf.matmul(coefs, x)
        return vals

    def calculate_attention(self, x, input_dim, adj_mat, sparse_inputs, act, bias, att_key_type='identity', training=None):
        if att_key_type == 'identity':
            key = x
        elif att_key_type == 'sum':
            key = tf.sparse.sparse_dense_matmul(adj_mat, x)
        elif att_key_type == 'mean':
            key = tf.sparse.sparse_dense_matmul(adj_mat, x) / tf.sparse_reduce_sum(adj_mat, axis=1, keep_dims=True)
        elif att_key_type == 'sig':
            key = tf.sigmoid(tf.sparse.sparse_dense_matmul(adj_mat, x))
        f1 = Dense(input_dim, 1, 0., sparse_inputs=sparse_inputs,
             act=act, bias=bias)(key, training)
        f2 = Dense(input_dim, 1, 0., sparse_inputs=sparse_inputs,
             act=act, bias=bias)(x, training)
        f1 = adj_mat * f1
        f2 = adj_mat * tf.transpose(f2, [1, 0])

        vals = self.attention_coef_reweight(f1, f2, x, self.coef_drop, self.in_drop, sparse=self.adj_mat_sparse, training=training)

        ret = tf.contrib.layers.bias_add(vals)
        #TODO add residual connection

        return self.act(ret)


    def call(self, inputs, adj_mat, training=None):
        self.adj_mat = adj_mat

        attns = []

        if self.share_weight == 'share':
            x = Dense(self.input_dim, self.output_dim, dropout=self.dropout, sparse_inputs=self.sparse_inputs,
                 act=lambda x:x, bias=False)(inputs, training)
            for i in range(self.n_heads):
                #support_value = []
                #for i in range(len(self.adj_mat)):
                #    support_value.append(self.calculate_attention(x=x, input_dim=self.output_dim, adj_mat=self.adj_mat[i],
                #                        placeholders=self.placeholders, sparse_inputs=False, act=self.act, bias=False))
                #support_result = tf.add_n(support_value)
                support_result = self.calculate_attention(x=x, input_dim=self.output_dim, adj_mat=self.adj_mat,
                                         sparse_inputs=False, act=self.act, bias=False, att_key_type=self.att_key_type, training=training)
                attns.append(support_result)

        elif self.share_weight == 'independent':
            for i in range(self.n_heads):
                x = Dense(self.input_dim, self.output_dim, self.placeholders, dropout=self.dropout, sparse_inputs=self.sparse_inputs,
                     act=lambda x:x, bias=False)(inputs, training)
                #support_value = []
                #for i in range(len(self.adj_mat)):
                #    support_value.append(self.calculate_attention(x=x, input_dim=self.output_dim, adj_mat=self.adj_mat[i],
                #                        placeholders=self.placeholders, sparse_inputs=False, act=self.act, bias=False))
                #support_result = tf.add_n(support_value)
                support_result = self.calculate_attention(x=x, input_dim=self.output_dim, adj_mat=self.adj_mat,
                                        sparse_inputs=False, act=self.act, bias=False, att_key_type=self.att_key_type, training=training)
                attns.append(support_result)

        if self.head_agg_type == 'mean':
            res = tf.add_n(attns) / self.n_heads
        else:
            res = tf.concat(attns, axis=-1)

        return res

    def get_config(self):
        config = super(Linear, self).get_config()
        config.update({"input_dim":self.input_dim,
                      "output_dim":self.output_dim,
                      "n_heads":n_heads,
                      "sparse_inputs":self.sparse_inputs,
                      "adj_mat_sparse":adj_mat_sparse,
                      "bias":bias,
                      "dropout":self.dropout,
                      "in_drop":in_drop,
                      "coef_drop":coef_drop,
                      "head_agg_type":head_agg_type,
                      "share_weight":share_weight,
                      "reverse_axis":reverse_axis,
                      "att_key_type":att_key_type}
                      )
        return config




class InnerProductDecoder(keras.layers.Layer):
    """Decoder model layer for link prediction."""
    def __init__(self, input_dim, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(InnerProductDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act

    def __call__(self, inputs, training=None):
        x = tf.transpose(inputs)
        if training:
            inputs = tf.nn.dropout(inputs, self.dropout)
            x = tf.nn.dropout(x, self.dropout)
        x = tf.matmul(inputs, x)
        x = tf.reshape(x, [-1])
        outputs = self.act(x)
        return outputs



