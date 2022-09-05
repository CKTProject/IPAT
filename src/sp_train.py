from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from utils import *
from models import GCN, MLP, SAGNN
import scipy.sparse as sp

import sys

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'sagnn', 'Model string.')  # 'gcn', 'sagnn', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 300, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

#train ratio in supervised setting
#flags.DEFINE_float('train_ratio', 0.8, 'label ratio in trainning.')
#supervised data split
#adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data_supervised(FLAGS.dataset, FLAGS.train_ratio)


# Load data
#semi supervised data split
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_gcn_data(FLAGS.dataset)

adj = np.ones(adj.shape) / adj.shape[0] + adj
#adj = np.ones(adj.shape) + adj

edge_num = adj.nonzero()

# Some preprocessing
features = preprocess_features(features)
if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
elif FLAGS.model == 'sagnn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = SAGNN
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Create model
if model_func == 'sagnn':
    model = SAGNN(placeholders, edge_num=edge_num, input_dim=features[2][1], logging=True)
else:
    model = model_func(placeholders, input_dim=features[2][1], logging=True)

# Initialize session
sess = tf.Session()


# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)

def check_wrong(features, support, labels, mask, placeholders):
    id_array = np.array(list(range(labels.shape[0])))
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.correct, model.wrong], feed_dict=feed_dict_val)
    correct_idx = id_array[outs_val[0] > 0]
    error_idx = id_array[outs_val[1] > 0]
    correct_y = outs_val[0][outs_val[0] > 0]
    error_y = outs_val[1][outs_val[1] > 0]
    return correct_idx, error_idx, correct_y, error_y


saver = tf.train.Saver()
# Init variables
sess.run(tf.global_variables_initializer())

best_val_loss = None
best_val_acc = 0.0
cur_stop_iter = 0
#cost_val = []

# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_gcn_feed_dict(features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

    # Validation
    cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
    #cost_val.append(cost)
    if not best_val_loss:
        best_val_loss = cost
    if cost <= best_val_loss and acc > best_val_acc:
        best_val_loss = cost
        best_val_acc = acc
        save_path = saver.save(sess, "/tmp/sagnn.ckpt")

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

    #if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
    #    print("Early stopping...")
    #    break

print("Optimization Finished!")

tf.reset_default_graph()
saver.restore(sess, "/tmp/sagnn.ckpt")
print("Reload model end")
print("record val loss %.5f and acc %.f"%(best_val_acc, best_val_acc))
# Testing
cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
print("Reload model Val results:", "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

#correct_idx, error_idx, correct_y, error_y = check_wrong(features, support, y_test, test_mask, placeholders)
#with open("temp/gcn_case_study_%s.pkl"%(FLAGS.dataset), 'wb') as fout:
#    pkl.dump([correct_idx, error_idx, correct_y, error_y], fout)
