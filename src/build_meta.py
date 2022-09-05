import utils
from utils import Graphdata
import numpy as np

dataset = 'citeseer'
gdata = Graphdata()
gdata.load_gcn_data(dataset, fix_node_test=True, node_label=True)

X, Y = gdata.get_node_classification_data('all')
y = np.argmax(Y, axis=1)

out_file = 'visualization/meta_data/%s_meta.tsv'%dataset
utils.save_embedding(out_file, y, delimiter='\n', header=None)